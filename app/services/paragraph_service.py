import logging
from typing import List
from sqlalchemy import text
from app.models.paragraph import Paragraph
from app.utils.embed import get_embedding
from app.utils.reranker import rerank
from app.core.database import engine

logger = logging.getLogger(__name__)

def _parse_embedding(emb_str) -> list:
    """Parse pgvector text representation '[-0.037,-0.021,...]' into a float list."""
    if emb_str is None:
        return None
    try:
        cleaned = emb_str.strip("[]")
        return [float(x) for x in cleaned.split(",")]
    except Exception:
        return None


def hybrid_search(
    query: str,
    top_k: int = 5,
    alpha: float = 0.5,
    min_score: float = 0.0,
    use_reranker: bool = True,
) -> List[Paragraph]:
    query_embedding = get_embedding(query)
    vector_literal = "[" + ",".join(str(x) for x in query_embedding) + "]"

    sql_query = text(f"""
        WITH semantic_search AS (
            SELECT id, 1 - (embedding <=> '{vector_literal}'::vector) AS semantic_score
            FROM paragraph
            ORDER BY embedding <=> '{vector_literal}'::vector
            LIMIT 100
        ),
        keyword_search AS (
            SELECT id, ts_rank(fts_vector, websearch_to_tsquery('english', :query)) AS keyword_score
            FROM paragraph
            WHERE fts_vector @@ websearch_to_tsquery('english', :query)
            ORDER BY keyword_score DESC
            LIMIT 100
        )
        SELECT
            p.id,
            p.section_id,
            p.content,
            p.metadata,
            p.embedding::text AS embedding_str,
            (:alpha * COALESCE(ss.semantic_score, 0.0) +
             (1.0 - :alpha) * COALESCE(ks.keyword_score, 0.0)) AS score
        FROM paragraph p
        LEFT JOIN semantic_search ss ON p.id = ss.id
        LEFT JOIN keyword_search ks ON p.id = ks.id
        WHERE ss.id IS NOT NULL OR ks.id IS NOT NULL
        ORDER BY score DESC
        LIMIT :top_k;
    """)

    with engine.connect() as conn:
        rows = conn.execute(
            sql_query,
            {"query": query, "alpha": alpha, "top_k": top_k}
        ).fetchall()

        paragraphs = []
        for row in rows:
            row_dict = dict(row._mapping)
            score = row_dict.pop('score', 0.0)
            if score < min_score:
                continue
            embedding_list = _parse_embedding(row_dict.pop('embedding_str', None))
            para = Paragraph(
                id=row_dict["id"],
                section_id=row_dict["section_id"],
                content=row_dict["content"],
                meta=row_dict.get("metadata"),
                embedding=None,
            )
            para._score = score
            para._embedding_list = embedding_list
            paragraphs.append(para)

    if not paragraphs:
        return []

    if use_reranker:
        candidates = [
            {"id": p.id, "content": p.content, "metadata": p.meta, "_score": getattr(p, "_score", 0)}
            for p in paragraphs
        ]
        reranked = rerank(query, candidates, top_k=top_k)
        reranked_ids = {r["id"] for r in reranked}
        paragraphs = [p for p in paragraphs if p.id in reranked_ids]

    return paragraphs[:top_k]
