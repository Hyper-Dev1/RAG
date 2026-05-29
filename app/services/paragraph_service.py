from typing import List
from sqlmodel import Session
from sqlalchemy import text
from app.models.paragraph import Paragraph
from app.utils.embed import get_embedding
from app.core.database import engine

def hybrid_search(query: str, top_k: int = 5, alpha: float = 0.5) -> List[Paragraph]:
    query_embedding = get_embedding(query)
    # pgvector expects string representation like '[0.1, 0.2, ...]'
    query_embedding_str = str(query_embedding)

    sql_query = text("""
        WITH semantic_search AS (
            SELECT id, RANK() OVER (ORDER BY embedding <=> :query_embedding) AS rank
            FROM paragraph
            ORDER BY embedding <=> :query_embedding
            LIMIT 100
        ),
        keyword_search AS (
            SELECT id, RANK() OVER (ORDER BY ts_rank(fts_vector, plainto_tsquery('english', :query)) DESC) AS rank
            FROM paragraph
            WHERE fts_vector @@ plainto_tsquery('english', :query)
            ORDER BY ts_rank(fts_vector, plainto_tsquery('english', :query)) DESC
            LIMIT 100
        )
        SELECT p.*,
               (COALESCE(:alpha / (60 + ss.rank), 0.0) +
                COALESCE((1.0 - :alpha) / (60 + ks.rank), 0.0)) AS score
        FROM paragraph p
        LEFT JOIN semantic_search ss ON p.id = ss.id
        LEFT JOIN keyword_search ks ON p.id = ks.id
        WHERE ss.id IS NOT NULL OR ks.id IS NOT NULL
        ORDER BY score DESC
        LIMIT :top_k;
    """)

    with Session(engine) as session:
        statement = sql_query.bindparams(
            query=query,
            query_embedding=query_embedding_str,
            alpha=alpha,
            top_k=top_k
        )
        results = session.exec(statement).all()
        
        # results are SQLAlchemy Row objects. Map them back to Paragraph objects.
        paragraphs = []
        for row in results:
            row_dict = dict(row._mapping)
            # Remove the 'score' column as it's not part of the Paragraph model
            row_dict.pop('score', None)
            paragraphs.append(Paragraph(**row_dict))

    return paragraphs
