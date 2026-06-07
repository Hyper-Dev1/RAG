from fastapi import APIRouter, Query
from app.services.paragraph_service import hybrid_search

router = APIRouter()

@router.get("/search")
def search_paragraphs(
    query: str = Query(..., min_length=1, max_length=500), 
    top_k: int = Query(5, ge=1, le=20),
    alpha: float = Query(0.5, ge=0.0, le=1.0),
    min_score: float = Query(0.0, ge=0.0, le=1.0),
    use_reranker: bool = Query(True),
):
    paragraphs = hybrid_search(query=query, top_k=top_k, alpha=alpha, min_score=min_score, use_reranker=use_reranker)
    return {
        "query": query,
        "alpha": alpha,
        "min_score": min_score,
        "use_reranker": use_reranker,
        "results": [
            {
                "id": p.id,
                "content": p.content,
                "section_id": p.section_id,
                "metadata": p.meta,
                "score": getattr(p, "_score", None),
            } for p in paragraphs
        ]
    }
