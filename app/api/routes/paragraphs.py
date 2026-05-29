from fastapi import APIRouter, Query
from app.services.paragraph_service import hybrid_search

router = APIRouter()

@router.get("/search")
def search_paragraphs(
    query: str = Query(..., min_length=1, max_length=500), 
    top_k: int = Query(5, ge=1, le=20),
    alpha: float = Query(0.5, ge=0.0, le=1.0)
):
    paragraphs = hybrid_search(query=query, top_k=top_k, alpha=alpha)
    return {
        "query": query,
        "alpha": alpha,
        "results": [
            {
                "id": p.id,
                "content": p.content,
                "section_id": p.section_id,
                "metadata": p.meta
            } for p in paragraphs
        ]
    }
