from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
import logging

from app.services.flashcard_service import (
    retrieve_paragraphs,
    generate_flashcards
)
from app.schemas.flashcard import (
    FlashcardRequest,
    FlashcardResponse,
    RetrievalResponse
)

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/retrieve", response_model=RetrievalResponse)
async def retrieve_for_query(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(5, ge=1, le=20, description="Number of paragraphs to retrieve")
):
    """
    Retrieve paragraphs matching a query using semantic search
    
    **Example:**
    ```
    POST /api/v1/flashcards/retrieve?query=photosynthesis&top_k=5
    ```
    """
    try:
        paragraphs = retrieve_paragraphs(query, top_k=top_k)
        
        return RetrievalResponse(
            query=query,
            paragraphs=paragraphs,
            total=len(paragraphs)
        )
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval failed: {str(e)}"
        )

@router.post("/generate", response_model=FlashcardResponse)
async def generate_flashcards_endpoint(request: FlashcardRequest):
    """
    Generate flashcards for a given query
    
    1. Retrieves relevant paragraphs using semantic search
    2. Sends to LLM to generate flashcards
    3. Returns formatted flashcards
    
    **Example:**
    ```json
    {
        "query": "photosynthesis",
        "top_k": 5,
        "num_flashcards": 5,
        "difficulty": "mixed"
    }
    ```
    """
    try:
        result = generate_flashcards(
            query=request.query,
            top_k=request.top_k,
            num_flashcards=request.num_flashcards,
            difficulty_level=request.difficulty or "mixed"
        )
        return result
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Flashcard generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Flashcard generation failed: {str(e)}"
        )

@router.get("/health")
def flashcard_health():
    """Health check endpoint"""
    return {"status": "ok", "service": "flashcards"}
