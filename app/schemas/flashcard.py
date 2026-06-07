from pydantic import BaseModel
from typing import List, Optional

class FlashcardRequest(BaseModel):
    """Request to generate flashcards"""
    query: str  # Search query (e.g., "photosynthesis")
    top_k: int = 5  # Number of paragraphs to retrieve
    num_flashcards: int = 5  # Number of flashcards to generate
    difficulty: Optional[str] = None  # e.g., "easy", "medium", "hard" (optional filter)
    min_score: Optional[float] = None  # Minimum relevance score threshold
    use_reranker: Optional[bool] = None  # Whether to use cross-encoder re-ranking
    use_mmr: Optional[bool] = None  # Whether to apply MMR diversity

class Flashcard(BaseModel):
    """Single flashcard wcith question and answer"""
    question: str
    answer: str
    difficulty: str  # "easy", "medium", "hard"
    hint: Optional[str] = None  # Optional hint for the question

class FlashcardResponse(BaseModel):
    """Response with generated flashcards"""
    query: str
    flashcards: List[Flashcard]
    source_paragraphs: int  # How many paragraphs were used

class RetrievalResponse(BaseModel):
    """Raw retrieval response (before flashcard generation)"""
    query: str
    paragraphs: List[dict]  # List of retrieved paragraphs
    total: int  # Total paragraphs retrieved
