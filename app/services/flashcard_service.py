import json
import re
import logging
from typing import List
from app.services.paragraph_service import hybrid_search
from app.utils.llm import ollama_chat
from app.utils.reranker import rerank as cross_rerank, mmr_diversify
from app.schemas.flashcard import Flashcard, FlashcardResponse
from app.utils.embed import get_embedding

logger = logging.getLogger(__name__)

def retrieve_paragraphs(
    query: str,
    top_k: int = 5,
    min_score: float = 0.0,
    use_reranker: bool = True,
    use_mmr: bool = True,
    mmr_lambda: float = 0.7,
) -> List[dict]:
    """
    Retrieve paragraphs using hybrid search with re-ranking and diversity
    
    Args:
        query: Search query
        top_k: Number of paragraphs to retrieve
        min_score: Minimum hybrid score threshold
        use_reranker: Whether to apply cross-encoder re-ranking
        use_mmr: Whether to apply MMR diversity
        mmr_lambda: MMR diversity- relevance tradeoff (0 = diverse, 1 = relevant)
        
    Returns:
        List of paragraphs with content and metadata
    """
    logger.info(f"Retrieving {top_k} paragraphs for query: {query}")
    
    try:
        paragraphs = hybrid_search(query, top_k=top_k * 3, min_score=min_score, use_reranker=False)
        
        candidates = []
        for para in paragraphs:
            candidates.append({
                "id": para.id,
                "content": para.content,
                "metadata": para.meta,
                "embedding": getattr(para, "_embedding_list", None),
            })
        
        if not candidates:
            return []
        
        if use_reranker:
            candidates = cross_rerank(query, candidates, top_k=len(candidates))
        
        if use_mmr and candidates:
            query_vector = get_embedding(query)
            candidates = mmr_diversify(
                candidates,
                query_vector,
                top_k=top_k,
                lambda_param=mmr_lambda,
            )
        else:
            candidates = candidates[:top_k]
        
        result = []
        for c in candidates:
            result.append({
                "id": c["id"],
                "content": c["content"],
                "metadata": c["metadata"],
            })
        
        logger.info(f"Retrieved {len(result)} paragraphs")
        return result
        
    except Exception as e:
        logger.error(f"Error retrieving paragraphs: {e}")
        raise

def generate_flashcards_from_paragraphs(
    paragraphs: List[dict],
    num_flashcards: int = 5,
    difficulty_level: str = "mixed"
) -> List[Flashcard]:
    """
    Generate flashcards from retrieved paragraphs using LLM
    
    Args:
        paragraphs: List of paragraph dicts with 'content' key
        num_flashcards: Number of flashcards to generate
        difficulty_level: "easy", "medium", "hard", or "mixed"
        
    Returns:
        List of Flashcard objects
    """
    logger.info(f"Generating {num_flashcards} flashcards from {len(paragraphs)} paragraphs")
    
    if not paragraphs:
        logger.warning("No paragraphs provided for flashcard generation")
        return []
    
    # Combine paragraph contents
    combined_text = "\n\n".join([p["content"] for p in paragraphs])
    
    # Limit to avoid token overflow
    combined_text = combined_text[:5000]  # Limit to ~5000 chars
    
    # Create system prompt
    system_prompt = f"""You are an expert educational content creator.
Generate exactly {num_flashcards} flashcards (question-answer pairs) from the provided text.
Difficulty level: {difficulty_level}

Return ONLY valid JSON (no markdown, no explanation):
{{
  "flashcards": [
    {{
      "question": "...",
      "answer": "...",
      "difficulty": "easy|medium|hard",
      "hint": "..."
    }}
  ]
}}

Requirements:
- Questions should test understanding, not just recall
- Answers should be 1-3 sentences
- Hints should guide without giving away the answer
- Mix of difficulty levels if "mixed" is specified
- Valid JSON only"""
    
    user_prompt = f"""Here is the educational content:

{combined_text}

Generate {num_flashcards} flashcards from this content."""
    
    try:
        # Call LLM
        response = ollama_chat(system_prompt, user_prompt, temperature=0.5)
        
        # Parse JSON response
        # Remove markdown code blocks if present
        response = re.sub(r"```(?:json)?", "", response).strip()
        
        data = json.loads(response)
        
        # Convert to Flashcard objects
        flashcards = []
        for item in data.get("flashcards", []):
            flashcard = Flashcard(
                question=item.get("question", ""),
                answer=item.get("answer", ""),
                difficulty=item.get("difficulty", "medium"),
                hint=item.get("hint", None)
            )
            flashcards.append(flashcard)
        
        logger.info(f"Generated {len(flashcards)} flashcards successfully")
        return flashcards
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        logger.error(f"LLM response was: {response[:500]}")
        raise ValueError(f"LLM returned invalid JSON: {e}")
    except Exception as e:
        logger.error(f"Error generating flashcards: {e}")
        raise

def generate_flashcards(
    query: str,
    top_k: int = 5,
    num_flashcards: int = 5,
    difficulty_level: str = "mixed",
    min_score: float = 0.0,
    use_reranker: bool = True,
    use_mmr: bool = True,
) -> FlashcardResponse:
    """
    Complete pipeline: retrieve paragraphs + generate flashcards
    
    Args:
        query: Search query
        top_k: Number of paragraphs to retrieve
        num_flashcards: Number of flashcards to generate
        difficulty_level: Difficulty level for flashcards
        min_score: Minimum hybrid score threshold
        use_reranker: Whether to apply cross-encoder re-ranking
        use_mmr: Whether to apply MMR diversity
        
    Returns:
        FlashcardResponse with flashcards and metadata
    """
    logger.info(f"Generating flashcards for query: {query}")
    
    # Step 1: Retrieve paragraphs
    paragraphs = retrieve_paragraphs(
        query,
        top_k=top_k,
        min_score=min_score,
        use_reranker=use_reranker,
        use_mmr=use_mmr,
    )
    
    # Step 2: Generate flashcards
    flashcards = generate_flashcards_from_paragraphs(
        paragraphs,
        num_flashcards=num_flashcards,
        difficulty_level=difficulty_level
    )
    
    # Step 3: Return response
    return FlashcardResponse(
        query=query,
        flashcards=flashcards,
        source_paragraphs=len(paragraphs)
    )
