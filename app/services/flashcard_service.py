import json
import re
import logging
from typing import List
from app.services.paragraph_service import hybrid_search
from app.utils.llm import ollama_chat
from app.schemas.flashcard import Flashcard, FlashcardResponse

logger = logging.getLogger(__name__)

def retrieve_paragraphs(query: str, top_k: int = 5) -> List[dict]:
    """
    Retrieve paragraphs using semantic search
    
    Args:
        query: Search query
        top_k: Number of paragraphs to retrieve
        
    Returns:
        List of paragraphs with content and metadata
    """
    logger.info(f"Retrieving {top_k} paragraphs for query: {query}")
    
    try:
        paragraphs = hybrid_search(query, top_k=top_k)
        
        result = []
        for para in paragraphs:
            result.append({
                "id": para.id,
                "content": para.content,
                "metadata": para.meta,
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
    difficulty_level: str = "mixed"
) -> FlashcardResponse:
    """
    Complete pipeline: retrieve paragraphs + generate flashcards
    
    Args:
        query: Search query
        top_k: Number of paragraphs to retrieve
        num_flashcards: Number of flashcards to generate
        difficulty_level: Difficulty level for flashcards
        
    Returns:
        FlashcardResponse with flashcards and metadata
    """
    logger.info(f"Generating flashcards for query: {query}")
    
    # Step 1: Retrieve paragraphs
    paragraphs = retrieve_paragraphs(query, top_k=top_k)
    
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
