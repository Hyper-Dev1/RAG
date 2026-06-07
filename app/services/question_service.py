import json
import re
import logging
from typing import List
from app.utils.llm import ollama_chat
from app.services.paragraph_service import hybrid_search
from app.utils.reranker import rerank as cross_rerank, mmr_diversify
from app.utils.embed import get_embedding

logger = logging.getLogger(__name__)


def retrieve_paragraphs_for_questions(
    query: str,
    top_k: int = 3,
    min_score: float = 0.0,
    use_reranker: bool = True,
    use_mmr: bool = True,
) -> List[dict]:
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
        candidates = mmr_diversify(candidates, query_vector, top_k=top_k)
    else:
        candidates = candidates[:top_k]

    return candidates


def generate_questions_from_paragraph(paragraph_content: str, num_questions: int = 3) -> list[dict]:
    """
    Generate questions based on a paragraph.
    """
    system = (
        "You are an expert educational content creator. "
        "Given a text paragraph, generate insightful questions to test a student's understanding. "
        f"Generate exactly {num_questions} questions. "
        "Return ONLY a JSON array of objects. "
        "Each object must have exactly two keys: 'question' and 'answer'. "
        "No markdown fences, no extra text, just the raw JSON."
    )
    user = f"Paragraph:\n{paragraph_content}"
    
    result = ollama_chat(system, user, temperature=0.7)
    
    try:
        # Simple cleanup if the model still adds markdown
        clean = re.sub(r"```(?:json)?|```", "", result).strip()
        data = json.loads(clean)
        return data
    except json.JSONDecodeError:
        # Fallback or empty if parsing fails
        return []


def search_and_generate_questions(
    query: str,
    num_questions: int = 3,
    top_k: int = 3,
    min_score: float = 0.0,
    use_reranker: bool = True,
    use_mmr: bool = True,
) -> list[dict]:
    paragraphs = retrieve_paragraphs_for_questions(
        query,
        top_k=max(top_k, 2),
        min_score=min_score,
        use_reranker=use_reranker,
        use_mmr=use_mmr,
    )

    if not paragraphs:
        return []

    combined = "\n\n".join(p["content"] for p in paragraphs)
    return generate_questions_from_paragraph(combined, num_questions)
