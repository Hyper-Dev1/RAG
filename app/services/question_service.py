from app.utils.llm import ollama_chat
import json

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
        import re
        clean = re.sub(r"```(?:json)?|```", "", result).strip()
        data = json.loads(clean)
        return data
    except json.JSONDecodeError:
        # Fallback or empty if parsing fails
        return []
