from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session
from app.core.database import get_session
from app.models.paragraph import Paragraph
from app.schemas.question import GenerateQuestionsRequest, GenerateQuestionsResponse, QuestionResponse
from app.services.question_service import (
    generate_questions_from_paragraph,
    search_and_generate_questions as svc_search_and_generate,
)

router = APIRouter()

@router.post("/generate", response_model=GenerateQuestionsResponse)
def generate_questions(request: GenerateQuestionsRequest, session: Session = Depends(get_session)):
    if request.paragraph_id:
        paragraph = session.get(Paragraph, request.paragraph_id)
        if not paragraph:
            raise HTTPException(status_code=404, detail="Paragraph not found")
        content = paragraph.content
    elif request.paragraph_text:
        content = request.paragraph_text
    else:
        raise HTTPException(status_code=400, detail="Must provide either paragraph_id or paragraph_text")

    questions_data = generate_questions_from_paragraph(content, request.num_questions)
    
    questions = [QuestionResponse(**q) for q in questions_data if isinstance(q, dict) and 'question' in q and 'answer' in q]
    
    return GenerateQuestionsResponse(questions=questions)

@router.post("/search_and_generate", response_model=GenerateQuestionsResponse)
def search_and_generate_questions(
    query: str = Query(..., min_length=1, max_length=500), 
    num_questions: int = Query(3, ge=1, le=10),
    top_k: int = Query(2, ge=1, le=5),
    alpha: float = Query(0.5, ge=0.0, le=1.0),
    min_score: float = Query(0.0, ge=0.0, le=1.0),
    use_reranker: bool = Query(True),
    use_mmr: bool = Query(True),
):
    questions_data = svc_search_and_generate(
        query=query,
        num_questions=num_questions,
        top_k=top_k,
        min_score=min_score,
        use_reranker=use_reranker,
        use_mmr=use_mmr,
    )
    
    questions = [QuestionResponse(**q) for q in questions_data if isinstance(q, dict) and 'question' in q and 'answer' in q]
    
    return GenerateQuestionsResponse(questions=questions)
