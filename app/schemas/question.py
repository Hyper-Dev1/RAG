from pydantic import BaseModel
from typing import List, Optional

class QuestionResponse(BaseModel):
    question: str
    answer: str

class GenerateQuestionsRequest(BaseModel):
    paragraph_id: Optional[int] = None
    paragraph_text: Optional[str] = None
    num_questions: int = 3

class GenerateQuestionsResponse(BaseModel):
    questions: List[QuestionResponse]
