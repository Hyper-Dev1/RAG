from fastapi import APIRouter
from app.api.routes import questions
from app.api.routes import paragraphs
from app.api.routes import documents

api_router = APIRouter()
api_router.include_router(questions.router, prefix="/questions", tags=["questions"])
api_router.include_router(paragraphs.router, prefix="/paragraphs", tags=["paragraphs"])
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
