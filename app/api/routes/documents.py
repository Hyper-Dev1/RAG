from fastapi import APIRouter, Form, UploadFile, File
from fastapi.responses import JSONResponse
from app.services.document_service import process_pdf

router = APIRouter()

@router.post("/extract")
async def extract_pdf(
    toc_page: str | None = Form(None),
    file: UploadFile = File(...),
    class_name: str = Form("Class 1"),
    publication_name: str = Form("Default Publication"),
    book_name: str = Form("Default Book"),
):
    result = await process_pdf(file, class_name, publication_name, book_name, toc_page)
    return JSONResponse(result)
