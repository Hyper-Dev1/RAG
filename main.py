from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import AutoTokenizer
import requests
import base64
import fitz
import tempfile
import json
from sqlmodel import SQLModel, create_engine
from sentence_transformers import SentenceTransformer

app = FastAPI()

# uvicorn main:app --reload
# NVIDIA_API_URL = "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-ocr-v1"
NVIDIA_API_URL = "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v1"
NVIDIA_API_KEY = "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"
DATABASE_URL = "postgresql://postgres:aXiosNivid321@45.117.153.29:5432/rag"

model = SentenceTransformer("BAAI/bge-small-en")

embedding = model.encode("velocity definition")

print(len(embedding))  # should be 384


@app.get("/")
def root():
    return {"message": "Folder created, ego boosted"}


@app.post("/ocr")
async def ocr_image(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()

        # convert to base64
        image_b64 = base64.b64encode(file_bytes).decode()

        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "input": [
                {
                    "type": "image_url",
                    "url": f"data:{file.content_type};base64,{image_b64}"
                }
            ]
        }

        response = requests.post(
            NVIDIA_API_URL,
            headers=headers,
            json=payload
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=response.text
            )

        return response.json()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract")
async def extract_pdf(file: UploadFile = File(...)):
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    doc = fitz.open(tmp_path)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    MAX_TOKENS = 1000

    chunks = []
    current_chunk = []
    current_tokens = 0
    start_page = 1

    chunk_index = 1

    for page_num, page in enumerate(doc):
        blocks = page.get_text("blocks")

        page_text = []
        for b in blocks:
            text = b[4].strip()
            if text:
                page_text.append(text)

        page_text_str = "\n".join(page_text)
        page_tokens = tokenizer.encode(page_text_str)
        page_token_count = len(page_tokens)

        # If adding this page exceeds limit → flush current chunk
        if current_tokens + page_token_count > MAX_TOKENS and current_chunk:
            file_name = f"page{start_page}-{page_num}.json"

            with open(file_name, "w", encoding="utf-8") as f:
                json.dump({
                    "pages": current_chunk,
                    "tokens": current_tokens
                }, f, ensure_ascii=False, indent=2)

            chunks.append(file_name)

            # reset
            current_chunk = []
            current_tokens = 0
            start_page = page_num + 1

        # Add page to current chunk
        current_chunk.append({
            "page": page_num + 1,
            "content": page_text
        })

        current_tokens += page_token_count

    # Save last chunk
    if current_chunk:
        file_name = f"page{start_page}-{len(doc)}.json"

        with open(file_name, "w", encoding="utf-8") as f:
            json.dump({
                "pages": current_chunk,
                "tokens": current_tokens
            }, f, ensure_ascii=False, indent=2)

        chunks.append(file_name)

    doc.close()

    return {
        "chunks_created": chunks,
        "total_chunks": len(chunks)
    }
