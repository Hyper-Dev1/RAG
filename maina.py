"""
PDF Educational Content Extraction Pipeline
============================================
- Uses Ollama (qwen3.5:397b-cloud) for TOC parsing & lesson boundary detection
- Extracts hierarchy: class → publication → book → unit → lesson → paragraph
- Chunks paragraphs at 300–400 tokens, semantically meaningful
- Stores embeddings via SentenceTransformer
- Context window resets after each lesson (metadata preserved)
- Outputs: SQLite DB + per-lesson JSON context files
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
import fitz  # PyMuPDF
import tempfile
import json
import os
import re
import uuid
import logging
from pathlib import Path
from typing import Optional

import requests
from sentence_transformers import SentenceTransformer
from sqlmodel import SQLModel, Field, Session, create_engine, select, Relationship
from sqlalchemy import Column, Text, event, text
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Config ────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://172.29.128.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:31b-cloud")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "72c9cadffad94cccac859a88f1981936.e4cltr-luMb3PsIlh2yZ4pM-")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en")
DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:aXiosNivid321@45.117.153.29:5432/rag",
)
CHUNK_MIN = int(os.getenv("CHUNK_MIN", "300"))
CHUNK_MAX = int(os.getenv("CHUNK_MAX", "400"))
JSON_OUTPUT_DIR = Path(os.getenv("JSON_OUTPUT_DIR", "./lesson_contexts"))
JSON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Models / DB Schema ────────────────────────────────────────────────────────

class Class(SQLModel, table=True):
    __tablename__ = "class"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)


class Publication(SQLModel, table=True):
    __tablename__ = "publication"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    class_id: int = Field(foreign_key="class.id")


class Book(SQLModel, table=True):
    __tablename__ = "book"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    publication_id: int = Field(foreign_key="publication.id")


class Unit(SQLModel, table=True):
    __tablename__ = "unit"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    book_id: int = Field(foreign_key="book.id")


class Lesson(SQLModel, table=True):
    __tablename__ = "lesson"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    unit_id: int = Field(foreign_key="unit.id")


class Section(SQLModel, table=True):
    __tablename__ = "section"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str  # e.g. "1.1", "1.2"
    lesson_id: int = Field(foreign_key="lesson.id")


class Paragraph(SQLModel, table=True):
    __tablename__ = "paragraph"
    id: Optional[int] = Field(default=None, primary_key=True)
    section_id: int = Field(foreign_key="section.id")
    content: str = Field(sa_column=Column(Text))
    embedding: list = Field(default=None, sa_column=Column(Vector(384)))
    # Rich metadata stored as JSONB for fast key-based queries in PG
    meta: dict = Field(default=None, sa_column=Column("metadata", JSONB))


engine = create_engine(DB_URL, echo=False, pool_pre_ping=True)

# Enable pgvector extension and create tables
with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    conn.commit()

SQLModel.metadata.create_all(engine)

# HNSW index for fast cosine-similarity search (idempotent)
with engine.connect() as conn:
    conn.execute(text(
        "CREATE INDEX IF NOT EXISTS paragraph_embedding_hnsw "
        "ON paragraph USING hnsw (embedding vector_cosine_ops)"
    ))
    conn.commit()

# ─── Embedding model (loaded once) ────────────────────────────────────────────
embedder = SentenceTransformer(EMBED_MODEL)

# ─── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="PDF Educational Content Pipeline")


# ══════════════════════════════════════════════════════════════════════════════
#  Ollama helpers
# ══════════════════════════════════════════════════════════════════════════════

def ollama_chat(system: str, user: str, temperature: float = 0.0) -> str:
    """Call Ollama /api/chat and return the assistant message text."""
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "options": {"temperature": temperature},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    headers = {
        "Authorization": f"Bearer {OLLAMA_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        r = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, headers=headers, timeout=300)
        print(r)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except Exception as e:
        logger.error("Ollama call failed: %s", e)
        raise HTTPException(status_code=502, detail=f"Ollama error: {e}")


def parse_toc_with_llm(raw_toc_text: str) -> list[dict]:
    """
    Ask the LLM to parse a raw TOC string into structured JSON.
    Returns list of:
      { "unit": "...", "lesson": "...", "section": "...", "page": int }
    """
    system = (
        "You are a document structure parser. "
        "Given raw text from a table of contents, extract the hierarchy into JSON. "
        "Return ONLY a JSON array — no markdown, no commentary. "
        "Each element: {\"unit\": str, \"lesson\": str, \"section\": str, \"page\": int}. "
        "If a field is absent use null. Page numbers are integers."
    )
    result = ollama_chat(system, raw_toc_text)
    print(result)
    try:
        # Strip any accidental markdown fences
        clean = re.sub(r"```(?:json)?|```", "", result).strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        logger.warning("TOC parse returned non-JSON, falling back to empty TOC")
        return []


def detect_lesson_boundary(text: str, toc_entries: list[dict]) -> Optional[dict]:
    """
    Ask the LLM whether `text` marks the start of a new lesson/section
    from the TOC. Returns the matching TOC entry or None.
    """
    if not toc_entries:
        return None

    toc_summary = json.dumps([
        {"lesson": e.get("lesson"), "section": e.get("section")} for e in toc_entries
    ], ensure_ascii=False)

    system = (
        "You are a document section detector. "
        "Given a snippet of text and a TOC, decide if this text starts a new "
        "lesson or section from the TOC. "
        "Respond ONLY with JSON: {\"match\": true/false, \"lesson\": \"...\", \"section\": \"...\"}. "
        "If no match, {\"match\": false}."
    )
    user = f"TOC:\n{toc_summary}\n\nText snippet:\n{text[:600]}"
    result = ollama_chat(system, user)
    print(result)
    try:
        clean = re.sub(r"```(?:json)?|```", "", result).strip()
        data = json.loads(clean)
        if data.get("match"):
            # Find the full TOC entry
            for entry in toc_entries:
                if entry.get("lesson") == data.get("lesson"):
                    return entry
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  Text chunking helpers
# ══════════════════════════════════════════════════════════════════════════════

def word_count(text: str) -> int:
    return len(text.split())


def chunk_text(text: str, min_words: int = CHUNK_MIN, max_words: int = CHUNK_MAX) -> list[str]:
    """
    Split `text` into semantically-sized chunks of min_words–max_words words.
    Splits prefer paragraph/sentence boundaries.
    """
    # Split on double-newlines (paragraphs) first
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    chunks: list[str] = []
    current: list[str] = []
    cur_wc = 0

    for para in paragraphs:
        para_wc = word_count(para)
        print(para_wc)

        # Para is larger than max → split by sentences
        if para_wc > max_words:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                s_wc = word_count(sent)
                if cur_wc + s_wc > max_words and cur_wc >= min_words:
                    chunks.append(" ".join(current))
                    current, cur_wc = [], 0
                current.append(sent)
                cur_wc += s_wc
        else:
            if cur_wc + para_wc > max_words and cur_wc >= min_words:
                chunks.append(" ".join(current))
                current, cur_wc = [], 0
            current.append(para)
            cur_wc += para_wc

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if c.strip()]


# ══════════════════════════════════════════════════════════════════════════════
#  DB insert helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_or_create(session: Session, Model, **kwargs):
    """Fetch by all kwargs or create."""
    filters = [getattr(Model, k) == v for k, v in kwargs.items()]
    existing = session.exec(select(Model).where(*filters)).first()
    if existing:
        return existing
    obj = Model(**kwargs)
    session.add(obj)
    session.commit()
    session.refresh(obj)
    return obj


def insert_paragraph(
        session: Session,
        section_id: int,
        content: str,
        metadata: dict,
) -> Paragraph:
    # BGE-small-en: encode with query prefix for asymmetric retrieval
    vector = embedder.encode(
        f"passage: {content}", normalize_embeddings=True
    ).tolist()
    para = Paragraph(
        section_id=section_id,
        content=content,
        embedding=vector,  # pgvector Vector(384) — accepts a plain list
        meta=metadata,  # JSONB column aliased as 'meta' in the model
    )
    session.add(para)
    session.commit()
    session.refresh(para)
    return para


# ══════════════════════════════════════════════════════════════════════════════
#  Main endpoint
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/extract")
async def extract_pdf(
    toc_page: str | None = Form(None),
    file: UploadFile = File(...),
    class_name: str = Form("Class 1"),
    publication_name: str = Form("Default Publication"),
    book_name: str = Form("Default Book"),
):
    """
    Upload a PDF educational book. The pipeline will:
    1. Parse the table of contents (TOC) using Ollama.
    2. Scan pages accumulating text until a new lesson boundary is detected.
    3. Chunk each lesson into 300–400 word paragraphs.
    4. Embed each chunk and store in DB with full metadata.
    5. Write per-lesson context JSON files; context resets after each lesson.
    """
    content = await file.read()
    logger.info("INITIATED")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        doc = fitz.open(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot open PDF: {e}")

    # ── 1. Extract TOC ────────────────────────────────────────────────────────
    toc_entries: list[dict] = []

    if toc_page:
        page_text = doc[int(toc_page)].get_text()
        print(page_text)
        if re.search(r"(table of contents|contents|index)", page_text, re.I):
            toc_entries = parse_toc_with_llm(page_text)
            logger.info("LLM-parsed TOC: %d entries", len(toc_entries))
    else:
        # Try to find a TOC page by text heuristic
        logger.info("No native TOC — searching for TOC page via LLM...")
        for page_num in range(min(10, len(doc))):
            page_text = doc[page_num].get_text()
            if re.search(r"(table of contents|contents|index)", page_text, re.I):
                toc_entries = parse_toc_with_llm(page_text)
                logger.info("LLM-parsed TOC: %d entries", len(toc_entries))
                break

    # Map lesson→unit from TOC
    lesson_to_unit: dict[str, str] = {}
    current_unit_name = "Unit 1"
    for entry in toc_entries:
        if entry.get("unit"):
            current_unit_name = entry["unit"]
        if entry.get("lesson"):
            lesson_to_unit[entry["lesson"]] = current_unit_name

    # ── 2. Setup DB hierarchy (class / publication / book) ────────────────────
    with Session(engine) as session:
        cls_obj  = get_or_create(session, Class,       name=class_name)
        pub_obj  = get_or_create(session, Publication, name=publication_name, class_id=cls_obj.id)
        book_obj = get_or_create(session, Book,        name=book_name,        publication_id=pub_obj.id)

    # ── 3. Page-by-page scan ──────────────────────────────────────────────────
    # State

    lesson_buffer:   list[dict]  = []  # list of {"page": int, "text": str}
    context_window:  list[dict]  = []  # accumulates paragraphs for context
    current_lesson_entry: Optional[dict] = toc_entries[0] if toc_entries else None
    current_unit_name_state = lesson_to_unit.get(
        current_lesson_entry["lesson"] if current_lesson_entry else "", "Unit 1"
    )

    stats = {
        "paragraphs_inserted": 0,
        "lessons_processed":   0,
        "json_files_written":  [],
    }

    def flush_lesson(lesson_entry: dict, buffer: list[dict], ctx: list[dict]) -> list[dict]:
        """Process buffered pages for one lesson → DB + JSON. Returns new context."""
        nonlocal stats

        lesson_text = "\n\n".join(p["text"] for p in buffer)
        if not lesson_text.strip():
            return ctx

        lesson_name   = lesson_entry.get("lesson") or "Unknown Lesson"
        section_label = lesson_entry.get("section") or "1.1"
        unit_name     = lesson_to_unit.get(lesson_name, current_unit_name_state)

        with Session(engine) as session:
            unit_obj    = get_or_create(session, Unit,    name=unit_name,     book_id=book_obj.id)
            lesson_obj  = get_or_create(session, Lesson,  name=lesson_name,   unit_id=unit_obj.id)
            section_obj = get_or_create(session, Section, name=section_label, lesson_id=lesson_obj.id)

            meta_base = {
                "class":       class_name,
                "publication": publication_name,
                "book":        book_name,
                "unit":        unit_name,
                "lesson":      lesson_name,
                "section":     section_label,
            }

            chunks = chunk_text(lesson_text)
            lesson_paragraphs: list[dict] = []

            for chunk in chunks:
                para = insert_paragraph(session, section_obj.id, chunk, meta_base)
                stats["paragraphs_inserted"] += 1
                lesson_paragraphs.append({
                    "id":      str(para.id),
                    "content": chunk,
                    "metadata": meta_base,
                })
                ctx.append({"content": chunk, "metadata": meta_base})

            # Write per-lesson JSON (full context up to and including this lesson)
            json_path = JSON_OUTPUT_DIR / f"lesson_{lesson_obj.id}_{_safe_name(lesson_name)}.json"
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump({
                    "lesson":     lesson_name,
                    "section":    section_label,
                    "unit":       unit_name,
                    "metadata":   meta_base,
                    "paragraphs": lesson_paragraphs,
                }, jf, ensure_ascii=False, indent=2)

            stats["json_files_written"].append(str(json_path))
            stats["lessons_processed"] += 1
            logger.info("Flushed lesson '%s' → %d chunks", lesson_name, len(chunks))

        # Reset context: keep metadata skeleton but drop paragraph content
        # (previous lesson context cleared, only metadata preserved as reference)
        new_ctx = [{"metadata": p["metadata"]} for p in lesson_paragraphs]
        return new_ctx

    def _safe_name(s: str) -> str:
        return re.sub(r"[^\w\-]", "_", s)[:50]
    #
    # ── 4. Iterate pages ──────────────────────────────────────────────────────
    for page_num in range(len(doc)):
        page      = doc[page_num]
        blocks    = page.get_text("blocks")
        page_text = "\n".join(b[4].strip() for b in blocks if b[4].strip())

        if not page_text.strip():
            continue

        # Check boundary via LLM
        boundary = detect_lesson_boundary(page_text, toc_entries)

        if boundary and lesson_buffer:
            # Flush previous lesson
            context_window = flush_lesson(current_lesson_entry, lesson_buffer, context_window)
            lesson_buffer = []
            current_lesson_entry = boundary
            current_unit_name_state = lesson_to_unit.get(boundary.get("lesson", ""), current_unit_name_state)

        elif not current_lesson_entry and toc_entries:
            current_lesson_entry = toc_entries[0]

        lesson_buffer.append({"page": page_num + 1, "text": page_text})
    #
    # Flush final lesson
    if lesson_buffer and current_lesson_entry:
        context_window = flush_lesson(current_lesson_entry, lesson_buffer, context_window)

    doc.close()
    os.unlink(tmp_path)

    return JSONResponse({
        "status": "success",
        "toc": toc_entries,
        "paragraphs_inserted":  stats["paragraphs_inserted"],
        "lessons_processed":    stats["lessons_processed"],
        "json_files_written":   stats["json_files_written"],
        "db_url":               DB_URL,
    })


# ─── Health check ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": OLLAMA_MODEL}
