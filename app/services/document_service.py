import fitz
import tempfile
import json
import os
import re
import logging
from typing import Optional
from fastapi import UploadFile, HTTPException
from sqlmodel import Session, select
from app.core.database import engine
from app.core.config import JSON_OUTPUT_DIR, DB_URL
from app.models import Class, Publication, Book, Unit, Lesson, Section, Paragraph
from app.utils.llm import ollama_chat
from app.utils.text_processing import chunk_text
from app.utils.embed import get_embedding

logger = logging.getLogger(__name__)

def parse_toc_with_llm(raw_toc_text: str) -> list[dict]:
    system = (
        "You are a document structure parser. "
        "Given raw text from a table of contents, extract the hierarchy into JSON. "
        "Return ONLY a JSON array — no markdown, no commentary. "
        "Each element: {\"unit\": str, \"lesson\": str, \"section\": str, \"page\": int}. "
        "If a field is absent use null. Page numbers are integers."
    )
    result = ollama_chat(system, raw_toc_text)
    try:
        clean = re.sub(r"```(?:json)?|```", "", result).strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        logger.warning("TOC parse returned non-JSON, falling back to empty TOC")
        return []

def detect_lesson_boundary(text: str, toc_entries: list[dict]) -> Optional[dict]:
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
    try:
        clean = re.sub(r"```(?:json)?|```", "", result).strip()
        data = json.loads(clean)
        if data.get("match"):
            for entry in toc_entries:
                if entry.get("lesson") == data.get("lesson"):
                    return entry
    except Exception:
        pass
    return None

def get_or_create(session: Session, Model, **kwargs):
    filters = [getattr(Model, k) == v for k, v in kwargs.items()]
    existing = session.exec(select(Model).where(*filters)).first()
    if existing:
        return existing
    obj = Model(**kwargs)
    session.add(obj)
    session.commit()
    session.refresh(obj)
    return obj

def insert_paragraph(session: Session, section_id: int, content: str, metadata: dict) -> Paragraph:
    vector = get_embedding(content)
    para = Paragraph(
        section_id=section_id,
        content=content,
        embedding=vector,
        meta=metadata,
    )
    session.add(para)
    session.commit()
    session.refresh(para)
    return para

async def process_pdf(
    file: UploadFile,
    class_name: str,
    publication_name: str,
    book_name: str,
    toc_page: str | None = None
) -> dict:
    content = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        doc = fitz.open(tmp_path)
    except Exception as e:
        os.unlink(tmp_path)
        raise HTTPException(status_code=400, detail=f"Cannot open PDF: {e}")

    toc_entries: list[dict] = []
    if toc_page:
        page_text = doc[int(toc_page)].get_text()
        if re.search(r"(table of contents|contents|index)", page_text, re.I):
            toc_entries = parse_toc_with_llm(page_text)
    else:
        for page_num in range(min(10, len(doc))):
            page_text = doc[page_num].get_text()
            if re.search(r"(table of contents|contents|index)", page_text, re.I):
                toc_entries = parse_toc_with_llm(page_text)
                break

    lesson_to_unit: dict[str, str] = {}
    current_unit_name = "Unit 1"
    for entry in toc_entries:
        if entry.get("unit"):
            current_unit_name = entry["unit"]
        if entry.get("lesson"):
            lesson_to_unit[entry["lesson"]] = current_unit_name

    with Session(engine) as session:
        cls_obj = get_or_create(session, Class, name=class_name)
        pub_obj = get_or_create(session, Publication, name=publication_name, class_id=cls_obj.id)
        book_obj = get_or_create(session, Book, name=book_name, publication_id=pub_obj.id)

    lesson_buffer: list[dict] = []
    context_window: list[dict] = []
    current_lesson_entry: Optional[dict] = toc_entries[0] if toc_entries else None
    current_unit_name_state = lesson_to_unit.get(
        current_lesson_entry["lesson"] if current_lesson_entry else "", "Unit 1"
    )

    stats = {
        "paragraphs_inserted": 0,
        "lessons_processed": 0,
        "json_files_written": [],
    }

    def _safe_name(s: str) -> str:
        return re.sub(r"[^\w\-]", "_", s)[:50]

    def flush_lesson(lesson_entry: dict, buffer: list[dict], ctx: list[dict]) -> list[dict]:
        nonlocal stats
        lesson_text = "\n\n".join(p["text"] for p in buffer)
        if not lesson_text.strip():
            return ctx
        lesson_name = lesson_entry.get("lesson") or "Unknown Lesson"
        section_label = lesson_entry.get("section") or "1.1"
        unit_name = lesson_to_unit.get(lesson_name, current_unit_name_state)

        with Session(engine) as session:
            unit_obj = get_or_create(session, Unit, name=unit_name, book_id=book_obj.id)
            lesson_obj = get_or_create(session, Lesson, name=lesson_name, unit_id=unit_obj.id)
            section_obj = get_or_create(session, Section, name=section_label, lesson_id=lesson_obj.id)

            meta_base = {
                "class": class_name,
                "publication": publication_name,
                "book": book_name,
                "unit": unit_name,
                "lesson": lesson_name,
                "section": section_label,
            }

            chunks = chunk_text(lesson_text)
            lesson_paragraphs: list[dict] = []

            for chunk in chunks:
                para = insert_paragraph(session, section_obj.id, chunk, meta_base)
                stats["paragraphs_inserted"] += 1
                lesson_paragraphs.append({
                    "id": str(para.id),
                    "content": chunk,
                    "metadata": meta_base,
                })
                ctx.append({"content": chunk, "metadata": meta_base})

            json_path = JSON_OUTPUT_DIR / f"lesson_{lesson_obj.id}_{_safe_name(lesson_name)}.json"
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump({
                    "lesson": lesson_name,
                    "section": section_label,
                    "unit": unit_name,
                    "metadata": meta_base,
                    "paragraphs": lesson_paragraphs,
                }, jf, ensure_ascii=False, indent=2)

            stats["json_files_written"].append(str(json_path))
            stats["lessons_processed"] += 1

        new_ctx = [{"metadata": p["metadata"]} for p in lesson_paragraphs]
        return new_ctx

    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("blocks")
        page_text = "\n".join(b[4].strip() for b in blocks if b[4].strip())

        if not page_text.strip():
            continue

        boundary = detect_lesson_boundary(page_text, toc_entries)
        if boundary and lesson_buffer:
            context_window = flush_lesson(current_lesson_entry, lesson_buffer, context_window)
            lesson_buffer = []
            current_lesson_entry = boundary
            current_unit_name_state = lesson_to_unit.get(boundary.get("lesson", ""), current_unit_name_state)
        elif not current_lesson_entry and toc_entries:
            current_lesson_entry = toc_entries[0]

        lesson_buffer.append({"page": page_num + 1, "text": page_text})

    if lesson_buffer and current_lesson_entry:
        context_window = flush_lesson(current_lesson_entry, lesson_buffer, context_window)

    doc.close()
    os.unlink(tmp_path)

    return {
        "status": "success",
        "toc": toc_entries,
        "paragraphs_inserted": stats["paragraphs_inserted"],
        "lessons_processed": stats["lessons_processed"],
        "json_files_written": stats["json_files_written"],
        "db_url": DB_URL,
    }
