# AGENTS.md — RAG Question Generator API

## Quick start

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API serves at `http://localhost:8000`, docs at `/docs`.

## Architecture

FastAPI app. Entrypoint: `app/main.py` → includes router at `app/api/router.py` prefix `/api/v1`.

**Directory layout:**

| Path                | Purpose                                                                                     |
| ------------------- | ------------------------------------------------------------------------------------------- |
| `app/core/`         | Config (`config.py`) and DB init (`database.py`)                                            |
| `app/models/`       | SQLModel ORM tables: `Class → Publication → Book → Unit → Lesson → Section → Paragraph`     |
| `app/api/routes/`   | Route handlers (questions, paragraphs, documents)                                           |
| `app/services/`     | Business logic (question generation, hybrid search, PDF ingest)                             |
| `app/utils/`        | `llm.py` (Ollama chat), `embed.py` (sentence-transformers), `text_processing.py` (chunking) |
| `app/schemas/`      | Pydantic request/response models                                                            |
| `app/repositories/` | Empty — not used                                                                            |

**Endpoints:**

- `GET /health`
- `POST /api/v1/questions/generate` — paragraph_id or paragraph_text + num_questions
- `POST /api/v1/questions/search_and_generate` — query + num_questions + top_k + alpha
- `GET /api/v1/paragraphs/search` — hybrid search (semantic + FTS), returns paragraphs with score
- `POST /api/v1/documents/extract` — upload PDF, extract/chunk/embed + store in DB

## Key facts

- **No tests** exist. Add a `tests/` dir with `pytest` if writing tests.
- **No README** — start from this file.
- **`.env` is required** for runtime (DB creds, API keys, Ollama URL/model). Gitignored. Production defaults are hard-coded in `app/core/config.py`.
- `.gitignore` covers `/File`, `/lesson_contexts`, `.env`.
- `requirements.txt` (no `pyproject.toml`).
- Virtual env at `venv/`.
- **DB**: PostgreSQL + pgvector. HNSW index on 384-dim embeddings. Full-text search via `tsvector` on `paragraph.content`. Hybrid search uses `alpha` to blend semantic (`<=>` cosine) and keyword (`ts_rank`) scores.
- **Embeddings**: `BAAI/bge-small-en` via SentenceTransformers. Prepend `"passage: "` prefix.
- **LLM**: Ollama at `OLLAMA_BASE_URL/api/chat` with Bearer token auth. Expects raw JSON response — no markdown fences.
- **Text chunking**: `CHUNK_MIN=300`, `CHUNK_MAX=400` words (configurable via `.env`). Splits on paragraph/sentence boundaries.
- **PDF ingest** (`document_service.py`): Extracts TOC via LLM, detects lesson boundaries page-by-page, chunks per lesson, inserts paragraphs with embeddings into DB, writes JSON to `JSON_OUTPUT_DIR` (default `../lesson_contexts`).
