"""
Retrieval & Flashcard System
=============================
Add this file alongside your existing pipeline. It adds:

  GET  /                          → Flashcard web UI (HTML page)
  GET  /browse                    → Tree: classes → books → units → lessons
  GET  /lesson/{lesson_id}/cards  → Generate flashcards for a lesson
  POST /search/cards              → Semantic search → generate flashcards
  GET  /search                    → Semantic search (raw chunks, no cards)

Mount this on the same FastAPI app or run as a standalone app — both work.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from sqlmodel import Session, select
from sqlalchemy import text
import json, re, logging, os, requests

# ── Re-use the same engine, models, embedder, and ollama_chat from your pipeline
# In production: move shared code to a shared module (e.g. db.py, models.py)
# For now, paste or import from your main file.
# ─────────────────────────────────────────────────────────────────────────────
# >>> COPY THESE FROM YOUR EXISTING FILE (or import them) <<<
#
# from your_pipeline_file import (
#     engine, embedder, ollama_chat,
#     Class, Publication, Book, Unit, Lesson, Section, Paragraph
# )
#
# For this standalone example we re-declare what we need:


# ── Config (same env vars as pipeline) ────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "gemma4:31b-cloud")
OLLAMA_API_KEY  = os.getenv("OLLAMA_API_KEY",  "")
EMBED_MODEL     = os.getenv("EMBED_MODEL",     "BAAI/bge-small-en")
DB_URL          = os.getenv("DATABASE_URL",    "postgresql://postgres:password@localhost:5432/rag")

# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Flashcard Study App")


# ══════════════════════════════════════════════════════════════════════════════
#  RETRIEVAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_by_lesson(lesson_id: int) -> list[dict]:
    """
    Return all paragraphs that belong to a lesson.

    How it works:
      lesson → sections → paragraphs
      We join Section and Paragraph on section.lesson_id = lesson_id
    """
    with Session(engine) as session:
        # Step 1: get all sections for this lesson
        sections = session.exec(
            select(Section).where(Section.lesson_id == lesson_id)
        ).all()

        if not sections:
            return []

        section_ids = [s.id for s in sections]

        # Step 2: get all paragraphs whose section_id is in that list
        paragraphs = session.exec(
            select(Paragraph).where(Paragraph.section_id.in_(section_ids))
        ).all()

        return [
            {"id": p.id, "content": p.content, "metadata": p.meta}
            for p in paragraphs
        ]


def retrieve_by_semantic_search(query: str, top_k: int = 8) -> list[dict]:
    """
    Embed the query and find the most similar paragraphs using
    cosine similarity on the pgvector index.

    How it works:
      1. Convert query text → 384-dim vector using the same embedder
      2. Ask PostgreSQL to rank paragraphs by cosine distance to that vector
      3. Return top_k closest matches

    The SQL operator <=> means cosine distance (smaller = more similar).
    We use 1 - distance to get a similarity score (higher = more similar).
    """
    # BGE models use "query: " prefix for question/search inputs
    # (vs "passage: " used when storing paragraphs — asymmetric retrieval)
    query_vector = embedder.encode(
        f"query: {query}", normalize_embeddings=True
    ).tolist()

    # Build the vector as a PostgreSQL literal string  e.g. "[0.1, 0.2, ...]"
    vector_literal = "[" + ",".join(str(x) for x in query_vector) + "]"

    sql = text(f"""
        SELECT
            id,
            content,
            metadata,
            1 - (embedding <=> '{vector_literal}'::vector) AS similarity
        FROM paragraph
        ORDER BY embedding <=> '{vector_literal}'::vector
        LIMIT :top_k
    """)

    with engine.connect() as conn:
        rows = conn.execute(sql, {"top_k": top_k}).fetchall()

    return [
        {
            "id":         row.id,
            "content":    row.content,
            "metadata":   row.metadata,
            "similarity": round(float(row.similarity), 4),
        }
        for row in rows
    ]


# ══════════════════════════════════════════════════════════════════════════════
#  FLASHCARD GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_flashcards(chunks: list[dict], context_label: str = "") -> list[dict]:
    """
    Send retrieved text chunks to Ollama and ask it to produce flashcards.

    Returns a list of:
      { "question": str, "answer": str, "hint": str, "difficulty": "easy|medium|hard" }

    Strategy:
      - We join all chunk contents into one block of text
      - We give the AI a strict JSON-only system prompt
      - We ask for varied difficulty levels
      - We strip markdown fences from the response before JSON parsing
    """
    if not chunks:
        return []

    # Join chunk contents — limit total to ~3000 words to stay in context window
    combined_text = "\n\n".join(c["content"] for c in chunks)
    words = combined_text.split()
    if len(words) > 3000:
        combined_text = " ".join(words[:3000]) + "\n[... text truncated for context window ...]"

    system = """You are an expert educational content creator specializing in flashcards.

Given a passage of text, generate a set of high-quality flashcards.

Rules:
- Return ONLY a JSON array. No markdown. No preamble. No commentary.
- Each card: {"question": "...", "answer": "...", "hint": "...", "difficulty": "easy|medium|hard"}
- Questions should test understanding, NOT just memorization
- Mix difficulty levels: ~40% easy, 40% medium, 20% hard
- Answers should be concise (1-3 sentences)
- Hints should be a subtle nudge, not the answer
- Generate between 5 and 15 cards depending on content richness
- Do not repeat the same concept twice"""

    user = f"Generate flashcards from this educational content:\n\n{combined_text}"
    if context_label:
        user = f"Topic: {context_label}\n\n" + user

    raw = ollama_chat(system, user, temperature=0.4)

    # Strip markdown code fences if the model added them
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        cards = json.loads(clean)
        # Validate structure — keep only well-formed cards
        valid = []
        for card in cards:
            if isinstance(card, dict) and "question" in card and "answer" in card:
                valid.append({
                    "question":   card.get("question", ""),
                    "answer":     card.get("answer", ""),
                    "hint":       card.get("hint", ""),
                    "difficulty": card.get("difficulty", "medium"),
                })
        return valid
    except json.JSONDecodeError:
        logger.warning("Flashcard generation returned non-JSON: %s", raw[:200])
        return []


# ══════════════════════════════════════════════════════════════════════════════
#  BROWSE TREE ENDPOINT
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/browse", response_class=JSONResponse)
def browse():
    """
    Return the full content tree so the UI can build a lesson picker.

    Structure returned:
    [
      {
        "class_id": 1, "class_name": "Class 1",
        "books": [
          {
            "book_id": 1, "book_name": "Science Book",
            "units": [
              {
                "unit_id": 1, "unit_name": "Unit 1: Living Things",
                "lessons": [
                  { "lesson_id": 1, "lesson_name": "Lesson 1: Cells" }
                ]
              }
            ]
          }
        ]
      }
    ]
    """
    with Session(engine) as session:
        classes = session.exec(select(Class)).all()
        tree = []

        for cls in classes:
            cls_node = {"class_id": cls.id, "class_name": cls.name, "books": []}

            # Get publications for this class
            pubs = session.exec(
                select(Publication).where(Publication.class_id == cls.id)
            ).all()

            for pub in pubs:
                # Get books under this publication
                books = session.exec(
                    select(Book).where(Book.publication_id == pub.id)
                ).all()

                for book in books:
                    book_node = {
                        "book_id":   book.id,
                        "book_name": book.name,
                        "pub_name":  pub.name,
                        "units":     [],
                    }

                    units = session.exec(
                        select(Unit).where(Unit.book_id == book.id)
                    ).all()

                    for unit in units:
                        unit_node = {
                            "unit_id":   unit.id,
                            "unit_name": unit.name,
                            "lessons":   [],
                        }

                        lessons = session.exec(
                            select(Lesson).where(Lesson.unit_id == unit.id)
                        ).all()

                        for lesson in lessons:
                            unit_node["lessons"].append({
                                "lesson_id":   lesson.id,
                                "lesson_name": lesson.name,
                            })

                        book_node["units"].append(unit_node)

                    cls_node["books"].append(book_node)

            tree.append(cls_node)

    return tree


# ══════════════════════════════════════════════════════════════════════════════
#  LESSON → FLASHCARDS ENDPOINT
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/lesson/{lesson_id}/cards")
def lesson_flashcards(lesson_id: int):
    """
    1. Retrieve all paragraphs for this lesson from DB
    2. Send them to Ollama to generate flashcards
    3. Return cards + lesson metadata

    Usage: GET /lesson/3/cards
    """
    # Get lesson name for context label
    with Session(engine) as session:
        lesson = session.get(Lesson, lesson_id)
        if not lesson:
            raise HTTPException(status_code=404, detail=f"Lesson {lesson_id} not found")
        lesson_name = lesson.name

    chunks = retrieve_by_lesson(lesson_id)
    if not chunks:
        raise HTTPException(status_code=404, detail="No content found for this lesson")

    cards = generate_flashcards(chunks, context_label=lesson_name)

    return {
        "lesson_id":   lesson_id,
        "lesson_name": lesson_name,
        "chunks_used": len(chunks),
        "cards":       cards,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SEMANTIC SEARCH ENDPOINT (raw chunks)
# ══════════════════════════════════════════════════════════════════════════════

class SearchRequest(BaseModel):
    query: str
    top_k: int = 8

@app.post("/search")
def semantic_search(req: SearchRequest):
    """
    Return raw paragraph chunks ranked by semantic similarity to the query.
    Useful for debugging retrieval quality before card generation.

    Usage: POST /search  body: {"query": "how do plants make food", "top_k": 5}
    """
    results = retrieve_by_semantic_search(req.query, top_k=req.top_k)
    return {"query": req.query, "results": results}


# ══════════════════════════════════════════════════════════════════════════════
#  SEMANTIC SEARCH → FLASHCARDS ENDPOINT
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/search/cards")
def search_flashcards(req: SearchRequest):
    """
    1. Embed the query and find the most relevant paragraphs
    2. Generate flashcards from those paragraphs
    3. Return cards + the source chunks used

    Usage: POST /search/cards  body: {"query": "photosynthesis", "top_k": 6}
    """
    chunks = retrieve_by_semantic_search(req.query, top_k=req.top_k)
    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant content found")

    cards = generate_flashcards(chunks, context_label=req.query)

    return {
        "query":       req.query,
        "chunks_used": len(chunks),
        "sources":     [
            {
                "content":    c["content"][:200] + "..." if len(c["content"]) > 200 else c["content"],
                "metadata":   c["metadata"],
                "similarity": c.get("similarity"),
            }
            for c in chunks
        ],
        "cards": cards,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  WEB UI
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
def ui():
    """Serve the flashcard study interface."""
    return HTMLResponse(content=FLASHCARD_HTML)


FLASHCARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>StudyFlash</title>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --ink:     #1a1a2e;
    --paper:   #f5f0e8;
    --cream:   #ede8dc;
    --accent:  #c0392b;
    --gold:    #d4a843;
    --muted:   #8a8070;
    --easy:    #2ecc71;
    --medium:  #f39c12;
    --hard:    #e74c3c;
    --shadow:  0 4px 24px rgba(26,26,46,0.12);
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'DM Sans', sans-serif;
    background: var(--paper);
    color: var(--ink);
    min-height: 100vh;
  }

  /* ── Header ── */
  header {
    background: var(--ink);
    color: var(--paper);
    padding: 1.2rem 2rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    border-bottom: 3px solid var(--gold);
  }
  header h1 {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    letter-spacing: -0.02em;
  }
  header span { color: var(--gold); }

  /* ── Tabs ── */
  .tabs {
    display: flex;
    gap: 0;
    background: var(--cream);
    border-bottom: 2px solid #d4c9b5;
    padding: 0 2rem;
  }
  .tab {
    padding: 0.9rem 1.8rem;
    cursor: pointer;
    font-weight: 500;
    font-size: 0.95rem;
    color: var(--muted);
    border-bottom: 3px solid transparent;
    margin-bottom: -2px;
    transition: all 0.2s;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    font-size: 0.8rem;
  }
  .tab:hover { color: var(--ink); }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }

  /* ── Panels ── */
  .panel { display: none; padding: 2rem; max-width: 900px; margin: 0 auto; }
  .panel.active { display: block; }

  /* ── Browse tree ── */
  .tree { background: white; border-radius: 8px; box-shadow: var(--shadow); overflow: hidden; }
  .tree-class  { background: var(--ink); color: var(--paper); padding: 0.8rem 1.2rem; font-family: 'Playfair Display', serif; font-size: 1.05rem; }
  .tree-book   { padding: 0.7rem 1.2rem 0.7rem 2rem; background: var(--cream); font-weight: 500; border-bottom: 1px solid #d4c9b5; font-size: 0.9rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
  .tree-unit   { padding: 0.6rem 1.2rem 0.6rem 2.5rem; font-weight: 500; border-bottom: 1px solid #ede8dc; font-size: 0.95rem; background: #faf8f4; }
  .tree-lesson {
    padding: 0.55rem 1.2rem 0.55rem 3.5rem;
    border-bottom: 1px solid #f0ece4;
    cursor: pointer;
    font-size: 0.9rem;
    transition: background 0.15s;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .tree-lesson:hover { background: #f5f0e8; }
  .tree-lesson .study-btn {
    background: var(--accent);
    color: white;
    border: none;
    padding: 0.3rem 0.8rem;
    border-radius: 4px;
    font-size: 0.75rem;
    cursor: pointer;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    letter-spacing: 0.03em;
    opacity: 0;
    transition: opacity 0.15s;
  }
  .tree-lesson:hover .study-btn { opacity: 1; }

  /* ── Search box ── */
  .search-box {
    display: flex;
    gap: 0.8rem;
    margin-bottom: 1.5rem;
  }
  .search-box input {
    flex: 1;
    padding: 0.9rem 1.2rem;
    border: 2px solid #d4c9b5;
    border-radius: 6px;
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    background: white;
    color: var(--ink);
    transition: border-color 0.2s;
  }
  .search-box input:focus { outline: none; border-color: var(--accent); }
  .search-box button {
    padding: 0.9rem 1.8rem;
    background: var(--accent);
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 1rem;
    cursor: pointer;
    transition: background 0.2s;
  }
  .search-box button:hover { background: #a93226; }

  /* ── Loading state ── */
  .loading {
    text-align: center;
    padding: 3rem;
    color: var(--muted);
    font-size: 0.95rem;
  }
  .spinner {
    width: 36px; height: 36px;
    border: 3px solid #d4c9b5;
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin: 0 auto 1rem;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── Flashcard deck ── */
  #deck-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.5rem;
  }
  #deck-header h2 {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
  }
  #deck-header .meta { color: var(--muted); font-size: 0.85rem; }
  #back-btn {
    background: none;
    border: 2px solid var(--ink);
    padding: 0.4rem 1rem;
    border-radius: 4px;
    cursor: pointer;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    font-weight: 500;
    transition: all 0.2s;
  }
  #back-btn:hover { background: var(--ink); color: var(--paper); }

  /* ── Card ── */
  .card-area {
    perspective: 1200px;
    margin-bottom: 1.5rem;
  }
  .card {
    width: 100%;
    min-height: 240px;
    position: relative;
    transform-style: preserve-3d;
    transition: transform 0.55s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: pointer;
  }
  .card.flipped { transform: rotateY(180deg); }
  .card-face {
    position: absolute;
    inset: 0;
    backface-visibility: hidden;
    border-radius: 12px;
    padding: 2.5rem;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    box-shadow: var(--shadow);
    min-height: 240px;
  }
  .card-front {
    background: white;
    border: 2px solid #e8e2d6;
  }
  .card-back {
    background: var(--ink);
    color: var(--paper);
    transform: rotateY(180deg);
  }
  .card-label {
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1rem;
  }
  .card-back .card-label { color: var(--gold); }
  .card-question {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    line-height: 1.5;
    margin-bottom: 1rem;
  }
  .card-hint {
    font-size: 0.8rem;
    color: var(--muted);
    font-style: italic;
    margin-top: 0.5rem;
  }
  .card-answer {
    font-size: 1rem;
    line-height: 1.65;
    font-weight: 300;
  }
  .card-tap { font-size: 0.75rem; color: var(--muted); margin-top: 1rem; }

  /* ── Difficulty badge ── */
  .badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
  }
  .badge.easy   { background: #d5f5e3; color: #1a7a42; }
  .badge.medium { background: #fdebd0; color: #a05c00; }
  .badge.hard   { background: #fde8e8; color: #a01010; }

  /* ── Nav controls ── */
  .card-controls {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1rem;
    margin-bottom: 1.5rem;
  }
  .nav-btn {
    background: white;
    border: 2px solid #d4c9b5;
    width: 44px; height: 44px;
    border-radius: 50%;
    font-size: 1.2rem;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    transition: all 0.2s;
  }
  .nav-btn:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); }
  .nav-btn:disabled { opacity: 0.3; cursor: default; }
  .card-counter {
    font-size: 0.9rem;
    color: var(--muted);
    min-width: 80px;
    text-align: center;
  }

  /* ── Progress bar ── */
  .progress-track {
    background: #e8e2d6;
    border-radius: 2px;
    height: 4px;
    margin-bottom: 1.5rem;
    overflow: hidden;
  }
  .progress-fill {
    height: 100%;
    background: var(--accent);
    border-radius: 2px;
    transition: width 0.3s ease;
  }

  /* ── Sources ── */
  .sources-toggle {
    font-size: 0.8rem;
    color: var(--muted);
    cursor: pointer;
    text-decoration: underline;
    text-align: center;
    display: block;
    margin-top: 0.5rem;
  }
  .sources-list {
    display: none;
    margin-top: 1rem;
    background: white;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 0.8rem;
    color: var(--muted);
    line-height: 1.6;
    box-shadow: var(--shadow);
    border-left: 3px solid var(--gold);
  }
  .sources-list.visible { display: block; }

  /* ── Error ── */
  .error-box {
    background: #fde8e8;
    border: 1px solid #e74c3c;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    color: #a01010;
    font-size: 0.9rem;
    margin-top: 1rem;
  }
</style>
</head>
<body>

<header>
  <h1>Study<span>Flash</span></h1>
</header>

<!-- Tabs -->
<div class="tabs">
  <div class="tab active" onclick="switchTab('browse')">ð Browse Lessons</div>
  <div class="tab" onclick="switchTab('search')">ð Search Topic</div>
</div>

<!-- BROWSE PANEL -->
<div class="panel active" id="panel-browse">
  <div id="browse-tree">
    <div class="loading"><div class="spinner"></div>Loading content tree…</div>
  </div>
</div>

<!-- SEARCH PANEL -->
<div class="panel" id="panel-search">
  <div class="search-box">
    <input type="text" id="search-input" placeholder="e.g. photosynthesis, water cycle, fractions…" />
    <button onclick="doSearch()">Generate Cards</button>
  </div>
  <div id="search-results"></div>
</div>

<!-- FLASHCARD DECK (shared, shown over either panel) -->
<div class="panel" id="panel-deck" style="display:none;">
  <div id="deck-header">
    <div>
      <h2 id="deck-title">Flashcards</h2>
      <div class="meta" id="deck-meta"></div>
    </div>
    <button id="back-btn" onclick="closeDeck()">← Back</button>
  </div>

  <div class="progress-track">
    <div class="progress-fill" id="progress-fill" style="width:0%"></div>
  </div>

  <div class="card-area">
    <div class="card" id="flashcard" onclick="flipCard()">
      <div class="card-face card-front" id="card-front">
        <div class="card-label">Question</div>
        <div class="badge" id="card-badge"></div>
        <div class="card-question" id="card-question"></div>
        <div class="card-hint" id="card-hint"></div>
        <div class="card-tap">tap to reveal answer</div>
      </div>
      <div class="card-face card-back" id="card-back">
        <div class="card-label">Answer</div>
        <div class="card-answer" id="card-answer"></div>
      </div>
    </div>
  </div>

  <div class="card-controls">
    <button class="nav-btn" id="prev-btn" onclick="prevCard()">&#8592;</button>
    <div class="card-counter" id="card-counter">1 / 1</div>
    <button class="nav-btn" id="next-btn" onclick="nextCard()">&#8594;</button>
  </div>

  <a class="sources-toggle" onclick="toggleSources()">Show source passages</a>
  <div class="sources-list" id="sources-list"></div>
</div>

<script>
// ── State ─────────────────────────────────────────────────────────────────────
let cards    = [];
let sources  = [];
let cardIdx  = 0;
let prevTab  = 'browse';

// ── Tab switching ─────────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => {
    p.style.display = 'none';
    p.classList.remove('active');
  });
  const tabEls = document.querySelectorAll('.tab');
  tabEls[name === 'browse' ? 0 : 1].classList.add('active');
  const panel = document.getElementById('panel-' + name);
  panel.style.display = 'block';
  panel.classList.add('active');
  prevTab = name;
}

// ── Load browse tree ──────────────────────────────────────────────────────────
async function loadBrowseTree() {
  try {
    const res  = await fetch('/browse');
    const tree = await res.json();
    renderTree(tree);
  } catch(e) {
    document.getElementById('browse-tree').innerHTML =
      '<div class="error-box">Could not load content tree. Is the server running?</div>';
  }
}

function renderTree(tree) {
  if (!tree.length) {
    document.getElementById('browse-tree').innerHTML =
      '<div class="loading">No content found. Upload a PDF first via /extract.</div>';
    return;
  }

  let html = '<div class="tree">';
  tree.forEach(cls => {
    html += `<div class="tree-class">ð ${cls.class_name}</div>`;
    cls.books.forEach(book => {
      html += `<div class="tree-book">ð ${book.book_name} <span style="font-weight:300">(${book.pub_name})</span></div>`;
      book.units.forEach(unit => {
        html += `<div class="tree-unit">ð ${unit.unit_name}</div>`;
        unit.lessons.forEach(lesson => {
          html += `
            <div class="tree-lesson">
              <span>ð ${lesson.lesson_name}</span>
              <button class="study-btn" onclick="loadLessonCards(${lesson.lesson_id}, '${escHtml(lesson.lesson_name)}')">
                Study →
              </button>
            </div>`;
        });
      });
    });
  });
  html += '</div>';
  document.getElementById('browse-tree').innerHTML = html;
}

// ── Load lesson flashcards ────────────────────────────────────────────────────
async function loadLessonCards(lessonId, lessonName) {
  showDeck();
  document.getElementById('deck-title').textContent = lessonName;
  document.getElementById('deck-meta').textContent  = 'Generating flashcards…';
  document.getElementById('card-area')?.classList?.remove('ready');

  // Show spinner inside deck
  document.getElementById('flashcard').style.display = 'none';
  document.querySelector('.card-controls').style.display = 'none';
  document.getElementById('progress-fill').style.width = '0%';

  const loadDiv = document.createElement('div');
  loadDiv.className = 'loading';
  loadDiv.id = 'deck-loading';
  loadDiv.innerHTML = '<div class="spinner"></div>Reading lesson and generating cards…';
  document.getElementById('panel-deck').insertBefore(
    loadDiv, document.querySelector('.card-area')
  );

  try {
    const res  = await fetch(`/lesson/${lessonId}/cards`);
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    sources    = [];
    startDeck(data.cards, data.lesson_name, `${data.chunks_used} passages · ${data.cards.length} cards`, sources);
  } catch(e) {
    showDeckError(e.message);
  }
}

// ── Search flashcards ─────────────────────────────────────────────────────────
async function doSearch() {
  const query = document.getElementById('search-input').value.trim();
  if (!query) return;

  document.getElementById('search-results').innerHTML =
    '<div class="loading"><div class="spinner"></div>Searching and generating cards…</div>';

  try {
    const res  = await fetch('/search/cards', {
      method:  'POST',
      headers: {'Content-Type': 'application/json'},
      body:    JSON.stringify({ query, top_k: 8 }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();

    document.getElementById('search-results').innerHTML = '';
    showDeck();

    const srcTexts = data.sources.map(s =>
      `<b>${s.metadata?.lesson || 'Unknown'}</b> (similarity: ${s.similarity ?? '—'})<br>${s.content}`
    );

    startDeck(
      data.cards,
      `"${query}"`,
      `${data.chunks_used} passages · ${data.cards.length} cards`,
      srcTexts
    );
  } catch(e) {
    document.getElementById('search-results').innerHTML =
      `<div class="error-box">${e.message}</div>`;
  }
}

// Also allow pressing Enter in search box
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('search-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') doSearch();
  });
  loadBrowseTree();
});

// ── Deck management ───────────────────────────────────────────────────────────
function showDeck() {
  document.querySelectorAll('.panel').forEach(p => {
    p.style.display = 'none';
    p.classList.remove('active');
  });
  const deck = document.getElementById('panel-deck');
  deck.style.display = 'block';
  deck.classList.add('active');
}

function closeDeck() {
  document.getElementById('panel-deck').style.display = 'none';
  switchTab(prevTab);
}

function startDeck(newCards, title, meta, srcTexts) {
  // Remove loading spinner if present
  const ld = document.getElementById('deck-loading');
  if (ld) ld.remove();

  cards   = newCards;
  sources = srcTexts;
  cardIdx = 0;

  document.getElementById('deck-title').textContent = title;
  document.getElementById('deck-meta').textContent  = meta;
  document.getElementById('flashcard').style.display = '';
  document.querySelector('.card-controls').style.display = '';

  // Render sources
  if (sources.length) {
    document.getElementById('sources-list').innerHTML = sources.join('<hr style="margin:.5rem 0;border-color:#f0ece4">');
  } else {
    document.querySelector('.sources-toggle').style.display = 'none';
  }

  renderCard();
}

function showDeckError(msg) {
  const ld = document.getElementById('deck-loading');
  if (ld) ld.remove();
  document.getElementById('panel-deck').insertAdjacentHTML(
    'beforeend', `<div class="error-box">${msg}</div>`
  );
}

// ── Card rendering ────────────────────────────────────────────────────────────
function renderCard() {
  if (!cards.length) return;
  const card = cards[cardIdx];

  // Unflip
  document.getElementById('flashcard').classList.remove('flipped');

  // Badge
  const badge = document.getElementById('card-badge');
  badge.textContent  = card.difficulty || 'medium';
  badge.className    = `badge ${card.difficulty || 'medium'}`;

  document.getElementById('card-question').textContent = card.question;
  document.getElementById('card-hint').textContent     = card.hint ? `ð¡ ${card.hint}` : '';
  document.getElementById('card-answer').textContent   = card.answer;
  document.getElementById('card-counter').textContent  = `${cardIdx + 1} / ${cards.length}`;

  // Progress
  const pct = ((cardIdx + 1) / cards.length) * 100;
  document.getElementById('progress-fill').style.width = pct + '%';

  // Nav buttons
  document.getElementById('prev-btn').disabled = cardIdx === 0;
  document.getElementById('next-btn').disabled = cardIdx === cards.length - 1;
}

function flipCard() {
  document.getElementById('flashcard').classList.toggle('flipped');
}

function nextCard() {
  if (cardIdx < cards.length - 1) { cardIdx++; renderCard(); }
}

function prevCard() {
  if (cardIdx > 0) { cardIdx--; renderCard(); }
}

// ── Sources toggle ────────────────────────────────────────────────────────────
function toggleSources() {
  document.getElementById('sources-list').classList.toggle('visible');
}

// ── Keyboard navigation ───────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (document.getElementById('panel-deck').style.display === 'none') return;
  if (e.key === 'ArrowRight') nextCard();
  if (e.key === 'ArrowLeft')  prevCard();
  if (e.key === ' ')          { e.preventDefault(); flipCard(); }
  if (e.key === 'Escape')     closeDeck();
});

// ── Utility ───────────────────────────────────────────────────────────────────
function escHtml(s) {
  return s.replace(/'/g, "\\'").replace(/"/g, '&quot;');
}
</script>
</body>
</html>
"""


# ─── Run standalone ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("retrieval_flashcards:app", host="0.0.0.0", port=8001, reload=True)
