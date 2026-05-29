import os
from pathlib import Path

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:31b-cloud")
# OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "75779511a9a64c82b8a0daaf0c8d7465.BRpV76mAcnS9FH3apx0J4x_z")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en")
DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:aXiosNivid321@45.117.153.29:5432/rag",
)
CHUNK_MIN = int(os.getenv("CHUNK_MIN", "300"))
CHUNK_MAX = int(os.getenv("CHUNK_MAX", "400"))

JSON_OUTPUT_DIR = Path(os.getenv("JSON_OUTPUT_DIR", "../lesson_contexts"))
JSON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
