import requests
import logging
from app.core.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_API_KEY
from fastapi import HTTPException

logger = logging.getLogger(__name__)

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
    
    # Only add Authorization header for remote Ollama instances
    # Local Ollama (localhost) doesn't require authentication
    headers = {
        "Content-Type": "application/json"
    }
    
    # Check if this is a remote Ollama instance
    is_remote = "localhost" not in OLLAMA_BASE_URL and "127.0.0.1" not in OLLAMA_BASE_URL
    
    if is_remote and OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
    
    logger.debug(f"Calling Ollama at {OLLAMA_BASE_URL} with headers: {list(headers.keys())}")
    
    try:
        r = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, headers=headers, timeout=300)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except Exception as e:
        logger.error("Ollama call failed: %s", e)
        raise HTTPException(status_code=502, detail=f"Ollama error: {e}")
