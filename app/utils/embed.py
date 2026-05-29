from sentence_transformers import SentenceTransformer
from app.core.config import EMBED_MODEL

# Load once globally
embedder = SentenceTransformer(EMBED_MODEL)

def get_embedding(text: str) -> list[float]:
    return embedder.encode(
        f"passage: {text}", normalize_embeddings=True
    ).tolist()
