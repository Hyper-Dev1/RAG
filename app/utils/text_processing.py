import re
from app.core.config import CHUNK_MIN, CHUNK_MAX

def word_count(text: str) -> int:
    return len(text.split())

def chunk_text(text: str, min_words: int = CHUNK_MIN, max_words: int = CHUNK_MAX) -> list[str]:
    """
    Split `text` into semantically-sized chunks of min_words–max_words words.
    Splits prefer paragraph/sentence boundaries.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    chunks: list[str] = []
    current: list[str] = []
    cur_wc = 0

    for para in paragraphs:
        para_wc = word_count(para)

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
