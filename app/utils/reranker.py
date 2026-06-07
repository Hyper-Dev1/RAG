import logging
from typing import List

logger = logging.getLogger(__name__)

_reranker = None

def get_reranker(model_name: str = "BAAI/bge-reranker-v2-m3"):
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading reranker model: {model_name}")
            _reranker = CrossEncoder(model_name)
        except Exception as e:
            logger.warning(f"Failed to load reranker '{model_name}': {e}")
            _reranker = False
    return _reranker if _reranker is not False else None


def rerank(query: str, candidates: List[dict], top_k: int = 5) -> List[dict]:
    if not candidates:
        return []

    reranker = get_reranker()
    if reranker is None:
        return candidates[:top_k]

    pairs = [[query, c.get("content", "")] for c in candidates]
    scores = reranker.predict(pairs)

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    candidates.sort(key=lambda c: c.get("rerank_score", 0), reverse=True)
    return candidates[:top_k]


def mmr_diversify(
    candidates: List[dict],
    query_embedding: List[float],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> List[dict]:
    if not candidates or len(candidates) <= top_k:
        return candidates[:top_k]

    from sentence_transformers.util import cos_sim
    import numpy as np

    query_vec = np.array(query_embedding, dtype=np.float32)
    candidate_vecs = []
    for c in candidates:
        emb = c.get("embedding")
        if emb is not None:
            candidate_vecs.append(np.array(emb, dtype=np.float32))
        else:
            candidate_vecs.append(np.zeros(384, dtype=np.float32))

    selected = []
    remaining = list(range(len(candidates)))

    scores = [c.get("rerank_score", c.get("_score", 0)) for c in candidates]

    while len(selected) < top_k and remaining:
        best_idx = None
        best_mmr = -float("inf")

        for i in remaining:
            sim_to_query = scores[i]
            sim_to_selected = 0.0
            if selected:
                sel_vecs = [candidate_vecs[j] for j in selected]
                sims = cos_sim(candidate_vecs[i], np.array(sel_vecs))
                sim_to_selected = float(sims.max())

            mmr = lambda_param * sim_to_query - (1 - lambda_param) * sim_to_selected
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return [candidates[i] for i in selected]
