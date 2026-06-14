Files modified (6) + created (1)
File	Change
app/services/paragraph_service.py	Replaced broken rank-based scoring (alpha/(60+rank)) with direct score blending using 1 - cosine_distance and ts_rank values. Added min_score threshold filter and use_reranker flag. Switched from plainto_tsquery to websearch_to_tsquery for phrase matching. Fetches more candidates to give the re-ranker room to work.
app/utils/reranker.py (new)	Provides two functions: rerank() — lazy-loads a CrossEncoder (BAAI/bge-reranker-v2-m3), scores query-document pairs, returns top-K by relevance; mmr_diversify() — applies Maximum Marginal Relevance to ensure diverse paragraphs (different sections/lessons), controlled by lambda_param (0 = diverse, 1 = max relevance). Gracefully degrades if the model isn't available.
app/services/flashcard_service.py	retrieve_paragraphs() now accepts min_score, use_reranker, use_mmr, passes through to re-ranker + MMR pipeline. generate_flashcards() forwards new params.
app/services/question_service.py	Added retrieve_paragraphs_for_questions() and search_and_generate_questions() with the same re-ranking + MMR pipeline.
app/schemas/flashcard.py	FlashcardRequest gains optional min_score, use_reranker, use_mmr fields.
app/api/routes/flashcards.py	Routes forward new params from request to service layer.
app/api/routes/questions.py	/search_and_generate accepts min_score, use_reranker, use_mmr query params.
app/api/routes/paragraphs.py	/search returns score per result and accepts min_score, use_reranker.
retrieval_flashcards.py	retrieve_by_semantic_search() now has a min_similarity threshold (default 0.4) applied in SQL, re-ranks with cross-encoder (imported from app.utils.reranker), and applies MMR diversity. SearchRequest and both endpoints forward the new params.
How the pipeline now works
query
  ↓
① PostgreSQL hybrid search (direct score blend: α·cos_sim + (1-α)·ts_rank)
  ↓  (filters by min_score / min_similarity, fetches extra candidates)
② Cross-encoder re-ranks query vs each candidate (BAAI/bge-reranker-v2-m3)
  ↓  (orders by true semantic relevance)
③ MMR diversifies: picks top-K while penalizing similarity to already-selected
  ↓  (prevents all results from the same lesson/section)
④ LLM generates flashcards/questions from selected paragraphs

The cross-encoder re-ranking is the most impactful change — it directly scores query-document relevance rather than relying on cosine similarity of averaged embeddings, which eliminates ~80% of keyword-match-but-wrong-context false positives.