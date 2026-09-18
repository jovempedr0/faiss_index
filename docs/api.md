# API reference

## `FaissDocumentIndex(base_path, openai_key=None, embedding_model="text-embedding-3-large", embedding_dim=None, section_extraction_model="gpt-4o-mini", embedding_provider=None, chat_provider=None, rerank_provider=None, structure_provider=None, text_extractor=None, embedding_batch_size=100, num_threads=None, index_type="auto", auto_index_thresholds=(10_000, 80_000), ivf_nlist=None, ivf_nprobe=8, pq_m=8, pq_nbits=8, use_mps=None, embedding_query_prefix="", embedding_document_prefix="")`

Constructor. Every indexing/performance parameter has a sensible default, but none
is fixed — see [Performance configuration](configuration.md#performance-configuration).
`embedding_provider`/`chat_provider` override the default OpenAI-compatible backend
built from `openai_key`/`embedding_model`/`section_extraction_model`; `rerank_provider`/
`structure_provider` have no default at all (opt-in capabilities, not part of the
OpenAI-compatible path) — see
[Plugging in a custom provider](providers.md#plugging-in-a-custom-provider).
`text_extractor` replaces `read_document`'s own extraction with a callable of yours —
see [Indexing text extracted elsewhere](indexing.md#indexing-text-extracted-elsewhere).

## Building indices

- **`build_indices(document_type, base_data_dir='./data', output_index_dir='../faiss_index', doc_limit=None)`**
  Reads all supported documents under `base_data_dir/document_type/`, calibrates
  the section schema if needed, generates embeddings (in batches), and
  builds+saves the `full`, `sections` (if a schema exists), and `chunks` indices.
  A strategy that ends up with no vectors (e.g. `sections` when no document has a
  section) isn't registered as loaded, and any files a previous build left for it in
  the output directory are removed — so `load_indices` can't serve an outdated one.

- **`register_document_type(document_type, sample_texts, force_recalibrate=False)`**
  Manually calibrates (via LLM) the section schema for a type, from sample texts.
  Useful for recalibrating (`force_recalibrate=True`) or calibrating before
  indexing. Raises `SectionSchemaError` if the call itself failed, caching nothing so
  a later call tries again.

- **`add_new_documents(document_type, new_docs)`**
  Adds new documents (`List[Tuple[path, text]]`) to already-loaded indices, for
  every strategy that already exists for that `document_type`. Raises `ValueError`
  if no index is loaded for it.

## Search

- **`evaluate_strategy(query, document_type, strategy, k=10) -> Dict`**
  Searches on a single strategy. Returns search time, results (rank, distance,
  metadata, cosine similarity), and aggregated statistics.

- **`evaluate_strategy_hybrid(query, document_type, strategy, k=10, candidate_pool=None, rrf_k=60) -> Dict`**
  Fuses dense (FAISS) and lexical (BM25) rankings via Reciprocal Rank Fusion — see
  [Hybrid search](searching.md#hybrid-search-dense-bm25). Results carry `rrf_score`/
  `dense_rank`/`bm25_rank` instead of `distance`/`similarity`.

- **`rerank_results(query, results, k=None) -> List[Dict]`**
  Reorders a `"results"` list (from `evaluate_strategy` or `evaluate_strategy_hybrid`)
  via `self.rerank_provider` — see [Reranking](searching.md#reranking-retrieve-then-rerank). Raises
  `ValueError` if no `rerank_provider` was configured.

- **`compare_strategies(queries, document_type, strategies_compare, k=15, use_hybrid=False) -> Dict`**
  Runs `evaluate_strategy` (or `evaluate_strategy_hybrid`, if `use_hybrid=True`) for
  several queries × strategies, for comparison.

- **`evaluate_retrieval(labeled_queries, document_type, strategies, k=5, use_hybrid=False, rerank=False) -> Dict`**
  Measures each strategy against queries with known answers (`{"query",
  "relevant_files", "relevant_text"?}`): `recall_at_k`, `mrr`, `passage_recall_at_k`
  and `avg_result_chars`. The way to pick a strategy — see
  [Comparing strategies](searching.md#comparing-strategies-and-picking-the-best-one). Raises
  `ValueError` if one of `strategies` isn't loaded for `document_type`.

- **`calculate_heuristic_score(comparison_results, keywords=None) -> Dict`** — *deprecated*
  Scores each strategy (speed, distance/RRF score, variance, file diversity, and,
  optionally, presence of `keywords`). Doesn't measure relevance and favors `full`;
  emits a `DeprecationWarning`. Use `evaluate_retrieval`.

- **`generate_search(received_query, keywords, document_type, strategies_compare, use_hybrid=False) -> Tuple[Dict, Dict]`** — *deprecated*
  Shortcut: compares strategies for the query and computes the heuristic scores.
  Emits a `DeprecationWarning`, like `calculate_heuristic_score`.

- **`generate_search_by_type(received_query, document_type, strategy, require_gpu, k=5, use_hybrid=False, rerank=False) -> List[str]`**
  High-level shortcut: loads the index if it's not already in memory (once, however
  many concurrent calls need it), searches
  with `k` results (optionally via `evaluate_strategy_hybrid` and/or narrowed down
  with `rerank_results`), and returns just the found texts (chunks, sections or whole
  documents, depending on `strategy`).
  Raises `IndexLoadError` when the index isn't in memory and can't be loaded.

## Lifecycle of the in-memory indices

- **`load_indices(path_indices, document_types, strategies, use_gpu=True) -> dict`**
  Loads the index, metadata, embeddings (`mmap`), and section schema from disk.
  A strategy whose files are missing (or fail to load) is logged and skipped; a
  `document_type` for which nothing loads doesn't show up in `idx.indices` at all.
  `use_gpu` here is the **CUDA** path (faiss-gpu; irrelevant on macOS).
  When the installed FAISS has no CUDA support (the case for `faiss-cpu`, the only
  variant installable on macOS), this is detected before trying to move the index
  — it silently falls back to CPU (without trying `StandardGpuResources()` and
  failing on every index loaded).

- **`is_index_loaded(document_types, strategies, require_gpu) -> bool`**
  Checks whether all requested combinations are already loaded (and moved to the GPU
  via CUDA, if `require_gpu=True`).

- **`save_indices(document_type, output_index_dir=config.DEFAULT_OUTPUT_INDEX_DIR)`**
  Persists the loaded strategies of `document_type`, including documents added with
  `add_new_documents`, in the [on-disk layout](operations.md#on-disk-file-layout). Raises
  `ValueError` if no index is loaded for `document_type`, or if a strategy's index,
  metadata and embedding rows don't line up.

- **`unload_indices(document_type, strategy=None)`** / **`unload_all_indices()`**
  Frees indices from memory.

## Document reading/processing (used internally, but exposed)

- **`read_document(file_path) -> str`** — reads `.txt/.pdf/.doc/.docx` (with OCR fallback),
  or delegates to the constructor's `text_extractor` when one was given. A `.txt` file is
  decoded by its byte-order mark when it has one (UTF-8/16/32, the mark stripped), else as
  UTF-8, else as `constants.TEXT_FALLBACK_ENCODING` (`cp1252` — what Windows and older
  systems export, logged as a warning when it's what worked).
- **`extract_sections(text, document_type) -> Dict[str, str]`** — uses the calibrated schema.
- **`get_embeddings(texts, prefix="") -> np.ndarray`** — generates embeddings in batches
  (`prefix` is prepended to each non-empty text).
- **`create_embeddings_full/sections/chunks(docs, ...) -> Tuple[np.ndarray, List]`**
