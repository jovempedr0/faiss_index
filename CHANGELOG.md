# Changelog

Notable changes to this library. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) — while the major version is 0,
a minor bump is where breaking changes land.

## [0.2.0] — 2026-09-18

A full review of the library, its findings fixed one at a time, each verified against a
real 23-document corpus rather than only against mocks. Most entries under **Changed** alter
behavior: read that section before upgrading.

### Changed

Defaults and behavior. Each line says how to get the old behavior back where that's possible.

- **`.env` files are no longer loaded automatically.** Importing the package used to search
  upwards from the working directory for a `.env` and load it, which silently reconfigured
  applications that had their own settings. Only `FAISS_INDEX_DOTENV_PATH` loads one now.
  Applications that relied on the old behavior should call `load_dotenv()` themselves.
- **`auto_index_thresholds` defaults to `(100_000, 250_000)`** (was `(10_000, 80_000)`).
  Exact search now covers corpora up to 100k vectors: it costs 3.80 ms/query there and the
  same memory an `IndexIVFFlat` would, so the old limit traded recall for about 3 ms.
  Old behavior: `auto_index_thresholds=(10_000, 80_000)`.
- **`ivf_nprobe` defaults to 32** (was 8). 8 held 92.5-94.3% of the exact top-10 on realistic
  corpus geometry; 32 held ≥98.7%, costing 1.88 ms/query at 500k vectors against 0.52 ms.
  Old behavior: `ivf_nprobe=8`.
- **Large corpora get `ivf_sq8`, not `ivf_pq`.** Product quantization's 8-byte codes kept
  only 48% of the exact top-10 on real 1024-dim embeddings, and more probing didn't recover
  it; 8-bit scalar quantization kept 95% at a quarter of `ivf_flat`'s memory. `ivf_pq` is
  still built when asked for explicitly.
- **`OpenAICompatibleChatProvider` sends `temperature=0.0`** unless told otherwise. It used
  to send none, so the server's sampling applied and section-schema calibration returned a
  different schema on each run — which changed how every document of that type was cut.
- **`rerank_results` scores a long candidate by its best passage, not its opening.** A
  cross-encoder reads a fixed window (512 tokens for the default model) and drops the rest,
  so reranking whole documents *lowered* recall@5 from 79.7% to 31.2%. Candidates are now
  split into 150-word windows, BM25 picks the most promising ones, and the candidate takes
  its best window's score. Tunable via `constants.RERANK_PASSAGE_WORDS` and
  `RERANK_PASSAGES_PER_CANDIDATE`.
- **`"full"` and `"sections"` vectors are pooled, not truncated.** Both used to embed at most
  the first 8,000 characters, so anything past that was invisible to dense search. Both now
  pool the embeddings of the text's word windows.
- **Section metadata no longer repeats the whole document** in every section's `content`.
- **Embeddings are `float32`** in memory and on disk (were `float64` in places), halving
  index files with no measurable retrieval difference.
- **BM25 folds accents** on both corpus and query: accentless queries' BM25-only passage@5
  went from 61% to 84% on a real Portuguese corpus. Dense search and reranking still receive
  the query exactly as typed.
- **Queries are embedded as typed**, with optional `embedding_query_prefix` /
  `embedding_document_prefix` for models whose cards ask for them.
- **Silent failures now raise.** `evaluate_retrieval` with a strategy that isn't loaded, a
  search where no index could be loaded, and pages that need OCR when tesseract can't run it
  used to return empty results or skip content with a warning.
- **The built-in Docling structure provider was removed**; the `structure_provider` hook it
  used stays, so any layout parser can be plugged in.
- **The MPS (Apple GPU) search path was removed.** It measured 2.5-7x slower than FAISS's CPU
  search at every size tested.

### Added

- **`evaluate_retrieval`** — measures recall@k, MRR, passage recall and result size per
  strategy against labeled queries. The supported way to choose a strategy, replacing a
  heuristic score that never measured relevance.
- **`save_indices`** — persists documents added in memory with `add_new_documents`, which
  previously had no way to reach disk.
- **`text_extractor`** — a hook to replace this library's own document reading entirely, for
  when another pipeline (a different OCR engine, a document-understanding API, a cache)
  already produces the text.
- **`embedding_query_prefix` / `embedding_document_prefix`** constructor arguments.
- **A documentation site** at <https://jovempedr0.github.io/faiss_index/>, replacing a README
  that had grown to hold everything.
- **Reproducible evaluation scripts** in `fixtures/eval/` — labeled-query generation,
  retrieval measurement, index-default benchmarking, rerank tuning and an OCR A/B that runs
  each extraction's questions against both indices.

### Fixed

- `.doc`/`.docx` conversion deleted a same-named PDF sitting next to the source.
- The OCR fallback read the wrong page image for documents with non-contiguous page ranges.
- A page whose OCR failed was dropped from the document instead of reported.
- BM25 kept a stale cache after an already-searched document type was rebuilt.
- `generate_search_by_type(require_gpu=True)` reloaded the index on every call, discarding
  documents added since.
- `generate_search_by_type` raised `KeyError` for the `"full"` and `"sections"` strategies.
- `build_indices` silently skipped files with upper-case extensions.
- A `.txt` that isn't UTF-8 was skipped instead of read.
- A failed calibration call deleted the sections index.
- Concurrent first searches each loaded the index, and concurrent hybrid searches each
  rebuilt the BM25 index.
- `load_indices` registered a document type it had loaded nothing for, so later writes
  silently did nothing.
- A rebuild left the previous build's strategy files on disk, so the old corpus kept loading.
- A strategy that produced no vectors was registered as loaded, crashing later evaluation.
- Chunking produced a final window entirely contained in the previous one.
- Section-schema patterns matched arbitrary substrings instead of whole words.
- `utils_ocr` silenced every warning in the host process.
- Importing the package required the optional `[ocr]` extra.

### Deprecated

- `calculate_heuristic_score` and `generate_search` — the score combines speed, distance,
  variance, file diversity and keyword presence, none of which measures relevance, and it
  structurally favors the `full` strategy. Use `evaluate_retrieval`.
- `use_mps` — accepted and ignored, emits a `DeprecationWarning`.

## [0.1.0]

Initial version.
