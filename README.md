# FaissDocumentIndex

Semantic indexing and search (FAISS + pluggable embedding/chat providers) over
documents of **any type** — contracts, reports, evidence, statements of defense,
invoices, etc. The document type (`document_type`) is always received as a parameter
on the methods; nothing is fixed on the class.

By default, embeddings and section-schema calibration go through an OpenAI-compatible
provider (the real OpenAI API, or any local server that speaks its wire protocol — LM
Studio, oMLX, vLLM, etc.); any other backend can be plugged in instead — see
[Plugging in a custom provider](#plugging-in-a-custom-provider).

- Three indexing strategies per document type: **full** (the whole document),
  **sections** (structural sections, with an LLM-calibrated schema), and **chunks**
  (sliding text windows).
- Automatic FAISS index type selection by corpus size (`flat` → `IVFFlat` →
  `IVFPQ`), with CPU parallelism and, on Apple Silicon, GPU-accelerated search (MPS).

## Table of contents

- [Installation](#installation)
- [Required setup](#required-setup)
- [Core concepts](#core-concepts)
- [Quickstart](#quickstart)
- [Ways to use it](#ways-to-use-it)
- [API reference](#api-reference)
- [On-disk file layout](#on-disk-file-layout)
- [Performance configuration](#performance-configuration)
- [Models used](#models-used)
- [Plugging in a custom provider](#plugging-in-a-custom-provider)
- [The `config.py` and `constants.py` modules](#the-configpy-and-constantspy-modules)
- [Log language](#log-language)
- [Testing](#testing)

## Installation

Required dependencies:

```bash
pip install numpy faiss-cpu "openai>=1.40" nltk python-dotenv scikit-learn psutil
```

Alternatively, from this repository (editable install, with the OCR and MPS extras):

```bash
pip install -e ".[ocr,mps]"
```

Optional dependency (accelerates search on `flat` indices via GPU on Apple Silicon —
see [`use_mps`](#performance-configuration)); if absent, everything still works
normally on CPU:

```bash
pip install torch
```

`faiss-cpu` only exists for CPU. FAISS itself has no Metal/MPS backend — that's why
`torch`/MPS acceleration is done outside of FAISS (see the performance section).

## Required setup

1. **NLTK stopwords** (used by `clean_text`/`remove_stop_words`):

   ```python
   import nltk
   nltk.download("stopwords")
   ```

2. **OpenAI key** (only for the default provider — see
   [Plugging in a custom provider](#plugging-in-a-custom-provider) to use a different
   backend instead): via the `OPENAI_API_KEY` environment variable (or `.env`), or
   passed directly to the constructor (`openai_key=...`).

3. **Reading PDF/DOC/DOCX**: depends on `extract_text_from_file_ocr_fallback`
   (`utils_ocr.py`), which in turn needs `pdfplumber`, `pytesseract`, `pdf2image`,
   `Pillow`, and the `libreoffice` binary on the PATH (to convert `.doc`/`.docx`
   before OCR). `.txt` files are read directly and don't need any of this.
   `pytesseract` can be swapped for a vision-capable chat model via
   `FAISS_INDEX_OCR_VLM_MODEL` — see
   [The `config.py` and `constants.py` modules](#the-configpy-and-constantspy-modules).

## Core concepts

### `document_type` — never hardcoded

Every method that deals with indices receives `document_type: str` as a parameter.
The same instance can index and search across multiple document types at once
(`self.indices` is a dict keyed by `document_type` → by strategy).

### Indexing strategies

| Strategy | What it indexes | Main metadata |
|---|---|---|
| `full` | The whole document | `content` (a list with the full text) |
| `sections` | Each structural section of the document | `section_name`, `section_text` |
| `chunks` | ~500-word windows, with 50% overlap | `chunk_index`, `chunk_text` |

### LLM-calibrated section schema

The `sections` strategy **doesn't assume any fixed document structure**. Before it
can be used for a `document_type`, a *schema* must be calibrated (section name →
text patterns that mark its start), which is done automatically by `build_indices`
(using the documents themselves as a sample) or manually via
`register_document_type`. The calibrated schema is cached in memory
(`self.section_schemas[document_type]`) and persisted to disk
(`{document_type}_section_schema.json`), so it only needs to be calibrated once per
type.

If the LLM can't identify any section (a document with no recognizable structure),
the `sections` strategy is simply skipped for that type — `full` and `chunks` keep
working normally.

#### Alternative: structure-aware extraction via Docling

Pattern matching (above) works on already-flattened text, so it can't see real
document structure — heading hierarchy, tables, layout. For that, pass a
`structure_provider` on the constructor instead: `sections` is then built from each
document's *actual* structure, with no schema to calibrate at all.

```python
from faiss_index.providers import DoclingStructureProvider

idx = FaissDocumentIndex(
    base_path="./data",
    structure_provider=DoclingStructureProvider(),  # needs: pip install -e ".[docling]"
)
```

There's no default `structure_provider` (same reasoning as `rerank_provider`) — pass
your own object implementing `extract_sections(file_path: str) -> Dict[str, str]`
otherwise. One trade-off to know about: Docling parses the original file itself (it
needs real layout, not text `read_document` already flattened), so each file gets
processed twice when this is configured — once through the usual
pdfplumber/pytesseract path for `full`/`chunks`, once through Docling for `sections`.

**Broken-font PDFs:** some PDFs (a font-encoding issue with a broken/missing
ToUnicode CMap, seen in some Brazilian court-generated documents) make Docling emit
garbled text instead of real content. `DoclingStructureProvider` detects this per file
(`providers._looks_corrupted`) and falls back to `utils_ocr.py`'s pdfplumber+OCR
pipeline — the same one `full`/`chunks` already uses for these documents — to recover
flat text. That document's `sections` output is then just a single `completo` entry,
with no heading structure (same graceful degradation as when the LLM-calibrated schema
finds no structure at all).

Separately: `DoclingStructureProvider.__init__` disables Docling's own OCR by default
(`do_ocr=False`) and sets `KMP_DUPLICATE_LIB_OK`/`OMP_NUM_THREADS` before importing
Docling — both were needed to avoid a segfault (`faiss` and Docling's `torch`
dependency each bundle their own OpenMP runtime, which crashed when both loaded in one
process during testing on Python 3.14/macOS).

## Quickstart

All progress/status messages go through Python's standard `logging` (module-level
`logger`, one per file) — nothing is printed to stdout on its own. Configure a handler
before using the library if you want to see them:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

```python
from faiss_index import FaissDocumentIndex

idx = FaissDocumentIndex(base_path="./data")

# Builds the indices (calibrates the section schema automatically the first time)
idx.build_indices(
    document_type="contract",
    base_data_dir="./data",
    output_index_dir="./faiss_index",
)

# Search on a specific strategy
result = idx.evaluate_strategy(
    query="termination clause",
    document_type="contract",
    strategy="sections",
    k=5,
)
for r in result["results"]:
    print(r["rank"], r["distance"], r["metadata"]["file"])

# In a new session: load the already-built indices
idx2 = FaissDocumentIndex(base_path="./data")
idx2.load_indices(
    path_indices="./faiss_index",
    document_types=["contract"],
    strategies=["full", "sections", "chunks"],
)
```

## Ways to use it

### Before using it: configuration

Before any of the uses below:

1. Install the dependencies ([Installation](#installation)) and run
   `nltk.download("stopwords")` ([Required setup](#required-setup)).
2. Make sure `OPENAI_API_KEY` is set (`.env` or `openai_key=...` on the constructor).
3. If your project's directory structure isn't the default (`./data`,
   `../faiss_index`, etc.), or you need a non-default `.env`/NLTK data location,
   adjust the environment variables described in
   [The `config.py` and `constants.py` modules](#the-configpy-and-constantspy-modules)
   (`FAISS_INDEX_DOTENV_PATH`, `NLTK_DATA_PATH`, `FAISS_INDEX_BASE_DATA_DIR`,
   `FAISS_INDEX_OUTPUT_INDEX_DIR`, `FAISS_INDEX_PATH_INDICES`) — and, to force the
   log language, `FAISS_INDEX_LANG` ([Log language](#log-language)).

   **Important:** `config.py` reads these environment variables exactly once, when
   the module is imported (`import faiss_index` already triggers `import config`).
   Set them *before* the import — via `export` in the shell, a `.env` loaded
   earlier, or `os.environ[...] = ...` as the first lines of your script — never
   after:

   ```python
   import os
   os.environ["FAISS_INDEX_OUTPUT_INDEX_DIR"] = "/data/production/faiss_index"

   from faiss_index import FaissDocumentIndex  # only now does config.py read the variable above
   ```

### Multiple document types on the same instance

A single instance indexes and searches across several `document_type`s at once —
each type's section schema, indices, and embeddings stay isolated from one another:

```python
idx = FaissDocumentIndex(base_path="./data")

for doc_type in ["contract", "invoice"]:
    idx.build_indices(document_type=doc_type, base_data_dir="./data", output_index_dir="./faiss_index")

contract_result = idx.evaluate_strategy("termination clause", document_type="contract", strategy="sections")
invoice_result = idx.evaluate_strategy("total amount", document_type="invoice", strategy="chunks")
```

### Calibrating the section schema manually

`build_indices` calibrates the section schema automatically the first time (using
the documents themselves as a sample). To control this explicitly — for example,
calibrating with a hand-picked sample before indexing thousands of documents, or
recalibrating an existing type:

```python
schema = idx.register_document_type(
    document_type="contract",
    sample_texts=[sample_text_1, sample_text_2, sample_text_3],
    force_recalibrate=True,  # ignores any schema already cached/on disk
)
# {} if the LLM doesn't recognize any section — in that case build_indices skips
# "sections" and proceeds normally with "full"/"chunks" for that document_type.
```

### Hybrid search (dense + BM25)

Pure dense (embedding) search can miss exact terms — names, case/process numbers,
codes — that don't carry much semantic weight but matter a lot for recall.
`evaluate_strategy_hybrid` fuses the dense ranking with a BM25 (lexical) ranking via
Reciprocal Rank Fusion, so a document that only wins on the exact-term match still
surfaces:

```python
result = idx.evaluate_strategy_hybrid(
    query="processo 0829366-83.2025.8.14.0301",
    document_type="contract",
    strategy="chunks",
    k=5,
)
for r in result["results"]:
    print(r["rank"], r["rrf_score"], r["metadata"]["file"])
```

The BM25 side is built lazily, in memory, from the metadata already loaded for that
`document_type`/`strategy` (no extra files on disk, no LLM/embedding calls) and cached
on the instance — invalidated automatically by `add_new_documents`/`unload_indices`/
`load_indices`/`build_indices` (reloading or rebuilding a strategy already in memory
drops its stale BM25 index too, not just adding/removing one).
Each result carries `rrf_score` (used for ranking) and `dense_rank`/`bm25_rank`
(whichever list(s) it came from) instead of `evaluate_strategy`'s `distance`/
`similarity`.

### Reranking (retrieve-then-rerank)

A cross-encoder scores a (query, candidate) pair jointly, which is generally more
accurate than comparing independently-computed embeddings — but has to run at query
time over the candidate set, so it doesn't replace the initial retrieval, it refines
it. `rerank_results` takes the `"results"` list from either `evaluate_strategy` or
`evaluate_strategy_hybrid` and reorders it:

```python
from faiss_index.providers import CrossEncoderRerankProvider

idx = FaissDocumentIndex(
    base_path="./data",
    rerank_provider=CrossEncoderRerankProvider(),  # needs: pip install -e ".[rerank]"
)

candidates = idx.evaluate_strategy("termination clause", "contract", "chunks", k=30)["results"]
top5 = idx.rerank_results("termination clause", candidates, k=5)
for r in top5:
    print(r["rank"], r["rerank_score"], r["metadata"]["file"])
```

There's no default `rerank_provider` (unlike `embedding_provider`/`chat_provider`,
which fall back to the OpenAI-compatible path) — reranking is an opt-in capability
with a real new dependency, not something already built into the library. Any object
implementing `rerank(query: str, candidates: List[str]) -> List[float]` works, so a
non-cross-encoder backend (an LLM call, a hosted rerank API, etc.) can be plugged in
the same way as `embedding_provider`/`chat_provider` — see
[Plugging in a custom provider](#plugging-in-a-custom-provider).

### Comparing strategies and picking the best one

`evaluate_strategy` searches on a single strategy; `compare_strategies` +
`calculate_heuristic_score` run several queries × strategies and score each one
(speed, distance, variance, file diversity, and optionally `keywords`) to help
decide which strategy to use in production:

```python
comparison = idx.compare_strategies(
    queries=["termination clause", "late payment penalty"],
    document_type="contract",
    strategies_compare=["full", "sections", "chunks"],
    k=15,
)
scores = idx.calculate_heuristic_score(comparison, keywords=["termination", "penalty"])
best_strategy = max(scores, key=lambda s: scores[s]["mean_score"])
```

`use_hybrid=True` evaluates each strategy with `evaluate_strategy_hybrid` instead —
`calculate_heuristic_score` adapts its scoring automatically (there's no `avg_distance`
to work with there, so it uses the RRF scores' mean/spread instead):

```python
comparison = idx.compare_strategies(
    queries=["termination clause"],
    document_type="contract",
    strategies_compare=["full", "chunks"],
    use_hybrid=True,
)
```

`generate_search` is the shortcut for this same flow, cleaning the query first (it
also takes `use_hybrid`, passed straight through to `compare_strategies`):

```python
results, scores = idx.generate_search(
    received_query=["termination clause"],
    keywords=["termination", "penalty"],
    document_type="contract",
    strategies_compare=["full", "chunks"],
)
```

### High-level shortcut (e.g.: a search endpoint)

`generate_search_by_type` is meant for application code (e.g.: an HTTP handler): it
takes the raw query, makes sure the index is loaded (loading it on demand from
`config.DEFAULT_PATH_INDICES`/`FAISS_INDEX_PATH_INDICES` if it isn't yet), and
returns just the texts, ready to build a prompt/context:

```python
chunks = idx.generate_search_by_type(
    received_query="termination clause",
    document_type="contract",
    strategy="chunks",
    require_gpu=False,
)
# chunks: List[str], ready to become the context of an LLM prompt, for example.
```

Only works with `strategy="chunks"` — the return value reads
`metadata["chunk_text"]`, a key that only exists in the `chunks` strategy's
metadata (`full`/`sections` store the text under `content`/`section_text` and will
raise a `KeyError` here).

`k` (default 5), `use_hybrid`, and `rerank` are also accepted — `use_hybrid` switches
to `evaluate_strategy_hybrid`, and `rerank` retrieves a larger candidate pool and
narrows it to `k` via `rerank_results` (needs `rerank_provider` configured on the
constructor):

```python
chunks = idx.generate_search_by_type(
    received_query="termination clause",
    document_type="contract",
    strategy="chunks",
    require_gpu=False,
    k=5,
    use_hybrid=True,
    rerank=True,
)
```

### Adding documents without rebuilding the index

To incorporate new documents into indices already loaded in memory (this doesn't
rebuild the existing ones, it just adds vectors/metadata to each strategy's index
already present for the `document_type`):

```python
idx.load_indices(path_indices="./faiss_index", document_types=["contract"], strategies=["full", "chunks"])

new_docs = [("./data/contract/2026-01/new.pdf", extracted_text)]
idx.add_new_documents(document_type="contract", new_docs=new_docs)
```

This updates the indices **in memory**; to persist to disk, run `build_indices`
again (or save the index/metadata manually, following the
[file layout](#on-disk-file-layout)).

### Managing memory in long-running processes

In a process that serves multiple requests (e.g.: a server), load on demand and
unload what's no longer needed, instead of keeping everything in memory all the
time:

```python
if not idx.is_index_loaded(document_types=["contract"], strategies=["chunks"], require_gpu=False):
    idx.load_indices(path_indices="./faiss_index", document_types=["contract"], strategies=["chunks"])

# ... use the index ...

idx.unload_indices(document_type="contract", strategy="chunks")  # just that strategy
idx.unload_all_indices()  # everything
```

## API reference

### `FaissDocumentIndex(base_path, openai_key=None, embedding_model="text-embedding-3-large", embedding_dim=None, section_extraction_model="gpt-4o-mini", embedding_provider=None, chat_provider=None, rerank_provider=None, structure_provider=None, embedding_batch_size=100, num_threads=None, index_type="auto", auto_index_thresholds=(10_000, 80_000), ivf_nlist=None, ivf_nprobe=8, pq_m=8, pq_nbits=8, use_mps=True)`

Constructor. Every indexing/performance parameter has a sensible default, but none
is fixed — see [Performance configuration](#performance-configuration).
`embedding_provider`/`chat_provider` override the default OpenAI-compatible backend
built from `openai_key`/`embedding_model`/`section_extraction_model`; `rerank_provider`/
`structure_provider` have no default at all (opt-in capabilities, not part of the
OpenAI-compatible path) — see
[Plugging in a custom provider](#plugging-in-a-custom-provider).

### Building indices

- **`build_indices(document_type, base_data_dir='./data', output_index_dir='../faiss_index', doc_limit=None)`**
  Reads all supported documents under `base_data_dir/document_type/`, calibrates
  the section schema if needed, generates embeddings (in batches), and
  builds+saves the `full`, `sections` (if a schema exists), and `chunks` indices.

- **`register_document_type(document_type, sample_texts, force_recalibrate=False)`**
  Manually calibrates (via LLM) the section schema for a type, from sample texts.
  Useful for recalibrating (`force_recalibrate=True`) or calibrating before
  indexing.

- **`add_new_documents(document_type, new_docs)`**
  Adds new documents (`List[Tuple[path, text]]`) to already-loaded indices, for
  every strategy that already exists for that `document_type`.

### Search

- **`evaluate_strategy(query, document_type, strategy, k=10) -> Dict`**
  Searches on a single strategy. Returns search time, results (rank, distance,
  metadata, cosine similarity), and aggregated statistics.

- **`evaluate_strategy_hybrid(query, document_type, strategy, k=10, candidate_pool=None, rrf_k=60) -> Dict`**
  Fuses dense (FAISS) and lexical (BM25) rankings via Reciprocal Rank Fusion — see
  [Hybrid search](#hybrid-search-dense--bm25). Results carry `rrf_score`/
  `dense_rank`/`bm25_rank` instead of `distance`/`similarity`.

- **`rerank_results(query, results, k=None) -> List[Dict]`**
  Reorders a `"results"` list (from `evaluate_strategy` or `evaluate_strategy_hybrid`)
  via `self.rerank_provider` — see [Reranking](#reranking-retrieve-then-rerank). Raises
  `ValueError` if no `rerank_provider` was configured.

- **`compare_strategies(queries, document_type, strategies_compare, k=15, use_hybrid=False) -> Dict`**
  Runs `evaluate_strategy` (or `evaluate_strategy_hybrid`, if `use_hybrid=True`) for
  several queries × strategies, for comparison.

- **`calculate_heuristic_score(comparison_results, keywords=None) -> Dict`**
  Scores each strategy (speed, distance/RRF score, variance, file diversity, and,
  optionally, presence of `keywords`) to help pick which strategy to use — works with
  results from either `evaluate_strategy` or `evaluate_strategy_hybrid`.

- **`generate_search(received_query, keywords, document_type, strategies_compare, use_hybrid=False) -> Tuple[Dict, Dict]`**
  Shortcut: cleans the query, compares strategies, and computes the heuristic
  scores.

- **`generate_search_by_type(received_query, document_type, strategy, require_gpu, k=5, use_hybrid=False, rerank=False) -> List[str]`**
  High-level shortcut: loads the index if it's not already in memory, searches
  with `k` results (optionally via `evaluate_strategy_hybrid` and/or narrowed down
  with `rerank_results`), and returns just the found chunks' texts.

### Lifecycle of the in-memory indices

- **`load_indices(path_indices, document_types, strategies, use_gpu=True) -> dict`**
  Loads the index, metadata, embeddings (`mmap`), and section schema from disk.
  `use_gpu` here is the **CUDA** path (irrelevant on macOS — see
  [`use_mps`](#performance-configuration) for real acceleration on Apple Silicon).
  When the installed FAISS has no CUDA support (the case for `faiss-cpu`, the only
  variant installable on macOS), this is detected before trying to move the index
  — it silently falls back to CPU (without trying `StandardGpuResources()` and
  failing on every index loaded).

- **`is_index_loaded(document_types, strategies, require_gpu) -> bool`**
  Checks whether all requested combinations are already loaded (and GPU-accelerated
  — CUDA or MPS —, if `require_gpu=True`).

- **`unload_indices(document_type, strategy=None)`** / **`unload_all_indices()`**
  Frees indices from memory.

### Document reading/processing (used internally, but exposed)

- **`read_document(file_path) -> str`** — reads `.txt/.pdf/.doc/.docx` (with OCR fallback).
- **`extract_sections(text, document_type) -> Dict[str, str]`** — uses the calibrated schema.
- **`get_embeddings(texts) -> np.ndarray`** — generates embeddings in batches.
- **`create_embeddings_full/sections/chunks(docs, ...) -> Tuple[np.ndarray, List]`**

## On-disk file layout

```
<output_index_dir>/
  <document_type>/
    <document_type>_section_schema.json
    <document_type>_full.index
    <document_type>_full_metadata.json
    <document_type>_full_embeddings.npy
    <document_type>_sections.index          # if a schema was calibrated
    <document_type>_sections_metadata.json
    <document_type>_sections_embeddings.npy
    <document_type>_chunks.index
    <document_type>_chunks_metadata.json
    <document_type>_chunks_embeddings.npy
```

`load_indices(path_indices=..., document_types=[...])` expects exactly this layout
(one subfolder per `document_type` inside `path_indices`).

## Performance configuration

Everything below is configurable via the constructor — nothing is hardcoded in the
middle of the code:

| Parameter | Effect |
|---|---|
| `embedding_batch_size` | How many texts go per call to the embeddings API (fewer round-trips). |
| `num_threads` | Threads FAISS uses for search (`None` = all cores). |
| `index_type` | `"auto"` (by size) or fixed: `"flat"`, `"ivf_flat"`, `"ivf_pq"`. |
| `auto_index_thresholds` | `(flat_limit, ivf_flat_limit)` used when `index_type="auto"`. Defaults to `(10_000, 80_000)`. |
| `ivf_nlist` / `ivf_nprobe` | Number of clusters / clusters visited per search in `ivf_flat`/`ivf_pq`. |
| `pq_m` / `pq_nbits` | Compression parameters for `ivf_pq`. |
| `use_mps` | Accelerates search on `flat` indices via GPU (Apple Silicon), when `torch` with MPS is available. `ivf_*` indices keep using FAISS's native search. |

Automatic index type selection (`index_type="auto"`, the default):

- `n ≤ auto_index_thresholds[0]` → `IndexFlatL2` (exact search)
- `auto_index_thresholds[0] < n ≤ auto_index_thresholds[1]` → `IndexIVFFlat` (approximate)
- `n > auto_index_thresholds[1]` → `IndexIVFPQ` (approximate + compressed)

`IndexIVFPQ` needs at least `2**pq_nbits` vectors to train its product quantizer (256
with the default `pq_nbits=8`) — a separate, higher floor than `ivf_nlist`'s own
minimum. Below it, `_build_faiss_index` falls back to `IndexIVFFlat` for that
corpus (with a warning) instead of letting FAISS raise a training error. Only
reachable with `index_type="ivf_pq"` forced explicitly, or `auto_index_thresholds`
lowered well below the default `80_000` — the default thresholds never route a
corpus that small into `ivf_pq`.

## Models used

With the default provider (see [Plugging in a custom provider](#plugging-in-a-custom-provider)
for anything else), FaissDocumentIndex uses two independent OpenAI models, both
configurable on the constructor:

- **`embedding_model`** — generates the vectors that FAISS indexes and searches.
  Two OpenAI models are known out of the box (`constants.EMBEDDING_DIMENSIONS`),
  and their dimension is derived automatically:

  - `text-embedding-3-large` (3072 dimensions, the default) — better search
    quality, at the cost of a larger index/embeddings on disk (`.npy` files) and
    more memory per vector.
  - `text-embedding-3-small` (1536 dimensions) — half the vector size, so a
    lighter index that's faster to load/search and cheaper to generate, at some
    cost to search quality. Worth considering for large corpora, alongside the
    `index_type`/`auto_index_thresholds` choice in
    [Performance configuration](#performance-configuration).

  Any other `embedding_model` (e.g., a local/non-OpenAI embedding model served
  through an OpenAI-compatible API — see `openai_key`) requires passing
  **`embedding_dim`** explicitly with that model's actual output dimension;
  otherwise `__init__` raises `ValueError`.

- **`section_extraction_model`** — used only by
  `register_document_type`/`_infer_section_schema_via_llm` to calibrate the
  section schema: a single structured-output call per `document_type`, cached
  afterward (see [LLM-calibrated section schema](#llm-calibrated-section-schema)).
  Defaults to `gpt-4o-mini` — since it's a one-off, small, structured task (not
  the actual search or generation path), a lighter chat model is enough; there's
  no need for a larger model here.

## Plugging in a custom provider

`openai_key`/`embedding_model`/`section_extraction_model` are the quick path: under
the hood they build an OpenAI-compatible provider (`providers.py`), which works against
the real OpenAI API or any local server that speaks its wire protocol (LM Studio, oMLX,
vLLM, etc. — point `OPENAI_BASE_URL`, or pass `base_url` directly to the provider
classes below, at whichever backend serves your models).

For a backend that doesn't speak the OpenAI protocol at all, pass your own
`embedding_provider`/`chat_provider` instead — `FaissDocumentIndex` only depends on the
small Protocols in `providers.py`, not on any specific SDK:

```python
from typing import List
from faiss_index.providers import EmbeddingProvider  # structural — no need to subclass it

class MyEmbeddingProvider:
    dimension = 768

    def embed(self, texts: List[str]) -> List[List[float]]:
        return my_own_client.embed_documents(texts)

idx = FaissDocumentIndex(
    base_path="./data",
    embedding_provider=MyEmbeddingProvider(),
    chat_provider=my_chat_provider,  # implements .complete_structured(prompt, json_schema) -> dict
)
```

`chat_provider` needs `complete_structured(prompt: str, json_schema: dict) -> dict` —
called once per `document_type` to calibrate its section schema (see
[LLM-calibrated section schema](#llm-calibrated-section-schema)); how it gets structured
JSON back from the underlying model (native structured output, prompt engineering,
forced tool-calling, etc.) is entirely up to the provider.

The OCR fallback's VLM path (`FAISS_INDEX_OCR_VLM_MODEL`, see
[Required setup](#required-setup)) has the same escape hatch: call
`faiss_index.utils_ocr.set_vlm_provider(provider)` with any object implementing
`describe_image(image_png_bytes: bytes, prompt: str) -> str` to use a non-OpenAI-compatible
vision backend.

`rerank_provider` (see [Reranking](#reranking-retrieve-then-rerank)) works the same
way, except there's no default at all — pass `providers.CrossEncoderRerankProvider()`
(needs `pip install -e ".[rerank]"`) or your own object implementing
`rerank(query: str, candidates: List[str]) -> List[float]`.

`structure_provider` (see
[Structure-aware extraction via Docling](#alternative-structure-aware-extraction-via-docling))
is the same kind of opt-in, no-default capability — pass
`providers.DoclingStructureProvider()` (needs `pip install -e ".[docling]"`) or your
own object implementing `extract_sections(file_path: str) -> Dict[str, str]`.

## The `config.py` and `constants.py` modules

The package (`src/faiss_index/`) is a regular, self-contained Python package —
`core.py`, `utils_ocr.py`, `providers.py`, `config.py`, `constants.py`, and `i18n.py`
all live together and import each other as relative submodules, with no dependency on
an external `src.utils` package or any assumption about the host project's directory
structure. `config.py`/`constants.py` hold what used to be scattered (or hardcoded)
inside the main class:

`core.py` itself only holds the `FaissDocumentIndex` constructor; its methods are
implemented across five internal, single-concern mixins that `core.py` composes into
the class (each still callable as if it were a plain method — mixin methods call each
other freely via `self`, regardless of which module they're defined in): the
LLM-calibrated section schema (`_sections.py`), turning documents into
embeddings/metadata (`_ingestion.py`), FAISS index construction and the dense-search
path incl. MPS (`_index_backend.py`), build/save/load/unload lifecycle
(`_lifecycle.py`), and dense/hybrid search, reranking and the `generate_search*`
shortcuts (`_search.py`). These `_*.py` modules are an implementation detail of
`core.py`, not meant to be imported directly.

- **`constants.py`** — fixed protocol/algorithm values that don't vary by
  environment: supported file extensions, embedding dimension per model, the JSON
  Schema used to calibrate sections via LLM, sampling/training limits, etc.
  They're not meant to be overridden — they just name magic numbers/strings that
  used to be loose in the code.

- **`config.py`** — everything that's environment/installation-dependent: loading
  `.env` and NLTK data, and the default values for `FaissDocumentIndex`'s
  parameters (embedding model, index thresholds, paths, etc.), all with a fallback
  to an environment variable. Nothing here assumes any specific project's
  directory structure:

  - **`FAISS_INDEX_DOTENV_PATH`**: path to a specific `.env`. If not set,
    `load_dotenv()` looks for a `.env` starting from the current directory and
    walking up the tree.
  - **`NLTK_DATA_PATH`**: extra NLTK data directory. If not set, only NLTK's own
    default paths are used (e.g.: `~/nltk_data`, populated by
    `nltk.download("stopwords")` — see [Required setup](#required-setup)).
  - **`FAISS_INDEX_BASE_DATA_DIR`** / **`FAISS_INDEX_OUTPUT_INDEX_DIR`**: defaults
    for `base_data_dir`/`output_index_dir` in `build_indices`. Default `./data` /
    `../faiss_index`.
  - **`FAISS_INDEX_PATH_INDICES`**: directory used by `generate_search_by_type` to
    auto-load an index not yet in memory (previously fixed at
    `../src/utils/mounted_volume/faiss_index` — an assumption about a specific
    project structure that no longer applies here). Default `../faiss_index`.
  - **`FAISS_INDEX_OCR_VLM_MODEL`**: switches image OCR (`utils_ocr.process_image`,
    used as a fallback for PDF pages with no extractable text) from `pytesseract`
    (the default, when unset) to a vision-capable chat model, called through the
    OpenAI-compatible client — useful to point at a local server (LM Studio, oMLX,
    vLLM, etc.) serving an OCR-purpose VLM, e.g. `Unlimited-OCR`. Uses
    `OPENAI_API_KEY`/`OPENAI_BASE_URL` from the environment (the OpenAI SDK's own
    convention), independent of the `openai_key` passed to `FaissDocumentIndex`.

## Log language

All `logger.*`/`print()` messages in `core.py` (and the `_*.py` mixins it composes), `config.py`, and `utils_ocr.py` go
through `i18n.py`, which uses Python's standard `gettext`. The text in the source code
is in English (that's gettext's `msgid`); `src/faiss_index/locale/pt/LC_MESSAGES/`
carries the Portuguese translation.

The language is detected from the machine's locale — environment variables
`LANGUAGE`, `LC_ALL`, `LC_MESSAGES`, `LANG`, in that priority order (that's how
`gettext.translation()` decides when no language is passed explicitly). On a
Portuguese machine (`LANG=pt_BR.UTF-8`, for example), the logs come out in
Portuguese; on any other language — including English, or any language without a
translated catalog — they fall back to the original English text (gettext's
default behavior when there's no translation: it returns the `msgid` as-is).

- **`FAISS_INDEX_LANG`**: forces a language (e.g.: `pt`), independent of the
  machine's locale.

### Adding a new language

1. Generate a `.po` from the template: `msginit --locale=es --input=src/faiss_index/locale/faiss_index.pot --output-file=src/faiss_index/locale/es/LC_MESSAGES/faiss_index.po` (creates the `es/LC_MESSAGES/` directory if needed).
2. Translate the `msgstr` entries in `src/faiss_index/locale/es/LC_MESSAGES/faiss_index.po`.
3. Compile to `.mo`: `msgfmt src/faiss_index/locale/es/LC_MESSAGES/faiss_index.po -o src/faiss_index/locale/es/LC_MESSAGES/faiss_index.mo`.

`msginit`/`msgfmt` are part of the `gettext` package (e.g.: `brew install gettext`
on macOS, `apt install gettext` on Linux) — they're only needed to
*generate/recompile* catalogs, not at runtime: the `gettext` used by `i18n.py` is
Python's standard module, no new dependency.

This makes the module usable from any project directory, without assuming a fixed
structure (`src/utils/...`) around it.

## Testing

```bash
pytest              # unit tests, mocked providers — no network, no cost, no server needed
pytest -m integration -v   # + integration test against a real, live OpenAI-compatible server
```

The integration test (`tests/test_integration_local_model.py`) is excluded by default
(`addopts` in `pyproject.toml`) and skips itself (doesn't fail) if `OPENAI_API_KEY`
isn't set or the server isn't reachable — it builds a tiny throwaway index and runs
`evaluate_strategy`/`evaluate_strategy_hybrid` for real. It defaults to this project's
own local dev setup (oMLX serving `jina-embeddings-v5-text-small-retrieval-mlx` +
`Qwen3-14B-4bit`); point it at different models via `FAISS_INDEX_TEST_EMBEDDING_MODEL`/
`FAISS_INDEX_TEST_EMBEDDING_DIM`/`FAISS_INDEX_TEST_CHAT_MODEL`.

**CI:** `.github/workflows/tests.yml` runs the unit suite (Python 3.11 and 3.12) on
every push/PR to `main`. It installs the `ocr` extra alongside `dev` — `core.py`
imports `utils_ocr` unconditionally, so `pdfplumber`/`pytesseract`/`pdf2image`/`Pillow`
are needed just to import the package, not only for real OCR calls. `rerank`/`docling`
aren't installed: both are imported lazily inside their provider classes
(`CrossEncoderRerankProvider`/`DoclingStructureProvider`), which the unit suite never
instantiates. The integration test stays excluded, same as running `pytest` locally.
