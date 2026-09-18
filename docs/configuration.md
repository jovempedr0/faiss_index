# Configuration and tuning

## Performance configuration

Everything below is configurable via the constructor — nothing is hardcoded in the
middle of the code:

| Parameter | Effect |
|---|---|
| `embedding_batch_size` | How many texts go per call to the embeddings API (fewer round-trips). |
| `num_threads` | Threads FAISS uses for search (`None` = all cores). |
| `index_type` | `"auto"` (by size) or fixed: `"flat"`, `"ivf_flat"`, `"ivf_sq8"`, `"ivf_pq"`. |
| `auto_index_thresholds` | `(flat_limit, ivf_flat_limit)` used when `index_type="auto"`. Defaults to `(10_000, 80_000)`. |
| `ivf_nlist` / `ivf_nprobe` | Number of clusters / clusters visited per search in `ivf_flat`/`ivf_sq8`/`ivf_pq`. `nprobe` defaults to 32 (see [below](#how-many-clusters-to-visit)). |
| `pq_m` / `pq_nbits` | Compression parameters for `ivf_pq` (see its recall cost below). |
| `use_mps` | *Deprecated, no effect.* Used to run `flat` searches on Apple Silicon's GPU via torch; that measured 2.5–7x slower than FAISS's CPU search at every size tested (1.3k–200k vectors), so the path was removed. |

Automatic index type selection (`index_type="auto"`, the default):

- `n ≤ auto_index_thresholds[0]` → `IndexFlatL2` (exact search)
- `auto_index_thresholds[0] < n ≤ auto_index_thresholds[1]` → `IndexIVFFlat` (approximate)
- `n > auto_index_thresholds[1]` → `IndexIVFScalarQuantizer` with 8-bit codes,
  `"ivf_sq8"` (approximate + compressed: 1 byte per dimension instead of 4)

How much each approximate kind keeps of the exact top-10, on 10,824 real 1024-dim
embeddings (the fixture court documents chunked at 60 words) and 57 real questions,
all with `nlist=104` and `nprobe=8`:

| `index_type` | Recall@10 vs. exact | Same top-1 | Code size per vector |
|---|---|---|---|
| `ivf_flat` | 95.3% | 96% | 4,096 bytes |
| `ivf_sq8` | 95.1% | 96% | 1,024 bytes |
| `ivf_pq` (`pq_m=8`) | 48.1% | 14% | 8 bytes |
| `ivf_pq` (`pq_m=64`) | 70.9% | 63% | 64 bytes |

`ivf_pq` is never chosen automatically: it's for when memory matters more than
retrieval quality, and raising `nprobe` doesn't recover its recall (the loss is in
the codes themselves — `pq_m=8` gave the same 48.1% at `nprobe=64`); raising `pq_m`
(it must divide the embedding dimension) helps only partly. Its `reconstruct`ed
vectors are just as coarse, so `evaluate_strategy`'s `similarity` is off by up to
0.27 there (under 0.001 for `ivf_sq8`).

### How many clusters to visit

`nprobe` defaults to 32. Recall@10 against exact search on the same corpus, at 100,000
vectors, as the synthetic corpus is made less tightly clustered (the noise the vectors
are generated with, as a fraction of the real mean nearest-neighbour distance):

| `nprobe` | tight (0.5) | realistic (1.5) | loose (3.0) |
|---|---|---|---|
| 8 | 98.8% | 92.5% | 94.3% |
| 16 | 99.8% | 97.9% | 97.1% |
| 32 | 100% | 99.5% | 98.7% |

The tight column is the optimistic one — it is where `nprobe=8` looks safe, and it is
also the column furthest from a real corpus, whose documents spread over many more
topics than the 23 the fixture corpus has. The two harder columns agree with the 95.3%
that `nprobe=8` measured on 10,824 real embeddings.

What it costs: at 500,000 vectors `nprobe=32` searches in 1.88 ms/query against 0.52 ms
for `nprobe=8` — both far under the 18.95 ms of exact search over the same corpus. The
reproduction script is `fixtures/eval/benchmark_index_defaults.py`.

`IndexIVFPQ` needs at least `2**pq_nbits` vectors to train its product quantizer (256
with the default `pq_nbits=8`) — a separate, higher floor than `ivf_nlist`'s own
minimum. Below it, `_build_faiss_index` falls back to `IndexIVFFlat` for that
corpus (with a warning) instead of letting FAISS raise a training error. `ivf_sq8`
likewise falls back to `IndexIVFFlat` below 1,000 vectors
(`constants.MIN_SQ8_TRAINING_POINTS`): its quantizer learns each dimension's value
range from the vectors it's built with and clips vectors added later to it, and
ranges learned from too few vectors clip too much. Both are only reachable with
`index_type` forced explicitly, or `auto_index_thresholds` lowered far below their
defaults.

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
path (`_index_backend.py`), build/save/load/unload lifecycle
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

  - **`FAISS_INDEX_DOTENV_PATH`**: path to a `.env` for this library to load. If it
    isn't set, no `.env` is loaded at all: importing a library shouldn't put variables
    into the process from a file the application never named — and what a `.env` holds
    reaches every other library in that process too. An application that wants one
    calls `load_dotenv()` itself, before importing this package.
  - **`NLTK_DATA_PATH`**: extra NLTK data directory. If not set, only NLTK's own
    default paths are used (e.g.: `~/nltk_data`, populated by
    `nltk.download("stopwords")` — see [Required setup](installation.md#required-setup)).
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
