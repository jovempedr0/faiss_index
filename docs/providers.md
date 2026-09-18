# Providers and models

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

`OpenAICompatibleChatProvider` takes a `temperature` (0.0 by default — see
[the section schema](concepts.md#llm-calibrated-section-schema) for why calibration
shouldn't sample). Any `chat_provider` of your own needs
`complete_structured(prompt: str, json_schema: dict) -> dict` —
called once per `document_type` to calibrate its section schema (see
[LLM-calibrated section schema](concepts.md#llm-calibrated-section-schema)); how it gets structured
JSON back from the underlying model (native structured output, prompt engineering,
forced tool-calling, etc.) is entirely up to the provider.

The OCR fallback's VLM path (`FAISS_INDEX_OCR_VLM_MODEL`, see
[Required setup](installation.md#required-setup)) has the same escape hatch: call
`faiss_index.utils_ocr.set_vlm_provider(provider)` with any object implementing
`describe_image(image_png_bytes: bytes, prompt: str) -> str` to use a non-OpenAI-compatible
vision backend.

`rerank_provider` (see [Reranking](searching.md#reranking-retrieve-then-rerank)) works the same
way, except there's no default at all — pass `providers.CrossEncoderRerankProvider()`
(needs `pip install -e ".[rerank]"`) or your own object implementing
`rerank(query: str, candidates: List[str]) -> List[float]`.

`structure_provider` (see
[Sections from the document's own structure](concepts.md#alternative-sections-from-the-documents-own-structure))
is the same kind of opt-in capability, except there's no built-in implementation at
all — pass your own object implementing
`extract_sections(file_path: str) -> Dict[str, str]`.

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
    [Performance configuration](configuration.md#performance-configuration).

  Any other `embedding_model` (e.g., a local/non-OpenAI embedding model served
  through an OpenAI-compatible API — see `openai_key`) requires passing
  **`embedding_dim`** explicitly with that model's actual output dimension;
  otherwise `__init__` raises `ValueError`.

- **`embedding_query_prefix` / `embedding_document_prefix`** — text prepended to
  every search query / every indexed chunk or section before embedding. Many
  retrieval embedding models are trained with asymmetric prefixes and search
  noticeably better with them; OpenAI's `text-embedding-3-*` use none (the default
  `""`). Check your model's card — for example:

  | Model | `embedding_query_prefix` | `embedding_document_prefix` |
  |---|---|---|
  | `jina-embeddings-v5-text-*-retrieval` | `"Query: "` | `"Document: "` |
  | `intfloat/multilingual-e5-*` | `"query: "` | `"passage: "` |

  The document prefix is baked into the saved vectors: it's recorded next to each
  index (`<document_type>_<strategy>_embedding.json`) and `load_indices` logs a
  warning if the instance loading it uses a different one — changing it means
  rebuilding. Queries are embedded as typed (only BM25, in hybrid search, lowercases
  them and strips punctuation/stopwords), so negations like "não"/"sem" reach the
  embedding model and the reranker intact.

- **`section_extraction_model`** — used only by
  `register_document_type`/`_infer_section_schema_via_llm` to calibrate the
  section schema: a single structured-output call per `document_type`, cached
  afterward (see [LLM-calibrated section schema](concepts.md#llm-calibrated-section-schema)).
  Defaults to `gpt-4o-mini` — since it's a one-off, small, structured task (not
  the actual search or generation path), a lighter chat model is enough; there's
  no need for a larger model here.
