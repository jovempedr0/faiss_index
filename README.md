# FaissDocumentIndex

Semantic indexing and search (FAISS + pluggable embedding/chat providers) over
documents of **any type** — contracts, reports, evidence, statements of defense,
invoices, etc. The document type (`document_type`) is always received as a parameter
on the methods; nothing is fixed on the class.

By default, embeddings and section-schema calibration go through an OpenAI-compatible
provider (the real OpenAI API, or any local server that speaks its wire protocol — LM
Studio, oMLX, vLLM, etc.); any other backend can be plugged in instead.

- Three indexing strategies per document type: **full** (the whole document),
  **sections** (structural sections, with an LLM-calibrated schema), and **chunks**
  (sliding text windows).
- Automatic FAISS index type selection by corpus size (`flat` → `IVFFlat` →
  `IVFSQ8`), with CPU parallelism.

## Documentation

**<https://jovempedr0.github.io/faiss_index/>** — the same pages live in
[`docs/`](docs/), so they're versioned with the code and reviewed in the same pull
request as the behaviour they describe.

| | |
|---|---|
| [Installation and setup](docs/installation.md) | dependencies, API key, NLTK data, OCR and its language data |
| [Core concepts](docs/concepts.md) | `document_type`, the three strategies, how sections are found |
| [Quickstart](docs/quickstart.md) | build and search an index in a few lines |
| [Indexing](docs/indexing.md) | building, adding documents, indexing text extracted elsewhere |
| [Searching](docs/searching.md) | hybrid search, reranking, comparing strategies, search endpoints |
| [API reference](docs/api.md) | every public method and its parameters |
| [Configuration and tuning](docs/configuration.md) | index selection, performance, environment variables, log language |
| [Providers and models](docs/providers.md) | plugging in your own backends; the models used by default |
| [File layout and testing](docs/operations.md) | what lands on disk, how to run the suites |

## Install

```bash
pip install -e ".[ocr]"
```

Reading `.txt` needs none of the OCR extra; `.pdf`/`.doc`/`.docx` do. See
[Installation and setup](docs/installation.md) for the rest (NLTK stopwords, the API
key, tesseract's `por` language data).

## In a few lines

```python
from faiss_index import FaissDocumentIndex

idx = FaissDocumentIndex(base_path="./data")

# Builds the indices (calibrates the section schema automatically the first time)
idx.build_indices(document_type="contract", base_data_dir="./data", output_index_dir="./faiss_index")

# Searches, loading the index from disk on demand
chunks = idx.generate_search_by_type(
    received_query="termination clause",
    document_type="contract",
    strategy="chunks",
    require_gpu=False,
)
```

[Quickstart](docs/quickstart.md) walks through the same thing with the output at each
step.
