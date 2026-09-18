# FaissDocumentIndex

Semantic + lexical search over your own documents, as a library. Point it at a folder,
get FAISS indices on disk and a search call that returns the passages — or the whole
documents — that answer a query.

- **Any document type.** `document_type` is a parameter on every method, never fixed on
  the class: contracts, court decisions, invoices, reports, all on one instance.
- **Three strategies per corpus:** `full` (one vector per document), `sections` (its
  structural parts) and `chunks` (~500-word windows).
- **Hybrid retrieval.** Dense search fused with BM25, so exact terms — names, case
  numbers — aren't lost to semantics. Optional cross-encoder reranking on top.
- **Your backend.** Embeddings and the schema-calibration call speak the OpenAI
  protocol, so the real API or a local server (LM Studio, oMLX, vLLM) both work; any
  other backend plugs in as a provider object.

```bash
pip install -e ".[ocr]"      # the extra is only needed for .pdf/.doc/.docx
```

```python
from faiss_index import FaissDocumentIndex

idx = FaissDocumentIndex(base_path="./data")
idx.build_indices(document_type="contract", base_data_dir="./data", output_index_dir="./faiss_index")

chunks = idx.generate_search_by_type(
    received_query="termination clause",
    document_type="contract",
    strategy="chunks",
    require_gpu=False,
    use_hybrid=True,
)
# chunks: List[str] — ready to become the context of an LLM prompt.
```

That example skips the setup that matters (the API key, NLTK stopwords, OCR and its
language data): see [Installation and setup](docs/installation.md).

## Documentation

**<https://jovempedr0.github.io/faiss_index/>** — the pages live in [`docs/`](docs/), so
they're versioned with the code and reviewed alongside the behaviour they describe.

New here? [**Where this fits**](docs/use-cases.md) is the one to read first: what this is
good for, which strategy to pick, how big a corpus it handles, and what it deliberately
isn't.

| | |
|---|---|
| [Where this fits](docs/use-cases.md) | suggested applications, strategy choice, scale, limitations |
| [Installation and setup](docs/installation.md) | dependencies, API key, NLTK data, OCR and its language data |
| [Core concepts](docs/concepts.md) | `document_type`, the three strategies, how sections are found |
| [Quickstart](docs/quickstart.md) | build and search an index, step by step |
| [Indexing](docs/indexing.md) | building, adding documents, indexing text extracted elsewhere |
| [Searching](docs/searching.md) | hybrid search, reranking, comparing strategies, search endpoints |
| [API reference](docs/api.md) | every public method and its parameters |
| [Configuration and tuning](docs/configuration.md) | index selection, performance, environment variables, log language |
| [Providers and models](docs/providers.md) | plugging in your own backends; the models used by default |
| [File layout and testing](docs/operations.md) | what lands on disk, how to run the suites |
