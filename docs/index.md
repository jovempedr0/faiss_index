# FaissDocumentIndex

Semantic indexing and search (FAISS + pluggable embedding/chat providers) over
documents of **any type** — contracts, reports, evidence, statements of defense,
invoices, etc. The document type (`document_type`) is always received as a parameter
on the methods; nothing is fixed on the class.

By default, embeddings and section-schema calibration go through an OpenAI-compatible
provider (the real OpenAI API, or any local server that speaks its wire protocol — LM
Studio, oMLX, vLLM, etc.); any other backend can be plugged in instead — see
[Plugging in a custom provider](providers.md#plugging-in-a-custom-provider).

- Three indexing strategies per document type: **full** (the whole document),
  **sections** (structural sections, with an LLM-calibrated schema), and **chunks**
  (sliding text windows).
- Automatic FAISS index type selection by corpus size (`flat` → `IVFFlat` →
  `IVFSQ8`), with CPU parallelism.

## Where to start

| If you want to… | Read |
|---|---|
| see what this is good for, and what it isn't | [Where this fits](use-cases.md) |
| install it and get the key/NLTK data in place | [Installation and setup](installation.md) |
| understand what `full`/`sections`/`chunks` are, and how sections are found | [Core concepts](concepts.md) |
| build and search an index in a few lines | [Quickstart](quickstart.md) |
| build indices, add documents, index text extracted elsewhere | [Indexing](indexing.md) |
| run hybrid search, rerank, compare strategies, serve a search endpoint | [Searching](searching.md) |
| look up a method's signature | [API reference](api.md) |
| tune index type, batch sizes, thresholds, or change the log language | [Configuration and tuning](configuration.md) |
| plug in your own embedding/chat/rerank/structure backend | [Providers and models](providers.md) |
| know what lands on disk, or run the tests | [File layout and testing](operations.md) |
