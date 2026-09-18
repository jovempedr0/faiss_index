# Where this fits

This library is a retrieval layer over documents you already have: it turns a folder of
files into FAISS indices on disk, and answers queries with the passages — or the
documents — that match. It runs inside your process, against your own embedding backend.
Everything below is a suggestion of where that shape is useful, and what to reach for in
each case; none of it requires code beyond what the [API reference](api.md) covers.

## Suggested applications

### Context for an LLM (RAG)

The most direct use: retrieve passages, put them in a prompt.
[`generate_search_by_type`](api.md#search) returns a `List[str]` of texts, already ready
to concatenate — no result objects to unwrap.

Start with `strategy="chunks"` and `use_hybrid=True`; add `rerank=True` when precision at
the top matters more than latency. On the reference corpus (23 Brazilian court documents,
64 labeled questions, `k=5`):

| Strategy | Search | Recall@5 | MRR | Passage@5 | Chars per result |
|---|---|---|---|---|---|
| `chunks` | hybrid | 89.1% | 0.814 | 79.7% | 3,476 |
| `chunks` | dense | 81.2% | 0.659 | 64.1% | 3,505 |
| `sections` | hybrid | 84.4% | 0.728 | 57.8% | 30,411 |
| `full` | hybrid | 79.7% | 0.649 | 84.4% | 117,894 |

Read those as a starting point, not a promise: 64 questions is a small sample, where a
2-3 point difference is noise, and each one is labeled with a single correct file, so a
question generic enough for several documents to answer counts every other answer as a
miss. [Comparing strategies](searching.md#comparing-strategies-and-picking-the-best-one)
shows how to measure this on your own corpus, which is the only number that should decide
anything.

### A search endpoint

`generate_search_by_type` loads the index from disk the first time it's needed, so a
server doesn't have to warm anything up at boot. Concurrent first requests wait for one
load instead of each doing their own, and
[`unload_indices`](api.md#lifecycle-of-the-in-memory-indices) gives a long-running
process a way to release document types it isn't serving.

Treat `IndexLoadError` as a 500, not as an empty result: it means the index isn't there,
which is a deployment problem, not an answer.

### Triage across documents that repeat a structure

Contracts, court decisions, invoices, incident reports — anything where every document
has roughly the same parts. The `sections` strategy indexes each part separately, so a
query about obligations lands on the obligations section rather than on whichever
paragraph happens to be closest. The schema is calibrated once per `document_type`
([how](concepts.md#llm-calibrated-section-schema)), from the documents themselves.

### "Which document covers this?"

When the unit of the answer is the whole file — routing a question to the right case
file, finding the contract a clause came from — `strategy="full"` indexes one vector per
document, pooled from its chunks, and returns the whole text. It finds the right file
easily precisely because it returns everything, so watch `avg_result_chars` if that text
is going into a prompt.

### A corpus that keeps growing

Documents that arrive over time don't need a rebuild:
[`add_new_documents`](indexing.md#adding-documents-without-rebuilding-the-index) appends
them to the loaded indices and `save_indices` writes them back. Rebuild when the
documents themselves change, or when you change how they're embedded (a different model,
different prefixes, a different chunk size).

### Text that another pipeline already extracted

If your documents arrive as text — from a document-management system, an OCR service, a
database, a scraper — pass a
[`text_extractor`](indexing.md#indexing-text-extracted-elsewhere) and this library never
reads the files itself. That's also the way out when a PDF defeats the built-in
extraction.

### Retrieval that stays on the machine

Both the embedding and the schema-calibration calls speak the OpenAI protocol, so a local
server (LM Studio, oMLX, vLLM) is a configuration change, not a fork — point
`OPENAI_BASE_URL` at it. The indices are files in a directory you choose. For corpora
that can't leave a network, that combination is the reason to use something like this
instead of a hosted search API.

### Deciding between approaches with numbers

Before committing to a strategy, `evaluate_retrieval` scores any of them against your own
labeled queries (recall@k, MRR, passage recall, average result size). It's the same
function the measurements above come from, and it's worth running on a few dozen real
questions from your domain: the right answer differs by corpus.

## Picking a strategy

| | What a result is | Reach for it when | Cost |
|---|---|---|---|
| `chunks` | a ~500-word window | the answer is a passage: RAG context, quoting, question answering | one vector per window — the largest index |
| `sections` | one structural part of a document | documents repeat a structure and the section is the unit of meaning | needs a schema per document type; rebuild when it changes |
| `full` | the entire document | the answer is "which document", or the whole file is the deliverable | returns a lot of text; check `avg_result_chars` before prompting with it |

They aren't exclusive — a common shape is `full` to pick the document and `chunks` to
quote from it, both built in the same `build_indices` call.

## Does it fit your size?

Exact search (`flat`) over 1024-dimensional vectors, measured on an Apple M-series CPU,
`k=5`:

| Vectors | Search |
|---|---|
| 10,000 | 0.55 ms |
| 100,000 | 3.74 ms |
| 200,000 | 7.31 ms |

In a real request that number is rarely what you wait for: embedding the query is a call
to your embedding backend, and it dominates. Above the
[thresholds](configuration.md#performance-configuration) the index type switches to
`IVFFlat` and then to `IVFSQ8` — approximate, but measured at 99.2% of the exact top-10
while storing a quarter of the bytes.

For memory, count roughly 4 KB per vector for `flat`/`IVFFlat` at 1024 dimensions
(1 KB with `IVFSQ8`), plus the metadata, which holds the text itself.

## What this is not

Worth knowing before you build on it:

- **Not a database service.** There's no server, no authentication, no multi-tenancy. It's
  a library that loads indices into your process's memory.
- **No deletion or in-place update.** You can add documents; removing or changing one
  means rebuilding that document type's indices.
- **Single machine.** No sharding, no replication. The scale it addresses is the one in
  the table above, not billions of vectors.
- **One writer.** Several processes can read the same index directory, but building or
  saving from more than one at a time isn't coordinated.
- **Portuguese by default.** The stopword list and the log language default to
  Portuguese, and the text normalisation was tuned on Portuguese documents. Both are
  configurable ([stopwords](configuration.md#the-configpy-and-constantspy-modules),
  [logs](configuration.md#log-language)), but that's the ground it was tested on.
