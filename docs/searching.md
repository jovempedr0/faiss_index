# Searching

## Hybrid search (dense + BM25)

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
drops its stale BM25 index too, not just adding/removing one). Concurrent hybrid
searches that find it not built yet build it once: the others wait for it.
Both the indexed texts and the query are tokenized with `clean_text` (lowercase, no
punctuation, Portuguese stopwords removed) and with **accents folded**, so a query
typed without accents ("execucao da sentenca") matches "execução da sentença" — on a
real 23-document legal corpus, accentless queries got the same results as accented
ones (`chunks` hybrid recall@5 93% → 98%, BM25-only passage@5 61% → 84%, measured on
the corpus as it was extracted then — see
[Where this fits](use-cases.md#suggested-applications) for today's baseline). Only the
BM25 side is normalized like this; dense search and reranking read the query as typed.
Each result carries `rrf_score` (used for ranking) and `dense_rank`/`bm25_rank`
(whichever list(s) it came from) instead of `evaluate_strategy`'s `distance`/
`similarity`.

## Reranking (retrieve-then-rerank)

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

A cross-encoder reads a fixed number of tokens (512 for the default model, roughly 350
words of Portuguese) and silently ignores the rest, so a candidate longer than that
would be judged by its opening alone — which is most of a court document. Candidates
are split into 150-word windows instead, and each one scores as its best window. Only
the 4 windows BM25 ranks highest against the query are actually sent to the model
(`constants.RERANK_PASSAGE_WORDS` / `RERANK_PASSAGES_PER_CANDIDATE`): scoring every
window of every candidate is accurate and unaffordable, about a minute per query on
whole documents against 1.77 s this way.

There's no default `rerank_provider` (unlike `embedding_provider`/`chat_provider`,
which fall back to the OpenAI-compatible path) — reranking is an opt-in capability
with a real new dependency, not something already built into the library. Any object
implementing `rerank(query: str, candidates: List[str]) -> List[float]` works, so a
non-cross-encoder backend (an LLM call, a hosted rerank API, etc.) can be plugged in
the same way as `embedding_provider`/`chat_provider` — see
[Plugging in a custom provider](providers.md#plugging-in-a-custom-provider).

## Comparing strategies and picking the best one

Pick a strategy (and dense vs. hybrid, rerank, embedding prefixes...) by measuring
it on your own documents: write down some queries a user would actually type, with
the file(s) — and ideally a passage — that answer each, and run
`evaluate_retrieval`:

```python
labeled_queries = [
    {
        "query": "Which clause allows early termination without penalty?",
        "relevant_files": ["./data/contract/2025-03/acme.pdf"],   # as stored in metadata["file"]
        "relevant_text": "may terminate this agreement at any time without penalty",  # optional
    },
    # ... a few dozen of these go a long way
]
report = idx.evaluate_retrieval(
    labeled_queries, document_type="contract", strategies=["full", "sections", "chunks"],
    k=5, use_hybrid=True,
)
# {"chunks": {"recall_at_k": 0.98, "mrr": 0.81, "passage_recall_at_k": 0.86,
#             "avg_result_chars": 3417.0, "n_queries": 57}, ...}
```

`recall_at_k`/`mrr` look at whether the right *file* comes back; `passage_recall_at_k`
at whether the returned text actually contains the answer; `avg_result_chars` at how
much text you'd hand to an LLM to get it. Read them together: `full` finds the right
file easily by returning whole documents (on a real 23-document legal corpus: 89% file
recall at ~146k characters per result, vs. 98% for `chunks` at ~3.4k). Reranking helps most
exactly where retrieval is weakest: on that corpus (64 labeled questions, hybrid, k=5)
`rerank=True` took `full` from 79.7% to 89.1% recall and `sections` from 84.4% to
89.1%, and left `chunks` where it was at 89.1% — a chunk is already about the size of
one passage, so there is little for the reranker to pick between. It costs 1.77 s per
query on `full` against 0.03 s without it, so it is worth it when you're returning long
texts and not worth it on `chunks`. Writing the labeled queries by hand is the most reliable;
having an LLM write one question per sampled chunk (keeping the chunk as
`relevant_text`) is a quick way to get started.

One trap with generated queries: questions written from a corpus's own text inherit
that text's spellings and its mistakes, so they favour the pipeline that produced it.
That makes them fine for comparing *retrieval* choices on one corpus (dense vs. hybrid,
prefixes, `k`) and misleading for comparing two *extraction* pipelines — two OCR
engines, two chunkers — against each other. Compare those by generating a set from each
pipeline's own output and running every set against every index: a pipeline that only
wins on its own questions hasn't won. Question sets also differ in difficulty, so read
down a column (same questions, different index) and never across a row.

`compare_strategies` runs several queries × strategies and returns the raw results.
The older way to score them, `calculate_heuristic_score` (and `generate_search`,
which calls it), is **deprecated**: it combines speed, distance, variance, file
diversity and keyword presence, none of which measures relevance, and it structurally
favors `full` (every `full` result is a different file, so its diversity is always
1.0) — on the corpus above it ranked `full` > `sections` > `chunks`:

```python
comparison = idx.compare_strategies(
    queries=["termination clause", "late payment penalty"],
    document_type="contract",
    strategies_compare=["full", "sections", "chunks"],
    k=15,
)
scores = idx.calculate_heuristic_score(comparison, keywords=["termination", "penalty"])  # DeprecationWarning
```

`use_hybrid=True` evaluates each strategy with `evaluate_strategy_hybrid` instead
(`calculate_heuristic_score` then uses the RRF scores' mean/spread, since there's no
`avg_distance`):

```python
comparison = idx.compare_strategies(
    queries=["termination clause"],
    document_type="contract",
    strategies_compare=["full", "chunks"],
    use_hybrid=True,
)
```

`generate_search` (also deprecated) is the shortcut for this same flow, searching the
query as typed (it also takes `use_hybrid`, passed straight through to
`compare_strategies`):

```python
results, scores = idx.generate_search(
    received_query=["termination clause"],
    keywords=["termination", "penalty"],
    document_type="contract",
    strategies_compare=["full", "chunks"],
)
```

## High-level shortcut (e.g.: a search endpoint)

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

Works with any strategy: each returned text is a chunk for `strategy="chunks"`, a
section for `"sections"`, or a whole document for `"full"`.

Concurrent requests for an index that isn't loaded yet (e.g.: the first requests a
server gets after starting) load it from disk once: the first one loads it and the
others wait for it. Requests on an index that's already loaded don't wait on each
other, nor on another index being loaded.

If there's no index to load — never built, files missing from that directory, a
corrupted index — it raises `IndexLoadError` (the log line before it says which of
those it was). It doesn't return an empty list: that's the answer for a query that
matched nothing, and a handler can't tell the two apart, so a broken deployment would
keep answering with no sources instead of failing. An index that loads but has no good
match still returns a list, empty or not.

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

## Managing memory in long-running processes

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
