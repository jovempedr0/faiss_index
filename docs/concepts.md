# Core concepts

## `document_type` — never hardcoded

Every method that deals with indices receives `document_type: str` as a parameter.
The same instance can index and search across multiple document types at once
(`self.indices` is a dict keyed by `document_type` → by strategy).

## Indexing strategies

| Strategy | What it indexes | Main metadata |
|---|---|---|
| `full` | The whole document | `content` (a list with the full text) |
| `sections` | Each structural section of the document | `section_name`, `section_text` |
| `chunks` | ~500-word windows, with 50% overlap | `chunk_index`, `chunk_text` |

Every metadata entry also carries `file`. To expand a section or chunk hit into its
whole document, look the `file` up in the `full` strategy's metadata — the full text is
stored once per file there (section entries used to repeat it in a `content` key, which
multiplied the size of the sections metadata; indices saved with it still load fine).

A `full` vector is the mean of the document's chunk embeddings, L2-normalized before
and after averaging — so it represents the entire text (a single embedding call would
only see the first `MAX_EMBEDDING_INPUT_CHARS` characters), and L2 search over `full`
still ranks by cosine similarity, like the other strategies. `build_indices` pools
the embeddings it already computed for `chunks`, so `full` costs no extra embedding
calls.

A `sections` vector is pooled the same way, from the section's own ~500-word windows
(embedded with the document prefix): sections easily run past
`MAX_EMBEDDING_INPUT_CHARS` (35 of 113 on a real 23-document legal corpus, up to ~420k
characters), and a single embedding call only represented their beginning. On that
corpus, as extracted at the time, this took `sections` from 65% to 81% file recall@5
with dense search (hybrid:
86% → 89%, passage recall 70% → 74%). It costs one embedding call per window instead
of one per section; `sections` indices built before this still load and search as
before — rebuild them to get the pooled vectors.

## LLM-calibrated section schema

The `sections` strategy **doesn't assume any fixed document structure**. Before it
can be used for a `document_type`, a *schema* must be calibrated (section name →
text patterns that mark its start), which is done automatically by `build_indices`
(using the documents themselves as a sample) or manually via
`register_document_type`. The calibrated schema is cached in memory
(`self.section_schemas[document_type]`) and persisted to disk
(`{document_type}_section_schema.json`), so it only needs to be calibrated once per
type.

The call asks for it deterministically (`temperature=0` on the built-in
`OpenAICompatibleChatProvider`) and for at most `constants.MAX_SECTIONS_PER_SCHEMA`
sections, named in snake_case. Both matter more than they look: the schema is written to
disk and then decides how every document of that type is cut, so a sampled answer meant
the same corpus segmented differently on different days — and every extra section is one
more vector per document, splitting a document's content thinner rather than describing
it better.

If the LLM can't identify any section (a document with no recognizable structure),
the `sections` strategy is simply skipped for that type — `full` and `chunks` keep
working normally.

A calibration **call that fails** (timeout, connection error, a response that isn't the
JSON asked for) is a different thing entirely, and is treated as one: it raises
`SectionSchemaError`, nothing is cached, and `build_indices` logs the error, builds
`full`/`chunks` as usual and leaves the `sections` files of the previous build exactly
where they are — they then describe the corpus of the last successful build, which is
warned about, and the next build rebuilds them. Reading a failed call as "no sections
here" would cache that answer for the life of the instance (also stopping the schema on
disk from ever being loaded again) and delete a sections index that costs one embedding
call per section to rebuild.

### Alternative: sections from the document's own structure

Pattern matching (above) works on already-flattened text, so it can't see real
document structure — heading hierarchy, tables, layout. For that, pass a
`structure_provider` on the constructor: `sections` is then built from each document's
*actual* structure, with no schema to calibrate at all.

```python
class MyLayoutParser:                       # any object with this one method
    def extract_sections(self, file_path: str) -> dict[str, str]:
        return {"introducao": "...", "conclusao": "..."}

idx = FaissDocumentIndex(base_path="./data", structure_provider=MyLayoutParser())
```

There's no default and no built-in implementation (same reasoning as
`rerank_provider`): a layout parser is a heavy dependency with its own failure modes,
and which one fits depends on your documents. A previous version shipped one built on
Docling; it was removed once the local OCR pipeline proved better on this corpus, and
it's in the history (`git log -- src/faiss_index/providers.py`) if you want it as a
starting point.

Two things that cost real time to find out, if you write your own:

- A parser reads the original file, not the text `read_document` already flattened, so
  each file gets processed twice when this is configured — once through the usual
  pdfplumber/OCR path for `full`/`chunks`, once through your parser for `sections`.
- A torch-based parser bundles its own OpenMP runtime, and loading it in the same
  process as `faiss` segfaulted on Python 3.14/macOS until `KMP_DUPLICATE_LIB_OK=TRUE`
  and `OMP_NUM_THREADS=1` were set before importing it.

Whatever it returns is used as-is: a document it can't parse into headings is best
returned as a single entry with the flat text, which degrades the same way as an
LLM-calibrated schema that finds no structure.
