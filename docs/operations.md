# File layout and testing

## On-disk file layout

```
<output_index_dir>/
  <document_type>/
    <document_type>_section_schema.json
    <document_type>_full.index
    <document_type>_full_metadata.json
    <document_type>_full_embeddings.npy
    <document_type>_full_embedding.json     # embedding_document_prefix used at build time
    <document_type>_sections.index          # if a schema was calibrated
    <document_type>_sections_metadata.json
    <document_type>_sections_embeddings.npy
    <document_type>_sections_embedding.json
    <document_type>_chunks.index
    <document_type>_chunks_metadata.json
    <document_type>_chunks_embeddings.npy
    <document_type>_chunks_embedding.json
```

The `*_embedding.json` files are optional on load (indices saved before they existed
are treated as built without a document prefix).

`load_indices(path_indices=..., document_types=[...])` expects exactly this layout
(one subfolder per `document_type` inside `path_indices`); `build_indices` and
`save_indices` write it.

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
are needed just to import the package, not only for real OCR calls. `rerank` isn't
installed: it's imported lazily inside `CrossEncoderRerankProvider`, which the unit
suite never instantiates. The integration test stays excluded, same as running `pytest` locally.
