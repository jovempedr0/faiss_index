# Indexing

## Multiple document types on the same instance

A single instance indexes and searches across several `document_type`s at once —
each type's section schema, indices, and embeddings stay isolated from one another:

```python
idx = FaissDocumentIndex(base_path="./data")

for doc_type in ["contract", "invoice"]:
    idx.build_indices(document_type=doc_type, base_data_dir="./data", output_index_dir="./faiss_index")

contract_result = idx.evaluate_strategy("termination clause", document_type="contract", strategy="sections")
invoice_result = idx.evaluate_strategy("total amount", document_type="invoice", strategy="chunks")
```

## Calibrating the section schema manually

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

## Adding documents without rebuilding the index

To incorporate new documents into indices already loaded in memory (this doesn't
rebuild the existing ones, it just adds vectors/metadata to each strategy's index
already present for the `document_type`):

```python
idx.load_indices(path_indices="./faiss_index", document_types=["contract"], strategies=["full", "chunks"])

new_docs = [("./data/contract/2026-01/new.pdf", extracted_text)]
idx.add_new_documents(document_type="contract", new_docs=new_docs)
idx.save_indices(document_type="contract", output_index_dir="./faiss_index")  # persist
```

The new documents' chunks are embedded once and shared by `full` and `chunks` —
with only `full` loaded, they're still embedded (one call per ~250 words, not one per
document), since that's what the `full` vector is pooled from.

`add_new_documents` updates the indices **in memory**; `save_indices` writes every
loaded strategy of the document type back to disk (index, metadata, embedding rows —
including the added ones — and the section schema), in the layout `load_indices`
reads. Saving over the directory the indices were loaded from is safe. Reloading or
unloading a strategy before saving discards what was added to it.

## Indexing text extracted elsewhere

Everything above assumes this library extracts the text (`read_document`: direct read
for `.txt`, `pdfplumber` + OCR fallback for the rest). When another pipeline already
does that — a different OCR engine, a document-understanding API, a cache or database
filled by an earlier job — pass a `text_extractor` and it is used instead, for every
file:

```python
def extract(file_path: str) -> str:
    return my_ocr_service.extract(file_path)  # or: cache[file_path], db.fetch(...), ...

idx = FaissDocumentIndex(base_path="./data", text_extractor=extract)
idx.build_indices(document_type="contract", base_data_dir="./data", output_index_dir="./faiss_index")
```

`build_indices` still walks the directory and still records the real file path in each
entry's `metadata["file"]` — only the extraction step changes, so chunking, sections,
search and the on-disk layout behave exactly as they do otherwise. Nothing here opens
the files, so the `[ocr]` extra isn't needed and neither is a working tesseract.
An extractor that raises for one file is treated like any other read error: that
document is logged and skipped, the build carries on.

For documents arriving one at a time into an index that's already loaded, the same text
can go straight to
[`add_new_documents`](#adding-documents-without-rebuilding-the-index), which takes
`(file_path, text)` pairs and never reads the file either.
