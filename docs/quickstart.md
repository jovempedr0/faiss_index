# Quickstart

All progress/status messages go through Python's standard `logging` (module-level
`logger`, one per file) — nothing is printed to stdout on its own. Configure a handler
before using the library if you want to see them:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

```python
from faiss_index import FaissDocumentIndex

idx = FaissDocumentIndex(base_path="./data")

# Builds the indices (calibrates the section schema automatically the first time)
idx.build_indices(
    document_type="contract",
    base_data_dir="./data",
    output_index_dir="./faiss_index",
)

# Search on a specific strategy
result = idx.evaluate_strategy(
    query="termination clause",
    document_type="contract",
    strategy="sections",
    k=5,
)
for r in result["results"]:
    print(r["rank"], r["distance"], r["metadata"]["file"])

# In a new session: load the already-built indices
idx2 = FaissDocumentIndex(base_path="./data")
idx2.load_indices(
    path_indices="./faiss_index",
    document_types=["contract"],
    strategies=["full", "sections", "chunks"],
)
```
