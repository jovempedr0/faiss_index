"""
FAISS CUDA GPU capability detection, shared by FaissDocumentIndex's constructor
(`core.py`) and its `_lifecycle.py` mixin. Split out on its own so that mixin module
doesn't need to import from `core.py` (which imports it back, to compose the class) —
importing this instead avoids a circular import.
"""
import faiss

# faiss-cpu (the only variant installable on macOS) doesn't expose these attributes —
# only faiss-gpu (Linux/CUDA) has them. Checking the installed build's capability, instead
# of trying StandardGpuResources()/index_cpu_to_gpu() and relying on the except to find out
# it doesn't exist, avoids a "failed to move to GPU" warning on every index loaded on any
# machine without CUDA (macOS included).
FAISS_HAS_GPU_SUPPORT = all(
    hasattr(faiss, attr) for attr in ("StandardGpuResources", "index_cpu_to_gpu", "GpuIndex")
)
