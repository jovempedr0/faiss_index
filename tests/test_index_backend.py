import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def test_use_mps_is_deprecated_and_search_runs_on_faiss(make_index):
    # use_mps used to route "flat" searches through torch on Apple Silicon's GPU; it
    # measured 2.5-7x slower than FAISS's own CPU search at every size tested, so the
    # path was removed and the parameter only warns now.
    with pytest.deprecated_call():
        idx = make_index(embedding_dim=4, use_mps=True)
    index = idx._build_faiss_index(np.eye(4, dtype="float32"))

    indices, distances = idx._dense_search_indices(index, np.array([[0.0, 1.0, 0.0, 0.0]], dtype="float32"), 10)

    assert indices[0] == 1 and distances[0] == 0.0
    assert len(indices) == 4  # FAISS's -1 padding for n > ntotal is filtered out


def test_importing_the_package_does_not_import_torch():
    src_dir = Path(__file__).resolve().parent.parent / "src"
    script = f"import sys; sys.path.insert(0, {str(src_dir)!r}); import faiss_index; print('torch' in sys.modules)"

    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
