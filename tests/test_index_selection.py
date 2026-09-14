import faiss
import numpy as np


def test_auto_selects_flat_for_small_corpus(make_index):
    idx = make_index(auto_index_thresholds=(10, 20))
    assert idx._select_index_kind(5) == "flat"
    assert idx._select_index_kind(10) == "flat"


def test_auto_selects_ivf_flat_for_medium_corpus(make_index):
    idx = make_index(auto_index_thresholds=(10, 20))
    assert idx._select_index_kind(11) == "ivf_flat"
    assert idx._select_index_kind(20) == "ivf_flat"


def test_auto_selects_ivf_pq_for_large_corpus(make_index):
    idx = make_index(auto_index_thresholds=(10, 20))
    assert idx._select_index_kind(21) == "ivf_pq"


def test_fixed_index_type_ignores_corpus_size(make_index):
    idx = make_index(index_type="flat", auto_index_thresholds=(10, 20))
    assert idx._select_index_kind(10_000) == "flat"


def test_compute_nlist_uses_sqrt_by_default(make_index):
    idx = make_index()
    # sqrt(900) = 30, well under the 900 // 30 = 30 training-point cap.
    assert idx._compute_nlist(900) == 30


def test_compute_nlist_respects_explicit_value_when_enough_data(make_index):
    idx = make_index(ivf_nlist=5)
    assert idx._compute_nlist(1000) == 5


def test_compute_nlist_clamped_by_min_training_points(make_index):
    idx = make_index(ivf_nlist=50)
    # Only 100 vectors and MIN_TRAINING_POINTS_PER_CLUSTER=30 -> at most 100 // 30 = 3 clusters.
    assert idx._compute_nlist(100) == 3


def test_compute_nlist_never_below_one(make_index):
    idx = make_index(ivf_nlist=50)
    assert idx._compute_nlist(1) == 1


def test_ivf_pq_falls_back_to_ivf_flat_when_corpus_too_small_to_train_pq(make_index):
    # Regression test: IndexIVFPQ's product-quantizer training needs at least
    # 2**pq_nbits training points, independent of nlist — a much higher floor than
    # _compute_nlist's IVFFlat-only minimum (MIN_TRAINING_POINTS_PER_CLUSTER per
    # cluster). Forcing index_type="ivf_pq" on a corpus smaller than that used to
    # crash build_indices with a FAISS C++ RuntimeError instead of a warning.
    idx = make_index(embedding_dim=8, index_type="ivf_pq", pq_nbits=8)  # needs 2**8 = 256 points
    embeddings = np.random.rand(5, 8).astype("float32")

    index = idx._build_faiss_index(embeddings)

    assert not isinstance(index, faiss.IndexIVFPQ)
    assert isinstance(index, faiss.IndexIVFFlat)
    assert index.ntotal == 5


def test_ivf_pq_builds_normally_when_corpus_is_large_enough(make_index):
    idx = make_index(embedding_dim=8, index_type="ivf_pq", pq_nbits=4)  # needs 2**4 = 16 points
    embeddings = np.random.rand(20, 8).astype("float32")

    index = idx._build_faiss_index(embeddings)

    assert isinstance(index, faiss.IndexIVFPQ)
    assert index.ntotal == 20
