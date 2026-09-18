import faiss
import numpy as np
import pytest


def test_auto_selects_flat_for_small_corpus(make_index):
    idx = make_index(auto_index_thresholds=(10, 20))
    assert idx._select_index_kind(5) == "flat"
    assert idx._select_index_kind(10) == "flat"


def test_auto_selects_ivf_flat_for_medium_corpus(make_index):
    idx = make_index(auto_index_thresholds=(10, 20))
    assert idx._select_index_kind(11) == "ivf_flat"
    assert idx._select_index_kind(20) == "ivf_flat"


def test_auto_selects_ivf_sq8_for_large_corpus(make_index):
    # Regression test: the large tier used to be "ivf_pq", whose default 8-byte codes
    # kept only 48% of the exact top-10 on real embeddings (ivf_sq8: 95%).
    idx = make_index(auto_index_thresholds=(10, 20))
    assert idx._select_index_kind(21) == "ivf_sq8"


def test_ivf_pq_is_still_built_when_asked_for_explicitly(make_index):
    idx = make_index(index_type="ivf_pq", auto_index_thresholds=(10, 20))
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


def _clustered_vectors(n, dim, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(8, dim)).astype("float32")
    return centers[rng.integers(0, 8, n)] + 0.1 * rng.normal(size=(n, dim)).astype("float32")


def test_ivf_sq8_builds_a_searchable_reconstructable_index(make_index):
    idx = make_index(embedding_dim=8, index_type="ivf_sq8", ivf_nprobe=3)
    embeddings = _clustered_vectors(idx.MIN_SQ8_TRAINING_POINTS, 8)

    index = idx._build_faiss_index(embeddings)

    assert isinstance(index, faiss.IndexIVFScalarQuantizer)
    assert index.ntotal == len(embeddings) and index.nprobe == 3
    indices, _distances = idx._dense_search_indices(index, embeddings[[42]], 1)
    assert indices[0] == 42
    # 8-bit codes: reconstruct (evaluate_strategy's "similarity") is close, not exact.
    assert np.allclose(index.reconstruct(42), embeddings[42], atol=0.05)


def test_ivf_sq8_falls_back_to_ivf_flat_when_corpus_too_small_to_learn_its_ranges(make_index):
    # FAISS trains a scalar quantizer on any number of points without complaint, but the
    # per-dimension ranges learned from a handful are too narrow for vectors added later
    # (clipped to them) — e.g. trained on 1 real embedding, then given 10.8k more, it kept
    # 0.2% of the exact top-10.
    idx = make_index(embedding_dim=8, index_type="ivf_sq8")
    embeddings = _clustered_vectors(idx.MIN_SQ8_TRAINING_POINTS - 1, 8)

    index = idx._build_faiss_index(embeddings)

    assert not isinstance(index, faiss.IndexIVFScalarQuantizer)
    assert isinstance(index, faiss.IndexIVFFlat)
    assert index.ntotal == len(embeddings)


def test_ivf_sq8_index_saves_loads_and_takes_new_documents(make_index, tmp_path):
    builder = make_index(embedding_dim=8, index_type="ivf_sq8", ivf_nprobe=2)
    # Distinct character sums, so the fake provider's vectors span its whole value range.
    texts = [chr(0x4E00 + i) for i in range(builder.MIN_SQ8_TRAINING_POINTS)]
    embeddings = builder.get_embeddings(texts)
    metadata = [{"file": f"f{i}.txt", "chunk_index": 0, "type": "chunk", "chunk_text": t} for i, t in enumerate(texts)]
    (tmp_path / "out" / "contrato").mkdir(parents=True)
    builder._create_and_save_index("chunks", embeddings, metadata, "contrato", tmp_path / "out" / "contrato")

    idx = make_index(embedding_dim=8, ivf_nprobe=5)
    idx.load_indices(str(tmp_path / "out"), ["contrato"], ["chunks"], use_gpu=False)
    index = idx.indices["contrato"]["chunks"][0]
    assert isinstance(index, faiss.IndexIVFScalarQuantizer)
    assert index.nprobe == 5  # search-time setting comes from the loading instance

    idx.add_new_documents("contrato", [("novo.txt", "conteudo novo")])

    assert index.ntotal == len(texts) + 1
    assert np.allclose(index.reconstruct(len(texts)), idx.get_embeddings(["conteudo novo"])[0], atol=0.5)
    top = idx.evaluate_strategy("conteudo novo", "contrato", "chunks", k=1)["results"][0]
    assert top["similarity"] == pytest.approx(1.0, abs=1e-3)
