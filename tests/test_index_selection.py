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
