import pytest


def _fake_results(texts):
    return [
        {"rank": i + 1, "metadata": {"file": f"doc_{i}.txt", "type": "chunk", "chunk_text": text}}
        for i, text in enumerate(texts)
    ]


def test_rerank_results_reorders_by_score_descending(make_index, fake_rerank_provider):
    idx = make_index(rerank_provider=fake_rerank_provider)
    results = _fake_results(["texto irrelevante", "texto muito relevante", "texto pouco relevante"])
    fake_rerank_provider.scores_by_candidate = {
        "texto irrelevante": 0.1,
        "texto muito relevante": 0.9,
        "texto pouco relevante": 0.5,
    }

    reranked = idx.rerank_results("query qualquer", results)

    assert [r["metadata"]["chunk_text"] for r in reranked] == [
        "texto muito relevante",
        "texto pouco relevante",
        "texto irrelevante",
    ]
    assert [r["rank"] for r in reranked] == [1, 2, 3]
    assert [r["rerank_score"] for r in reranked] == [0.9, 0.5, 0.1]


def test_rerank_results_truncates_to_k(make_index, fake_rerank_provider):
    idx = make_index(rerank_provider=fake_rerank_provider)
    results = _fake_results(["a", "b", "c"])
    fake_rerank_provider.scores_by_candidate = {"a": 0.1, "b": 0.9, "c": 0.5}

    reranked = idx.rerank_results("query", results, k=2)

    assert len(reranked) == 2
    assert [r["metadata"]["chunk_text"] for r in reranked] == ["b", "c"]


def test_rerank_results_passes_extracted_text_to_provider(make_index, fake_rerank_provider):
    idx = make_index(rerank_provider=fake_rerank_provider)
    results = _fake_results(["primeiro", "segundo"])

    idx.rerank_results("minha query", results)

    assert len(fake_rerank_provider.calls) == 1
    query, candidates = fake_rerank_provider.calls[0]
    assert query == "minha query"
    assert candidates == ["primeiro", "segundo"]


def test_rerank_results_empty_input_returns_empty_without_calling_provider(make_index, fake_rerank_provider):
    idx = make_index(rerank_provider=fake_rerank_provider)

    assert idx.rerank_results("query", []) == []
    assert fake_rerank_provider.calls == []


def test_rerank_results_without_provider_configured_raises(make_index):
    idx = make_index()  # rerank_provider defaults to None

    with pytest.raises(ValueError):
        idx.rerank_results("query", _fake_results(["a", "b"]))
