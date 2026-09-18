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


class _ContainsTermScorer:
    """Scores a candidate 1.0 when it contains `term` — stands in for a cross-encoder
    that can only recognize what it is actually shown."""

    def __init__(self, term):
        self.term = term
        self.calls = []

    def rerank(self, query, candidates):
        self.calls.append((query, candidates))
        return [1.0 if self.term in c else 0.0 for c in candidates]


def test_long_candidate_is_scored_by_its_best_passage(make_index):
    # Regression test: a candidate longer than the cross-encoder's window used to be
    # handed over whole, so the model read its opening and nothing else. On the real
    # corpus that took "full"'s recall@5 from 79.7% to 31.2%.
    buried = " ".join(["preambulo"] * 900) + " penhora de veiculo"
    scorer = _ContainsTermScorer("penhora")
    idx = make_index(rerank_provider=scorer)

    reranked = idx.rerank_results("penhora de veiculo", _fake_results(["texto curto", buried]))

    assert reranked[0]["metadata"]["chunk_text"] == buried
    assert reranked[0]["rerank_score"] == 1.0
    scored = scorer.calls[0][1]
    assert buried not in scored  # split into passages, never sent whole
    assert any("penhora" in passage for passage in scored)


def test_long_candidate_costs_a_bounded_number_of_passes(make_index, fake_rerank_provider):
    # Scoring every window of a long candidate is what made this unaffordable: a 118k-char
    # document is ~240 windows, about a minute per query once multiplied by the candidates.
    from faiss_index import constants

    long_text = " ".join(f"palavra{i}" for i in range(3000))
    idx = make_index(rerank_provider=fake_rerank_provider)

    idx.rerank_results("palavra42", _fake_results([long_text]))

    passages = fake_rerank_provider.calls[0][1]
    assert len(passages) == constants.RERANK_PASSAGES_PER_CANDIDATE
    assert all(len(p.split()) <= constants.RERANK_PASSAGE_WORDS for p in passages)
    assert any("palavra42" in p for p in passages)  # BM25 kept the window with the term
