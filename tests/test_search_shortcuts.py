import numpy as np
import faiss


def _build_chunks_index(idx, texts):
    metadata = [
        {"file": f"doc_{i}.txt", "type": "chunk", "chunk_text": text}
        for i, text in enumerate(texts)
    ]
    embeddings = idx.get_embeddings(texts)
    index = faiss.IndexFlatL2(idx.embedding_dim)
    index.add(embeddings.astype("float32"))
    idx.indices["doctype"] = {"chunks": (index, metadata, embeddings)}
    return metadata


def test_calculate_heuristic_score_handles_dense_results(make_index):
    idx = make_index(embedding_dim=4)
    _build_chunks_index(idx, ["primeiro texto", "segundo texto", "terceiro texto"])
    dense_results = idx.evaluate_strategy("um texto qualquer", "doctype", "chunks", k=3)

    scores = idx.calculate_heuristic_score({"query": {"chunks": dense_results}})

    assert "chunks" in scores
    assert scores["chunks"]["mean_score"] >= 0


def test_calculate_heuristic_score_handles_hybrid_results_without_keyerror(make_index):
    idx = make_index(embedding_dim=4)
    _build_chunks_index(idx, ["primeiro texto", "segundo texto", "terceiro texto"])
    hybrid_results = idx.evaluate_strategy_hybrid("um texto qualquer", "doctype", "chunks", k=3)

    # Before the fix, this raised KeyError: 'avg_distance' (evaluate_strategy_hybrid's
    # results don't have it — only per-item "rrf_score").
    scores = idx.calculate_heuristic_score({"query": {"chunks": hybrid_results}})

    assert "chunks" in scores
    assert scores["chunks"]["mean_score"] >= 0


def test_compare_strategies_dense_by_default(make_index):
    idx = make_index(embedding_dim=4)
    _build_chunks_index(idx, ["primeiro texto", "segundo texto"])

    comparison = idx.compare_strategies(["primeiro"], "doctype", ["chunks"], k=2)

    chunks_result = comparison["primeiro"]["chunks"]
    assert "avg_distance" in chunks_result
    assert "distance" in chunks_result["results"][0]
    assert "rrf_score" not in chunks_result["results"][0]


def test_compare_strategies_use_hybrid(make_index):
    idx = make_index(embedding_dim=4)
    _build_chunks_index(idx, ["primeiro texto", "segundo texto"])

    comparison = idx.compare_strategies(["primeiro"], "doctype", ["chunks"], k=2, use_hybrid=True)

    chunks_result = comparison["primeiro"]["chunks"]
    assert "avg_distance" not in chunks_result
    assert "rrf_score" in chunks_result["results"][0]


def test_generate_search_by_type_use_hybrid_recovers_exact_term(make_index, monkeypatch):
    idx = make_index(embedding_dim=2)
    # Same 3-document setup as test_hybrid_search.py's controlled scenario: doc_b (the
    # exact-term match) sits far from the query in this fake embedding space, so a
    # dense-only top-2 would miss it entirely — only BM25+RRF brings it back in. (A
    # 2-document version of this is a coin flip: with only two candidates, dense and
    # BM25 rankings are exact opposites, so their RRF contributions tie exactly and the
    # winner comes down to sort stability rather than the fusion actually mattering.)
    metadata = [
        {"file": "doc_a.txt", "type": "chunk", "chunk_text": "informacoes gerais sobre contratos"},
        {"file": "doc_b.txt", "type": "chunk", "chunk_text": "processo numero 0829366-83.2025.8.14.0301"},
        {"file": "doc_c.txt", "type": "chunk", "chunk_text": "outro documento qualquer sem relacao"},
    ]
    embeddings = np.array([[1.0, 0.0], [0.0, 100.0], [0.9, 0.1]], dtype="float32")
    index = faiss.IndexFlatL2(2)
    index.add(embeddings)
    idx.indices["doctype"] = {"chunks": (index, metadata, embeddings)}
    monkeypatch.setattr(idx, "get_embeddings", lambda texts: np.array([[1.0, 0.0]], dtype="float32"))

    dense_only = idx.generate_search_by_type(
        "0829366-83.2025.8.14.0301", "doctype", "chunks", require_gpu=False, k=2, use_hybrid=False
    )
    assert "processo numero 0829366-83.2025.8.14.0301" not in dense_only

    hybrid = idx.generate_search_by_type(
        "0829366-83.2025.8.14.0301", "doctype", "chunks", require_gpu=False, k=2, use_hybrid=True
    )
    assert "processo numero 0829366-83.2025.8.14.0301" in hybrid


def test_generate_search_by_type_rerank_narrows_to_k(make_index, fake_rerank_provider):
    idx = make_index(embedding_dim=4, rerank_provider=fake_rerank_provider)
    _build_chunks_index(idx, ["texto irrelevante", "texto muito relevante", "texto medio"])
    fake_rerank_provider.scores_by_candidate = {
        "texto irrelevante": 0.1,
        "texto muito relevante": 0.9,
        "texto medio": 0.5,
    }

    chunks = idx.generate_search_by_type(
        "qualquer coisa", "doctype", "chunks", require_gpu=False, k=2, rerank=True
    )

    assert chunks == ["texto muito relevante", "texto medio"]


def test_generate_search_by_type_default_k_is_five(make_index):
    idx = make_index(embedding_dim=4)
    _build_chunks_index(idx, [f"texto numero {i}" for i in range(8)])

    chunks = idx.generate_search_by_type("texto", "doctype", "chunks", require_gpu=False)

    assert len(chunks) == 5
