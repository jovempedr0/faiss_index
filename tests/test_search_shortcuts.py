import numpy as np
import faiss
import pytest

from faiss_index import config


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

    with pytest.deprecated_call():
        scores = idx.calculate_heuristic_score({"query": {"chunks": dense_results}})

    assert "chunks" in scores
    assert scores["chunks"]["mean_score"] >= 0


def test_calculate_heuristic_score_handles_hybrid_results_without_keyerror(make_index):
    idx = make_index(embedding_dim=4)
    _build_chunks_index(idx, ["primeiro texto", "segundo texto", "terceiro texto"])
    hybrid_results = idx.evaluate_strategy_hybrid("um texto qualquer", "doctype", "chunks", k=3)

    # Before the fix, this raised KeyError: 'avg_distance' (evaluate_strategy_hybrid's
    # results don't have it — only per-item "rrf_score").
    with pytest.deprecated_call():
        scores = idx.calculate_heuristic_score({"query": {"chunks": hybrid_results}})

    assert "chunks" in scores
    assert scores["chunks"]["mean_score"] >= 0


def test_evaluate_retrieval_reports_recall_mrr_passage_recall_and_result_size(make_index, monkeypatch):
    idx = make_index(embedding_dim=2)
    metadata = [
        {"file": "a.txt", "type": "chunk", "chunk_text": "clausula de rescisao sem multa"},
        {"file": "b.txt", "type": "chunk", "chunk_text": "pagamento  em\nduas parcelas"},
        {"file": "c.txt", "type": "chunk", "chunk_text": "foro da comarca"},
    ]
    embeddings = np.array([[0.0, 1.0], [2.0, 0.0], [5.0, 5.0]], dtype="float32")
    index = faiss.IndexFlatL2(2)
    index.add(embeddings)
    idx.indices["doctype"] = {"chunks": (index, metadata, embeddings)}
    query_vectors = {"rescisao": [0.0, 1.0], "parcelas": [0.1, 1.0], "inexistente": [5.0, 5.0]}
    monkeypatch.setattr(idx, "get_embeddings", lambda texts, prefix="": np.array([query_vectors[texts[0]]], dtype="float32"))

    report = idx.evaluate_retrieval(
        [
            {"query": "rescisao", "relevant_files": ["a.txt"], "relevant_text": "rescisao sem multa"},  # rank 1
            {"query": "parcelas", "relevant_files": ["b.txt"], "relevant_text": "em duas parcelas"},    # rank 2
            {"query": "inexistente", "relevant_files": ["zzz.txt"]},                                   # miss
        ],
        document_type="doctype", strategies=["chunks"], k=2,
    )

    chunks = report["chunks"]
    assert chunks["recall_at_k"] == pytest.approx(2 / 3)
    assert chunks["mrr"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert chunks["passage_recall_at_k"] == 1.0  # b.txt's passage matched with whitespace normalized
    assert chunks["n_queries"] == 3
    a, b, c = (len(m["chunk_text"]) for m in metadata)
    assert chunks["avg_result_chars"] == pytest.approx(np.mean([a, b, a, b, c, b]))  # top-2: (a,b) (a,b) (c,b)


def test_calculate_heuristic_score_and_generate_search_are_deprecated(make_index):
    idx = make_index(embedding_dim=4)
    _build_chunks_index(idx, ["primeiro texto", "segundo texto"])
    comparison = idx.compare_strategies(["texto"], "doctype", ["chunks"], k=2)

    with pytest.deprecated_call():
        idx.calculate_heuristic_score(comparison)
    with pytest.deprecated_call():
        idx.generate_search(["texto"], keywords=[], document_type="doctype", strategies_compare=["chunks"])


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
    monkeypatch.setattr(idx, "get_embeddings", lambda texts, prefix="": np.array([[1.0, 0.0]], dtype="float32"))

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


def test_generate_search_by_type_sends_the_unaltered_query_to_embeddings_and_reranker(
    make_index, fake_rerank_provider, monkeypatch
):
    # Regression test: the shortcut clean_text'ed the query before searching, so the
    # embedding model and the cross-encoder got "clausula permite rescisao" for
    # "Cláusula que NÃO permite rescisão?" — punctuation, case and stopwords like
    # "não"/"sem" (which invert meaning) stripped, while documents are embedded as-is.
    idx = make_index(embedding_dim=4, rerank_provider=fake_rerank_provider)
    _build_chunks_index(idx, ["texto um", "texto dois", "texto tres"])
    embedded_texts = []
    real_get_embeddings = idx.get_embeddings
    monkeypatch.setattr(idx, "get_embeddings", lambda texts, prefix="": embedded_texts.extend(texts) or real_get_embeddings(texts, prefix))
    query = "Cláusula que NÃO permite rescisão?"

    idx.generate_search_by_type(query, "doctype", "chunks", require_gpu=False, k=2, use_hybrid=True, rerank=True)

    assert embedded_texts == [query]
    assert fake_rerank_provider.calls[0][0] == query


def test_generate_search_sends_the_unaltered_query_to_embeddings(make_index, monkeypatch):
    idx = make_index(embedding_dim=4)
    _build_chunks_index(idx, ["texto um", "texto dois"])
    embedded_texts = []
    real_get_embeddings = idx.get_embeddings
    monkeypatch.setattr(idx, "get_embeddings", lambda texts, prefix="": embedded_texts.extend(texts) or real_get_embeddings(texts, prefix))
    query = "Decisão sem efeito suspensivo?"

    with pytest.deprecated_call():
        results, _scores = idx.generate_search([query], keywords=[], document_type="doctype", strategies_compare=["chunks"])

    assert embedded_texts == [query]
    assert list(results) == [query]


def test_generate_search_by_type_empty_query_returns_empty_list(make_index):
    idx = make_index(embedding_dim=4)
    _build_chunks_index(idx, ["texto um", "texto dois"])

    # Regression test: clean_text() always returns a 1-element list ([text.strip()]),
    # even when the text becomes empty after stripping punctuation/stopwords — so a
    # query made of nothing but punctuation used to sail past the "empty query" check
    # (which tested the always-truthy wrapping list, not its content) and search with
    # a zero-vector embedding fallback instead of short-circuiting to [].
    chunks = idx.generate_search_by_type("... !!! ,,,", "doctype", "chunks", require_gpu=False)

    assert chunks == []


def _build_on_disk(idx, fake_chat_provider, tmp_path, texts):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "out"
    (data_dir / "doctype").mkdir(parents=True)
    for i, text in enumerate(texts):
        (data_dir / "doctype" / f"doc_{i}.txt").write_text(text, encoding="utf-8")
    fake_chat_provider.responses.append({"sections": []})  # no structure -> skip "sections"
    idx.build_indices(document_type="doctype", base_data_dir=str(data_dir), output_index_dir=str(output_dir))
    return output_dir


def test_generate_search_by_type_loads_index_from_default_path_when_not_in_memory(
    make_index, fake_chat_provider, tmp_path, monkeypatch
):
    output_dir = _build_on_disk(make_index(embedding_dim=4), fake_chat_provider, tmp_path, ["texto um", "texto dois"])
    monkeypatch.setattr(config, "DEFAULT_PATH_INDICES", str(output_dir))
    idx = make_index(embedding_dim=4)  # fresh instance, nothing in memory

    chunks = idx.generate_search_by_type("texto um", "doctype", "chunks", require_gpu=False, k=2)

    assert sorted(chunks) == ["texto dois", "texto um"]


def test_generate_search_by_type_require_gpu_keeps_index_already_in_memory(
    make_index, fake_chat_provider, tmp_path, monkeypatch
):
    # Regression test: with require_gpu=True and no GPU acceleration for the index
    # (no CUDA), the loaded-check always failed, so
    # every call reloaded the index from disk over the in-memory one — silently
    # discarding documents added via add_new_documents since the index was saved.
    idx = make_index(embedding_dim=4)
    output_dir = _build_on_disk(idx, fake_chat_provider, tmp_path, ["texto um", "texto dois"])
    monkeypatch.setattr(config, "DEFAULT_PATH_INDICES", str(output_dir))
    idx.add_new_documents("doctype", [("novo.txt", "documento recem adicionado")])

    chunks = idx.generate_search_by_type("documento recem adicionado", "doctype", "chunks", require_gpu=True, k=1)

    assert chunks == ["documento recem adicionado"]
    assert idx.indices["doctype"]["chunks"][0].ntotal == 3


def test_generate_search_by_type_returns_texts_for_full_and_sections_strategies(make_index, fake_chat_provider, tmp_path):
    # Regression test: the shortcut read metadata["chunk_text"], which only the
    # "chunks" strategy stores — "full" (content) and "sections" (section_text)
    # raised KeyError instead of returning their texts.
    data_dir = tmp_path / "data"
    (data_dir / "doctype").mkdir(parents=True)
    (data_dir / "doctype" / "doc.txt").write_text("Cabecalho\nPedidos finais do autor", encoding="utf-8")
    idx = make_index(embedding_dim=4)
    fake_chat_provider.responses.append({"sections": [{"name": "pedidos", "patterns": ["pedidos"]}]})
    idx.build_indices(document_type="doctype", base_data_dir=str(data_dir), output_index_dir=str(tmp_path / "out"))

    full = idx.generate_search_by_type("pedidos", "doctype", "full", require_gpu=False, k=1)
    sections = idx.generate_search_by_type("pedidos", "doctype", "sections", require_gpu=False, k=2)

    assert full == ["Cabecalho\nPedidos finais do autor"]
    assert sorted(sections) == ["Cabecalho", "Pedidos finais do autor"]


def test_generate_search_by_type_default_k_is_five(make_index):
    idx = make_index(embedding_dim=4)
    _build_chunks_index(idx, [f"texto numero {i}" for i in range(8)])

    chunks = idx.generate_search_by_type("texto", "doctype", "chunks", require_gpu=False)

    assert len(chunks) == 5
