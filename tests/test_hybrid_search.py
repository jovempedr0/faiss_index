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


def test_evaluate_strategy_hybrid_missing_index_returns_empty_dict(make_index):
    idx = make_index()
    assert idx.evaluate_strategy_hybrid("query", "nao_existe", "chunks") == {}


def test_tokenize_for_bm25_normalizes_like_clean_text(make_index):
    idx = make_index()
    assert idx._tokenize_for_bm25("Contratos, PAGAMENTOS!!") == ["contratos", "pagamentos"]


def test_get_bm25_index_is_cached(make_index):
    idx = make_index(embedding_dim=4)
    _build_chunks_index(idx, ["um texto qualquer", "outro texto diferente"])

    first = idx._get_bm25_index("doctype", "chunks")
    second = idx._get_bm25_index("doctype", "chunks")
    assert first is second


def test_hybrid_search_recovers_exact_term_missed_by_dense(make_index, monkeypatch):
    idx = make_index(embedding_dim=2)

    metadata = [
        {"file": "doc_a.txt", "type": "chunk", "chunk_text": "informacoes gerais sobre contratos e pagamentos"},
        {"file": "doc_b.txt", "type": "chunk", "chunk_text": "processo numero 0829366-83.2025.8.14.0301 decisao"},
        {"file": "doc_c.txt", "type": "chunk", "chunk_text": "outro documento qualquer sem relacao"},
    ]
    # Hand-picked embeddings (bypassing the fake provider's hash-based ones, via the
    # monkeypatch below): doc_b — the one with the exact process number — sits FAR from
    # the query in this fake embedding space, so dense search alone ranks it last.
    # Only BM25 (exact term match, both sides normalized the same way by
    # _tokenize_for_bm25) can surface it.
    embeddings = np.array([
        [1.0, 0.0],
        [0.0, 100.0],
        [0.9, 0.1],
    ], dtype="float32")

    index = faiss.IndexFlatL2(2)
    index.add(embeddings)
    idx.indices["doctype"] = {"chunks": (index, metadata, embeddings)}
    monkeypatch.setattr(idx, "get_embeddings", lambda texts: np.array([[1.0, 0.0]], dtype="float32"))

    query = "0829366-83.2025.8.14.0301"

    dense_only = idx.evaluate_strategy(query, "doctype", "chunks", k=3)
    dense_ranked_files = [r["metadata"]["file"] for r in dense_only["results"]]
    assert dense_ranked_files[-1] == "doc_b.txt"

    hybrid = idx.evaluate_strategy_hybrid(query, "doctype", "chunks", k=2)
    hybrid_files = [r["metadata"]["file"] for r in hybrid["results"]]
    assert "doc_b.txt" in hybrid_files


def test_add_new_documents_invalidates_bm25_cache(make_index):
    idx = make_index(embedding_dim=4)
    _build_chunks_index(idx, ["texto original um", "texto original dois"])

    idx.evaluate_strategy_hybrid("original", "doctype", "chunks", k=5)  # builds+caches BM25
    assert "chunks" in idx._bm25_indices["doctype"]

    idx.add_new_documents("doctype", new_docs=[("novo.txt", "termo exclusivo novidade")])
    assert "chunks" not in idx._bm25_indices.get("doctype", {})  # invalidated

    results = idx.evaluate_strategy_hybrid("novidade", "doctype", "chunks", k=5)
    found_files = {r["metadata"]["file"] for r in results["results"]}
    assert "novo.txt" in found_files


def test_unload_indices_invalidates_bm25_cache(make_index):
    idx = make_index(embedding_dim=4)
    _build_chunks_index(idx, ["um texto qualquer", "outro texto diferente"])

    idx.evaluate_strategy_hybrid("texto", "doctype", "chunks", k=5)
    assert "doctype" in idx._bm25_indices

    idx.unload_indices("doctype")
    assert "doctype" not in idx._bm25_indices
