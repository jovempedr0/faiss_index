import unicodedata

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


def test_tokenize_for_bm25_folds_accents_including_stopwords(make_index):
    idx = make_index()
    expected = ["execucao", "sentenca", "valida"]
    assert idx._tokenize_for_bm25("A execução da sentença não é válida") == expected
    assert idx._tokenize_for_bm25("A execucao da sentenca nao e valida") == expected
    # Decomposed (NFD) accents, as some PDF text extraction emits them.
    assert idx._tokenize_for_bm25(unicodedata.normalize("NFD", "A execução da sentença não é válida")) == expected


def test_bm25_matches_accentless_query_against_accented_corpus(make_index):
    idx = make_index(embedding_dim=4)
    _build_chunks_index(idx, ["pedido de execução da sentença", "outro texto qualquer", "mais um texto"])

    bm25 = idx._get_bm25_index("doctype", "chunks")
    scores = bm25.get_scores(idx._tokenize_for_bm25("execucao sentenca"))
    assert scores[0] > 0
    assert scores[1] == scores[2] == 0


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
    monkeypatch.setattr(idx, "get_embeddings", lambda texts, prefix="": np.array([[1.0, 0.0]], dtype="float32"))

    query = "0829366-83.2025.8.14.0301"

    dense_only = idx.evaluate_strategy(query, "doctype", "chunks", k=3)
    dense_ranked_files = [r["metadata"]["file"] for r in dense_only["results"]]
    assert dense_ranked_files[-1] == "doc_b.txt"

    hybrid = idx.evaluate_strategy_hybrid(query, "doctype", "chunks", k=2)
    hybrid_files = [r["metadata"]["file"] for r in hybrid["results"]]
    assert "doc_b.txt" in hybrid_files


def test_evaluate_strategy_hybrid_excludes_zero_score_documents_from_bm25_pool(make_index):
    # Regression test: bm25_indices used to be built with a plain
    # np.argsort(bm25_scores)[::-1][:pool], with no floor on the score — so once `pool`
    # reaches the corpus size (as it does here: default pool=max(k*4,50), capped at
    # ntotal=5), documents sharing NO term with the query still got included (and thus
    # a bm25_rank / RRF contribution) purely by filling out the slice, diluting the
    # whole point of the lexical signal.
    metadata = [
        {"file": "doc_0.txt", "type": "chunk", "chunk_text": "zebra selvagem correndo"},
        {"file": "doc_1.txt", "type": "chunk", "chunk_text": "elefante grande andando"},
        {"file": "doc_2.txt", "type": "chunk", "chunk_text": "leao forte rugindo"},
        {"file": "doc_3.txt", "type": "chunk", "chunk_text": "tigre listrado cacando"},
        {"file": "doc_4.txt", "type": "chunk", "chunk_text": "urso marrom dormindo"},
    ]
    idx = make_index(embedding_dim=4)
    embeddings = idx.get_embeddings([m["chunk_text"] for m in metadata])
    index = faiss.IndexFlatL2(4)
    index.add(embeddings.astype("float32"))
    idx.indices["doctype"] = {"chunks": (index, metadata, embeddings)}

    results = idx.evaluate_strategy_hybrid("zebra", "doctype", "chunks", k=5)

    by_file = {r["metadata"]["file"]: r for r in results["results"]}
    assert by_file["doc_0.txt"]["bm25_rank"] == 0
    for other_file in ("doc_1.txt", "doc_2.txt", "doc_3.txt", "doc_4.txt"):
        assert by_file[other_file]["bm25_rank"] is None


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


def test_reloading_a_strategy_invalidates_bm25_cache(make_index, fake_chat_provider, tmp_path):
    # Regression test: load_indices() overwrites self.indices[document_type][strategy]
    # directly (unlike add_new_documents/unload_indices, both of which pop the cached
    # BM25 index too) — reloading a strategy already in memory, without unload_indices
    # first, used to leave evaluate_strategy_hybrid's BM25 index built from the
    # *previous* corpus while self.indices already pointed at the new one, silently
    # attributing stale BM25 ranks/scores to the wrong documents.
    document_type = "doctype"
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "out"
    (data_dir / document_type).mkdir(parents=True)
    (data_dir / document_type / "a.txt").write_text("gato preto no quintal", encoding="utf-8")
    (data_dir / document_type / "b.txt").write_text("cachorro branco latindo", encoding="utf-8")

    idx = make_index(embedding_dim=8)
    fake_chat_provider.responses.append({"sections": []})  # no structure -> skip "sections"
    idx.build_indices(document_type=document_type, base_data_dir=str(data_dir), output_index_dir=str(output_dir))
    idx.load_indices(path_indices=str(output_dir), document_types=[document_type], strategies=["chunks"], use_gpu=False)

    idx.evaluate_strategy_hybrid("gato preto", document_type, "chunks", k=5)  # warms the BM25 cache
    assert "chunks" in idx._bm25_indices[document_type]

    # Corpus on disk changes completely; rebuilt and reloaded on the SAME instance,
    # with no unload_indices() in between.
    for f in (data_dir / document_type).iterdir():
        f.unlink()
    (data_dir / document_type / "c.txt").write_text("processo judicial numero um", encoding="utf-8")
    (data_dir / document_type / "d.txt").write_text("contrato de prestacao de servicos", encoding="utf-8")

    idx.build_indices(document_type=document_type, base_data_dir=str(data_dir), output_index_dir=str(output_dir))
    idx.load_indices(path_indices=str(output_dir), document_types=[document_type], strategies=["chunks"], use_gpu=False)

    assert "chunks" not in idx._bm25_indices.get(document_type, {})  # invalidated by the reload

    results = idx.evaluate_strategy_hybrid("gato preto", document_type, "chunks", k=5)
    found_files = {r["metadata"]["file"] for r in results["results"]}
    assert found_files == {
        str(data_dir / document_type / "c.txt"),
        str(data_dir / document_type / "d.txt"),
    }


def test_rebuilding_a_document_type_invalidates_bm25_cache(make_index, fake_chat_provider, tmp_path):
    # Regression test: build_indices() replaces self.indices[document_type] with the
    # freshly built indices (no load_indices() needed to search them) but left
    # evaluate_strategy_hybrid's cached BM25 index built from the *previous* corpus.
    # BM25 positions from the old corpus were then looked up in the new, shorter
    # metadata list: wrong documents at best, IndexError at worst.
    document_type = "doctype"
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "out"
    (data_dir / document_type).mkdir(parents=True)
    (data_dir / document_type / "a.txt").write_text("gato preto no quintal", encoding="utf-8")
    (data_dir / document_type / "b.txt").write_text("cachorro branco latindo", encoding="utf-8")
    (data_dir / document_type / "c.txt").write_text("cavalo marrom correndo", encoding="utf-8")

    idx = make_index(embedding_dim=8)
    fake_chat_provider.responses.append({"sections": []})  # no structure -> skip "sections"
    idx.build_indices(document_type=document_type, base_data_dir=str(data_dir), output_index_dir=str(output_dir))
    idx.evaluate_strategy_hybrid("cavalo marrom", document_type, "chunks", k=5)  # BM25 cache over 3 chunks

    for f in (data_dir / document_type).iterdir():
        f.unlink()
    (data_dir / document_type / "d.txt").write_text("processo judicial numero um", encoding="utf-8")
    idx.build_indices(document_type=document_type, base_data_dir=str(data_dir), output_index_dir=str(output_dir))

    results = idx.evaluate_strategy_hybrid("cavalo marrom", document_type, "chunks", k=5)

    assert [r["metadata"]["file"] for r in results["results"]] == [str(data_dir / document_type / "d.txt")]
    assert idx._bm25_indices[document_type]["chunks"].corpus_size == 1
