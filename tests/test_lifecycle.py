import faiss
import pytest


def _build_full_index(idx, texts):
    docs = [(f"file_{i}.txt", t) for i, t in enumerate(texts)]
    embeddings, metadata = idx.create_embeddings_full(docs)
    faiss_index = faiss.IndexFlatL2(idx.embedding_dim)
    faiss_index.add(embeddings.astype("float32"))
    return faiss_index, metadata, embeddings


def test_is_index_loaded_true_when_present(make_index):
    idx = make_index(embedding_dim=4)
    index, metadata, embeddings = _build_full_index(idx, ["doc um", "doc dois"])
    idx.indices["contrato"] = {"full": (index, metadata, embeddings)}

    assert idx.is_index_loaded(document_types=["contrato"], strategies=["full"], require_gpu=False)


def test_is_index_loaded_false_for_missing_document_type(make_index):
    idx = make_index()
    assert not idx.is_index_loaded(document_types=["inexistente"], strategies=["full"], require_gpu=False)


def test_is_index_loaded_false_for_missing_strategy(make_index):
    idx = make_index(embedding_dim=4)
    index, metadata, embeddings = _build_full_index(idx, ["doc um"])
    idx.indices["contrato"] = {"full": (index, metadata, embeddings)}

    assert not idx.is_index_loaded(document_types=["contrato"], strategies=["chunks"], require_gpu=False)


def test_is_index_loaded_empty_lists_return_false(make_index):
    idx = make_index()
    assert not idx.is_index_loaded(document_types=[], strategies=["full"], require_gpu=False)
    assert not idx.is_index_loaded(document_types=["contrato"], strategies=[], require_gpu=False)


def test_unload_indices_removes_single_strategy(make_index):
    idx = make_index(embedding_dim=4)
    index, metadata, embeddings = _build_full_index(idx, ["doc um"])
    idx.indices["contrato"] = {
        "full": (index, metadata, embeddings),
        "chunks": (index, metadata, embeddings),
    }

    idx.unload_indices("contrato", strategy="full")

    assert "full" not in idx.indices["contrato"]
    assert "chunks" in idx.indices["contrato"]


def test_unload_indices_removes_document_type_when_last_strategy_goes(make_index):
    idx = make_index(embedding_dim=4)
    index, metadata, embeddings = _build_full_index(idx, ["doc um"])
    idx.indices["contrato"] = {"full": (index, metadata, embeddings)}

    idx.unload_indices("contrato", strategy="full")

    assert "contrato" not in idx.indices


def test_unload_indices_removes_all_strategies_when_no_strategy_given(make_index):
    idx = make_index(embedding_dim=4)
    index, metadata, embeddings = _build_full_index(idx, ["doc um"])
    idx.indices["contrato"] = {
        "full": (index, metadata, embeddings),
        "chunks": (index, metadata, embeddings),
    }

    idx.unload_indices("contrato")

    assert "contrato" not in idx.indices


def test_unload_all_indices_clears_everything(make_index):
    idx = make_index(embedding_dim=4)
    index, metadata, embeddings = _build_full_index(idx, ["doc um"])
    idx.indices["contrato"] = {"full": (index, metadata, embeddings)}
    idx.indices["invoice"] = {"full": (index, metadata, embeddings)}

    idx.unload_all_indices()

    assert idx.indices == {}


def test_add_new_documents_extends_index_and_metadata(make_index):
    idx = make_index(embedding_dim=4)
    index, metadata, embeddings = _build_full_index(idx, ["doc um", "doc dois"])
    idx.indices["contrato"] = {"full": (index, metadata, embeddings)}

    idx.add_new_documents("contrato", new_docs=[("novo.txt", "conteudo do novo documento")])

    stored_index, stored_metadata, _ = idx.indices["contrato"]["full"]
    assert stored_index.ntotal == 3
    assert len(stored_metadata) == 3
    assert stored_metadata[-1]["file"] == "novo.txt"


def test_add_new_documents_skips_strategies_not_built(make_index):
    idx = make_index(embedding_dim=4)
    index, metadata, embeddings = _build_full_index(idx, ["doc um"])
    idx.indices["contrato"] = {"full": (index, metadata, embeddings)}

    # Should not raise even though "sections"/"chunks" were never built for this type.
    idx.add_new_documents("contrato", new_docs=[("novo.txt", "conteudo")])

    assert set(idx.indices["contrato"].keys()) == {"full"}


def test_add_new_documents_unknown_document_type_raises(make_index):
    idx = make_index()

    with pytest.raises(ValueError):
        idx.add_new_documents("nao_carregado", new_docs=[("a.txt", "b")])
