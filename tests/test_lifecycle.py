import faiss
import numpy as np
import pytest

STRATEGIES = ["full", "sections", "chunks"]


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


def test_add_new_documents_embeds_new_chunks_once_for_full_and_chunks(make_index, fake_embedding_provider):
    idx = make_index(embedding_dim=4)
    docs = [("file_0.txt", "doc um"), ("file_1.txt", "doc dois")]
    chunk_embeddings, chunk_metadata = idx.create_embeddings_chunks(docs)
    chunks_index = faiss.IndexFlatL2(4)
    chunks_index.add(chunk_embeddings.astype("float32"))
    idx.indices["contrato"] = {
        "full": _build_full_index(idx, ["doc um", "doc dois"]),
        "chunks": (chunks_index, chunk_metadata, chunk_embeddings),
    }
    fake_embedding_provider.embedding_calls.clear()

    idx.add_new_documents("contrato", new_docs=[("novo.txt", "conteudo do novo documento")])

    assert fake_embedding_provider.embedding_calls == [1]  # the new doc's single chunk, once
    assert idx.indices["contrato"]["full"][0].ntotal == 3
    assert idx.indices["contrato"]["chunks"][0].ntotal == 3


def test_add_new_documents_skips_strategies_not_built(make_index):
    idx = make_index(embedding_dim=4)
    index, metadata, embeddings = _build_full_index(idx, ["doc um"])
    idx.indices["contrato"] = {"full": (index, metadata, embeddings)}

    # Should not raise even though "sections"/"chunks" were never built for this type.
    idx.add_new_documents("contrato", new_docs=[("novo.txt", "conteudo")])

    assert set(idx.indices["contrato"].keys()) == {"full"}


def _build_and_load(make_index, fake_chat_provider, tmp_path):
    (tmp_path / "data" / "contrato").mkdir(parents=True)
    (tmp_path / "data" / "contrato" / "a.txt").write_text("Cabecalho\ncorpo sobre contratos", encoding="utf-8")
    fake_chat_provider.responses.append({"sections": [{"name": "corpo", "patterns": ["corpo"]}]})
    make_index(embedding_dim=4).build_indices("contrato", base_data_dir=str(tmp_path / "data"), output_index_dir=str(tmp_path / "out"))
    idx = make_index(embedding_dim=4)
    idx.load_indices(str(tmp_path / "out"), ["contrato"], STRATEGIES, use_gpu=False)
    return idx


def test_save_indices_persists_added_documents_over_the_directory_they_were_loaded_from(
    make_index, fake_chat_provider, tmp_path
):
    # add_new_documents only ever changed memory — nothing could write the grown index
    # back short of rebuilding everything. Saved here over the very directory the
    # embeddings are memory-mapped from.
    idx = _build_and_load(make_index, fake_chat_provider, tmp_path)
    original_rows = {s: np.array(idx.indices["contrato"][s][2]) for s in STRATEGIES}
    idx.add_new_documents("contrato", [("novo.txt", "Cabecalho\ncorpo sobre pagamentos")])

    idx.save_indices("contrato", str(tmp_path / "out"))

    assert idx._unsaved_embeddings.get("contrato", {}) == {}
    reloaded = make_index(embedding_dim=4)
    reloaded.load_indices(str(tmp_path / "out"), ["contrato"], STRATEGIES, use_gpu=False)
    for strategy in STRATEGIES:
        index, metadata, embeddings = reloaded.indices["contrato"][strategy]
        assert index.ntotal == len(metadata) == len(embeddings) > len(original_rows[strategy])
        assert metadata[-1]["file"] == "novo.txt"
        assert np.array_equal(embeddings[: len(original_rows[strategy])], original_rows[strategy])
    new_chunk_row = reloaded.indices["contrato"]["chunks"][2][-1]
    assert np.array_equal(new_chunk_row, reloaded.get_embeddings(["Cabecalho corpo sobre pagamentos"])[0])


def test_reloading_a_strategy_discards_its_unsaved_rows(make_index, fake_chat_provider, tmp_path):
    idx = _build_and_load(make_index, fake_chat_provider, tmp_path)
    idx.add_new_documents("contrato", [("novo.txt", "Cabecalho\ncorpo sobre pagamentos")])

    idx.load_indices(str(tmp_path / "out"), ["contrato"], ["chunks"], use_gpu=False)  # back to what's on disk
    idx.unload_indices("contrato", "sections")
    idx.save_indices("contrato", str(tmp_path / "out"))  # "full" keeps its added doc; "chunks" doesn't

    reloaded = make_index(embedding_dim=4)
    reloaded.load_indices(str(tmp_path / "out"), ["contrato"], ["full", "chunks"], use_gpu=False)
    assert [m["file"] for m in reloaded.indices["contrato"]["full"][1]][-1] == "novo.txt"
    chunk_index, chunk_metadata, chunk_rows = reloaded.indices["contrato"]["chunks"]
    assert chunk_index.ntotal == len(chunk_metadata) == len(chunk_rows) == 1


def test_save_indices_refuses_a_strategy_whose_rows_dont_line_up(make_index, tmp_path):
    idx = make_index(embedding_dim=4)
    index, metadata, embeddings = _build_full_index(idx, ["doc um", "doc dois"])
    idx.indices["contrato"] = {"full": (index, metadata, embeddings[:1])}

    with pytest.raises(ValueError, match="2 vectors"):
        idx.save_indices("contrato", str(tmp_path / "out"))
    assert not (tmp_path / "out" / "contrato" / "contrato_full.index").exists()

    with pytest.raises(ValueError):
        idx.save_indices("nao_carregado", str(tmp_path / "out"))


def test_load_indices_that_finds_nothing_leaves_the_document_type_unloaded(make_index, fake_chat_provider, tmp_path):
    # Regression test: load_indices left self.indices[document_type] = {} when the
    # directory (or every requested strategy's files) was missing, and add_new_documents/
    # save_indices then returned without error, having done nothing.
    idx = make_index(embedding_dim=4)
    idx.load_indices(str(tmp_path / "nao_existe"), ["contrato"], STRATEGIES, use_gpu=False)
    assert "contrato" not in idx.indices
    with pytest.raises(ValueError, match="contrato"):
        idx.add_new_documents("contrato", [("novo.txt", "conteudo")])
    with pytest.raises(ValueError, match="contrato"):
        idx.save_indices("contrato", str(tmp_path / "out"))

    loaded = _build_and_load(make_index, fake_chat_provider, tmp_path)
    idx.load_indices(str(tmp_path / "out"), ["contrato"], ["nao_existe"], use_gpu=False)  # dir exists, strategy doesn't
    assert "contrato" not in idx.indices
    loaded.load_indices(str(tmp_path / "out"), ["contrato"], ["nao_existe"], use_gpu=False)
    assert set(loaded.indices["contrato"]) == set(STRATEGIES)  # what was already loaded stays


def test_add_new_documents_unknown_document_type_raises(make_index):
    idx = make_index()

    with pytest.raises(ValueError):
        idx.add_new_documents("nao_carregado", new_docs=[("a.txt", "b")])
