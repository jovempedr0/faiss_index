from pathlib import Path

import numpy as np

def test_build_load_and_search_roundtrip(make_index, fake_chat_provider, tmp_path):
    document_type = "doctype"
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "out"
    (data_dir / document_type).mkdir(parents=True)

    (data_dir / document_type / "doc1.txt").write_text(
        "Cabecalho inicial\nCorpo do documento fala sobre contratos.\n", encoding="utf-8"
    )
    (data_dir / document_type / "doc2.txt").write_text(
        "Cabecalho inicial\nCorpo do documento fala sobre pagamentos.\n", encoding="utf-8"
    )

    idx = make_index(embedding_dim=8)
    fake_chat_provider.responses.append(
        {"sections": [{"name": "corpo", "patterns": ["corpo do documento"]}]}
    )

    idx.build_indices(
        document_type=document_type,
        base_data_dir=str(data_dir),
        output_index_dir=str(output_dir),
    )

    strategy_dir = output_dir / document_type
    expected_files = [
        f"{document_type}_section_schema.json",
        f"{document_type}_full.index",
        f"{document_type}_full_metadata.json",
        f"{document_type}_full_embeddings.npy",
        f"{document_type}_sections.index",
        f"{document_type}_sections_metadata.json",
        f"{document_type}_sections_embeddings.npy",
        f"{document_type}_chunks.index",
        f"{document_type}_chunks_metadata.json",
        f"{document_type}_chunks_embeddings.npy",
    ]
    for filename in expected_files:
        assert (strategy_dir / filename).exists(), f"missing {filename}"

    # Fresh instance, simulating a new process loading the persisted index.
    idx2 = make_index(embedding_dim=8)
    idx2.load_indices(
        path_indices=str(output_dir),
        document_types=[document_type],
        strategies=["full", "sections", "chunks"],
        use_gpu=False,
    )

    assert idx2.is_index_loaded(
        document_types=[document_type], strategies=["full", "sections", "chunks"], require_gpu=False
    )

    results = idx2.evaluate_strategy(
        query="contratos e pagamentos", document_type=document_type, strategy="full", k=5
    )
    assert results["strategy"] == "full"
    assert len(results["results"]) == 2
    assert {r["metadata"]["file"] for r in results["results"]} == {
        str(data_dir / document_type / "doc1.txt"),
        str(data_dir / document_type / "doc2.txt"),
    }


def test_build_indices_embeds_chunks_once_and_pools_full_from_them(
    make_index, fake_embedding_provider, fake_chat_provider, tmp_path
):
    data_dir = tmp_path / "data"
    (data_dir / "doctype").mkdir(parents=True)
    (data_dir / "doctype" / "doc1.txt").write_text(" ".join(f"a{i}" for i in range(1000)), encoding="utf-8")
    (data_dir / "doctype" / "doc2.txt").write_text("documento curto", encoding="utf-8")

    idx = make_index(embedding_dim=8)
    fake_chat_provider.responses.append({"sections": []})  # no structure -> skip "sections"
    idx.build_indices(document_type="doctype", base_data_dir=str(data_dir), output_index_dir=str(tmp_path / "out"))

    _, chunk_metadata, _ = idx.indices["doctype"]["chunks"]
    full_index, _, _ = idx.indices["doctype"]["full"]
    assert len(chunk_metadata) == 4  # 3 + 1
    assert sum(fake_embedding_provider.embedding_calls) == 4  # no separate embedding calls for "full"
    assert full_index.ntotal == 2


def _build_txt_corpus(idx, fake_chat_provider, tmp_path, texts):
    data_dir = tmp_path / "data"
    (data_dir / "doctype").mkdir(parents=True, exist_ok=True)
    for i, text in enumerate(texts):
        (data_dir / "doctype" / f"doc{i}.txt").write_text(text, encoding="utf-8")
    fake_chat_provider.responses.append({"sections": [{"name": "corpo", "patterns": ["corpo"]}]})
    idx.build_indices(document_type="doctype", base_data_dir=str(data_dir), output_index_dir=str(tmp_path / "out"))
    return tmp_path / "out"


def test_embedding_prefixes_apply_to_indexed_texts_and_queries(
    make_index, fake_embedding_provider, fake_chat_provider, tmp_path, monkeypatch
):
    embedded = []
    real_embed = fake_embedding_provider.embed
    monkeypatch.setattr(fake_embedding_provider, "embed", lambda texts: embedded.extend(texts) or real_embed(texts))
    idx = make_index(embedding_dim=8, embedding_query_prefix="Query: ", embedding_document_prefix="Document: ")

    _build_txt_corpus(idx, fake_chat_provider, tmp_path, ["Cabecalho\ncorpo sobre contratos", "Cabecalho\ncorpo sobre pagamentos"])
    indexed = list(embedded)
    embedded.clear()
    idx.generate_search_by_type("Contratos sem multa?", "doctype", "chunks", require_gpu=False, k=1, use_hybrid=True)
    idx.evaluate_strategy("pagamentos", "doctype", "sections", k=1)

    assert indexed and all(text.startswith("Document: ") for text in indexed)
    assert embedded == ["Query: Contratos sem multa?", "Query: pagamentos"]


def test_load_indices_warns_when_saved_document_prefix_differs(make_index, fake_chat_provider, tmp_path, caplog):
    out = _build_txt_corpus(
        make_index(embedding_dim=8, embedding_document_prefix="passage: "), fake_chat_provider, tmp_path, ["corpo do documento"]
    )
    assert (out / "doctype" / "doctype_chunks_embedding.json").exists()

    with caplog.at_level("WARNING", logger="faiss_index._lifecycle"):
        make_index(embedding_dim=8, embedding_document_prefix="passage: ").load_indices(str(out), ["doctype"], ["chunks"], use_gpu=False)
    assert "embedding_document_prefix" not in caplog.text

    with caplog.at_level("WARNING", logger="faiss_index._lifecycle"):
        make_index(embedding_dim=8).load_indices(str(out), ["doctype"], ["chunks"], use_gpu=False)
    assert "embedding_document_prefix='passage: '" in caplog.text

    # An index saved before prefixes existed (no embedding.json) was built without one.
    (out / "doctype" / "doctype_chunks_embedding.json").unlink()
    caplog.clear()
    with caplog.at_level("WARNING", logger="faiss_index._lifecycle"):
        make_index(embedding_dim=8, embedding_document_prefix="Document: ").load_indices(str(out), ["doctype"], ["chunks"], use_gpu=False)
    assert "embedding_document_prefix=''" in caplog.text


def test_section_metadata_does_not_repeat_the_whole_document(make_index, fake_chat_provider, tmp_path):
    # Regression test: every section entry stored the entire document under "content",
    # so the sections metadata grew with (sections per document) x (document size) —
    # 12.3 of 14.7 MB on a real 23-document corpus.
    idx = make_index(embedding_dim=8)
    _build_txt_corpus(idx, fake_chat_provider, tmp_path, ["Cabecalho\ncorpo do documento " + "texto " * 200])

    _, section_metadata, _ = idx.indices["doctype"]["sections"]
    _, full_metadata, _ = idx.indices["doctype"]["full"]

    assert {m["section_name"] for m in section_metadata} == {"cabecalho", "corpo"}
    assert all("content" not in m for m in section_metadata)
    assert full_metadata[0]["file"] == section_metadata[0]["file"]  # whole text still reachable via "full"


def test_embeddings_are_float32_in_memory_and_on_disk(make_index, fake_chat_provider, tmp_path):
    # Regression test: get_embeddings built a float64 array (and so did the pooled "full"
    # vectors), doubling build-time memory and the saved .npy size for no gain — FAISS
    # indexes float32 either way.
    idx = make_index(embedding_dim=8)
    out = _build_txt_corpus(idx, fake_chat_provider, tmp_path, ["Cabecalho\ncorpo sobre contratos", "Cabecalho\ncorpo sobre pagamentos"])

    assert idx.get_embeddings(["um texto"]).dtype == np.float32
    for strategy in ("full", "sections", "chunks"):
        assert idx.indices["doctype"][strategy][2].dtype == np.float32
        assert np.load(out / "doctype" / f"doctype_{strategy}_embeddings.npy").dtype == np.float32


def test_rebuild_removes_the_files_of_a_strategy_it_no_longer_produces(
    make_index, fake_chat_provider, fake_structure_provider, tmp_path
):
    # Regression test: rebuilding into the same directory left a strategy's files from
    # the previous build on disk when this build didn't produce it, and load_indices
    # then served sections of a document no longer in the corpus.
    out = _build_txt_corpus(make_index(embedding_dim=8), fake_chat_provider, tmp_path, ["Cabecalho\ncorpo antigo"])
    (tmp_path / "data" / "doctype" / "doc0.txt").unlink()
    (tmp_path / "data" / "doctype" / "novo.txt").write_text("documento novo", encoding="utf-8")
    (out / "doctype" / "doctype_notes.txt").write_text("not the library's", encoding="utf-8")

    rebuilder = make_index(embedding_dim=8, structure_provider=fake_structure_provider)  # finds no sections
    rebuilder.build_indices(document_type="doctype", base_data_dir=str(tmp_path / "data"), output_index_dir=str(out))

    assert sorted(p.name for p in (out / "doctype").iterdir()) == sorted(
        [f"doctype_{strategy}{suffix}" for strategy in ("full", "chunks")
         for suffix in (".index", "_metadata.json", "_embeddings.npy", "_embedding.json")]
        + ["doctype_section_schema.json", "doctype_notes.txt"]
    )
    loaded = make_index(embedding_dim=8)
    loaded.load_indices(str(out), ["doctype"], ["full", "sections", "chunks"], use_gpu=False)
    assert set(loaded.indices["doctype"]) == {"full", "chunks"}
    assert [Path(m["file"]).name for m in loaded.indices["doctype"]["full"][1]] == ["novo.txt"]


def test_build_indices_picks_up_upper_case_extensions(make_index, fake_chat_provider, tmp_path):
    # Regression test: build_indices filtered files with a case-sensitive
    # `p.suffix in SUPPORTED_FILE_EXTENSIONS`, so "DOC2.TXT" (or a scanned "X.PDF")
    # was silently left out of the index, even though read_document handles it fine.
    data_dir = tmp_path / "data"
    (data_dir / "doctype").mkdir(parents=True)
    (data_dir / "doctype" / "doc1.txt").write_text("primeiro documento", encoding="utf-8")
    (data_dir / "doctype" / "DOC2.TXT").write_text("segundo documento", encoding="utf-8")
    (data_dir / "doctype" / "notes.md").write_text("unsupported extension", encoding="utf-8")

    idx = make_index(embedding_dim=8)
    fake_chat_provider.responses.append({"sections": []})
    idx.build_indices(document_type="doctype", base_data_dir=str(data_dir), output_index_dir=str(tmp_path / "out"))

    _, metadata, _ = idx.indices["doctype"]["full"]
    assert {Path(m["file"]).name for m in metadata} == {"doc1.txt", "DOC2.TXT"}


def test_build_indices_uses_text_extractor_instead_of_reading_the_files(make_index, fake_chat_provider, tmp_path):
    # The by-pass for documents this library can't extract on its own (scanned PDFs with
    # no usable OCR here, formats handled by another pipeline): the files on disk are
    # never parsed, only their paths are used, and the text comes from the extractor.
    data_dir = tmp_path / "data"
    (data_dir / "doctype").mkdir(parents=True)
    # Deliberately not a real PDF: parsing it would raise, so a passing build proves
    # nothing here opened it.
    (data_dir / "doctype" / "scan1.pdf").write_bytes(b"not a pdf at all")
    (data_dir / "doctype" / "scan2.pdf").write_bytes(b"not a pdf either")

    extracted = {
        "scan1.pdf": "Texto do primeiro processo, extraido por fora sobre contratos.",
        "scan2.pdf": "Texto do segundo processo, extraido por fora sobre pagamentos.",
    }
    seen = []

    def extractor(file_path):
        seen.append(file_path)
        return extracted[Path(file_path).name]

    idx = make_index(embedding_dim=8, text_extractor=extractor)
    fake_chat_provider.responses.append({"sections": []})
    idx.build_indices(document_type="doctype", base_data_dir=str(data_dir), output_index_dir=str(tmp_path / "out"))

    assert {Path(p).name for p in seen} == {"scan1.pdf", "scan2.pdf"}
    _, metadata, _ = idx.indices["doctype"]["full"]
    # Real paths, so the index still points back at the files on disk.
    assert {m["file"] for m in metadata} == {str(data_dir / "doctype" / name) for name in extracted}
    # "full" keeps each document's own text in a 1-element list.
    assert {m["content"][0] for m in metadata} == set(extracted.values())


def test_build_indices_skips_a_file_whose_text_extractor_raised(make_index, fake_chat_provider, tmp_path):
    # An extractor that fails for one document must not take the whole build down: the
    # file is reported as unreadable (like any other read error) and the rest is indexed.
    data_dir = tmp_path / "data"
    (data_dir / "doctype").mkdir(parents=True)
    (data_dir / "doctype" / "ok.pdf").write_bytes(b"irrelevant")
    (data_dir / "doctype" / "broken.pdf").write_bytes(b"irrelevant")

    def extractor(file_path):
        if Path(file_path).name == "broken.pdf":
            raise RuntimeError("upstream OCR service is down")
        return "documento legivel sobre contratos"

    idx = make_index(embedding_dim=8, text_extractor=extractor)
    fake_chat_provider.responses.append({"sections": []})
    idx.build_indices(document_type="doctype", base_data_dir=str(data_dir), output_index_dir=str(tmp_path / "out"))

    _, metadata, _ = idx.indices["doctype"]["full"]
    assert [Path(m["file"]).name for m in metadata] == ["ok.pdf"]


def test_read_document_with_a_text_extractor_skips_the_extension_check(make_index):
    # Direct calls aren't limited to SUPPORTED_FILE_EXTENSIONS anymore: what this library
    # can parse says nothing about what someone else's extractor can.
    idx = make_index(embedding_dim=8, text_extractor=lambda path: f"conteudo de {Path(path).name}")

    assert idx.read_document("/algum/lugar/planilha.xlsx") == "conteudo de planilha.xlsx"
