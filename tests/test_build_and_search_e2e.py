from pathlib import Path


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
