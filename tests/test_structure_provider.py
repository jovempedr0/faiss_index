"""The structure_provider hook: sections built from each document's own structure."""


def test_build_indices_with_structure_provider_skips_calibration(make_index, fake_structure_provider, tmp_path):
    idx = make_index(embedding_dim=4, structure_provider=fake_structure_provider)
    document_type = "doctype"
    data_dir = tmp_path / "data" / document_type
    data_dir.mkdir(parents=True)
    doc_path = data_dir / "doc1.txt"
    doc_path.write_text("conteudo de exemplo", encoding="utf-8")

    fake_structure_provider.sections_by_file = {
        str(doc_path): {"introducao": "texto da introducao", "conclusao": "texto da conclusao"},
    }

    output_dir = tmp_path / "out"
    idx.build_indices(
        document_type=document_type,
        base_data_dir=str(tmp_path / "data"),
        output_index_dir=str(output_dir),
    )

    # No calibration call happened — FakeChatProvider.complete_structured would have
    # raised (no queued response) if register_document_type had been reached.
    assert document_type not in idx.section_schemas
    assert not (output_dir / document_type / f"{document_type}_section_schema.json").exists()

    assert fake_structure_provider.calls == [str(doc_path)]

    sections_index, metadata, _ = idx.indices[document_type]["sections"]
    assert sections_index.ntotal == 2
    assert {m["section_name"] for m in metadata} == {"introducao", "conclusao"}


def test_add_new_documents_with_structure_provider(make_index, fake_structure_provider, tmp_path):
    idx = make_index(embedding_dim=4, structure_provider=fake_structure_provider)
    document_type = "doctype"
    data_dir = tmp_path / "data" / document_type
    data_dir.mkdir(parents=True)
    doc_path = data_dir / "doc1.txt"
    doc_path.write_text("conteudo de exemplo", encoding="utf-8")
    fake_structure_provider.sections_by_file = {str(doc_path): {"intro": "texto"}}

    idx.build_indices(
        document_type=document_type,
        base_data_dir=str(tmp_path / "data"),
        output_index_dir=str(tmp_path / "out"),
    )

    new_path = str(tmp_path / "novo.txt")
    fake_structure_provider.sections_by_file[new_path] = {"resumo": "texto novo"}

    idx.add_new_documents(document_type, new_docs=[(new_path, "conteudo novo")])

    sections_index, metadata, _ = idx.indices[document_type]["sections"]
    assert sections_index.ntotal == 2
    assert {m["section_name"] for m in metadata} == {"intro", "resumo"}


def test_strategy_with_no_vectors_is_not_registered_as_loaded(make_index, fake_structure_provider, tmp_path):
    # Regression test: a strategy that produced no vectors was stored as
    # (None, [], None), so searching it raised AttributeError ('NoneType' object has no
    # attribute 'search'/'ntotal') and add_new_documents crashed on index.add.
    idx = make_index(embedding_dim=4, structure_provider=fake_structure_provider)  # finds no sections
    (tmp_path / "data" / "doctype").mkdir(parents=True)
    (tmp_path / "data" / "doctype" / "doc1.txt").write_text("texto sem estrutura", encoding="utf-8")

    idx.build_indices("doctype", base_data_dir=str(tmp_path / "data"), output_index_dir=str(tmp_path / "out"))

    assert set(idx.indices["doctype"]) == {"full", "chunks"}
    assert not idx.is_index_loaded(["doctype"], ["sections"], require_gpu=False)
    assert idx.evaluate_strategy("texto", "doctype", "sections") == {}
    assert idx.evaluate_strategy_hybrid("texto", "doctype", "sections") == {}
    idx.add_new_documents("doctype", [(str(tmp_path / "novo.txt"), "outro texto")])
    assert idx.indices["doctype"]["chunks"][0].ntotal == 2


def test_build_indices_without_structure_provider_still_calibrates(make_index, fake_chat_provider, tmp_path):
    idx = make_index(embedding_dim=4)  # no structure_provider: today's behavior
    document_type = "doctype"
    data_dir = tmp_path / "data" / document_type
    data_dir.mkdir(parents=True)
    (data_dir / "doc1.txt").write_text("conteudo de exemplo", encoding="utf-8")

    fake_chat_provider.responses.append({"sections": [{"name": "intro", "patterns": ["conteudo"]}]})

    idx.build_indices(
        document_type=document_type,
        base_data_dir=str(tmp_path / "data"),
        output_index_dir=str(tmp_path / "out"),
    )

    assert idx.section_schemas[document_type] == {"intro": ["conteudo"]}
