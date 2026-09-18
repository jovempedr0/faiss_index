import pytest

from faiss_index.providers import DoclingStructureProvider, _looks_corrupted


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


@pytest.mark.parametrize("text,expected", [
    # Real sample of Docling's glyph-id fallback on a broken-font PDF (see providers.py).
    ("0 1 2 3\n\n3 4 5\n\ni255\n\n7 8 9 0 4\n\n10 11\n\n124 13 14\n\n9i255", True),
    ("Bruna Epitacio Alkimim Santos\n\nDe:\n\nCJ MANDADOS\n\nterça-feira, 1 de outubro de 2024", False),
    ("", True),
    ("   ", True),
    # Numeric-heavy but legitimate (protocol/CPF numbers alongside real words).
    ("Protocolo 133782476332914286 CPF 000.000.000-00 1069338-24.2024.4.01.3400", False),
])
def test_looks_corrupted(text, expected):
    assert _looks_corrupted(text) is expected


def test_docling_structure_provider_falls_back_to_ocr_on_corrupted_text(monkeypatch):
    # Bypasses __init__ (which imports/configures the real Docling converter) so this
    # test needs neither the 'docling' extra nor a real PDF — only extract_sections's
    # own corruption-detection/fallback logic is under test here.
    provider = DoclingStructureProvider.__new__(DoclingStructureProvider)

    class FakeDoc:
        def export_to_text(self):
            return "0 1 2 3\n\n3 4 5\n\ni255"

        def iterate_items(self):
            return []

    class FakeConverter:
        def convert(self, file_path):
            return type("Result", (), {"document": FakeDoc()})()

    provider._converter = FakeConverter()

    monkeypatch.setattr(
        "faiss_index.utils_ocr.extract_text_from_file_ocr_fallback",
        lambda file_path: "texto recuperado via OCR",
    )

    assert provider.extract_sections("qualquer.pdf") == {"completo": "texto recuperado via OCR"}


def test_docling_structure_provider_returns_empty_when_ocr_fallback_also_fails(monkeypatch):
    provider = DoclingStructureProvider.__new__(DoclingStructureProvider)

    class FakeDoc:
        def export_to_text(self):
            return ""

        def iterate_items(self):
            return []

    class FakeConverter:
        def convert(self, file_path):
            return type("Result", (), {"document": FakeDoc()})()

    provider._converter = FakeConverter()

    monkeypatch.setattr(
        "faiss_index.utils_ocr.extract_text_from_file_ocr_fallback",
        lambda file_path: "",
    )

    assert provider.extract_sections("qualquer.pdf") == {}


def test_docling_structure_provider_splits_sections_by_heading():
    # Covers extract_sections's main path (real, non-corrupted text): items are
    # grouped under whichever heading-labeled item ("section_header"/"title") most
    # recently preceded them, with everything before the first heading going to
    # "cabecalho" and blank/whitespace-only items skipped entirely.
    provider = DoclingStructureProvider.__new__(DoclingStructureProvider)

    class FakeItem:
        def __init__(self, text, label):
            self.text = text
            self.label = label

    items = [
        (FakeItem("Texto introdutório antes de qualquer título.", "text"), 0),
        (FakeItem("Cláusula 1 - Objeto", "section_header"), 1),
        (FakeItem("Descrição do objeto do contrato.", "text"), 2),
        (FakeItem("   ", "text"), 2),
        (FakeItem("Cláusula 2 - Pagamento", "section_header"), 1),
        (FakeItem("Condições de pagamento aqui descritas.", "text"), 2),
    ]
    full_text = "\n".join(item.text for item, _level in items)

    class FakeDoc:
        def export_to_text(self):
            return full_text

        def iterate_items(self):
            return items

    class FakeConverter:
        def convert(self, file_path):
            return type("Result", (), {"document": FakeDoc()})()

    provider._converter = FakeConverter()

    sections = provider.extract_sections("qualquer.pdf")

    assert sections == {
        "completo": full_text,
        "cabecalho": "Texto introdutório antes de qualquer título.",
        "cláusula_1_-_objeto": "Descrição do objeto do contrato.",
        "cláusula_2_-_pagamento": "Condições de pagamento aqui descritas.",
    }
