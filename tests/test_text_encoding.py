"""How read_document decides a .txt file's encoding (see _ingestion._decode_text_file)."""
import codecs

import pytest

TEXT = "Justiça Federal da 1ª Região — ação nº 1077/2025"


def _write(tmp_path, name, raw: bytes):
    path = tmp_path / name
    path.write_bytes(raw)
    return str(path)


def test_reads_utf8(make_index, tmp_path):
    idx = make_index()
    assert idx.read_document(_write(tmp_path, "doc.txt", TEXT.encode("utf-8"))) == TEXT


def test_strips_the_utf8_byte_order_mark(make_index, tmp_path):
    # A BOM from a Windows editor used to survive into the text, so the document's first
    # characters — what "full"/"sections" embed and match patterns against — began with
    # an invisible ﻿.
    idx = make_index()
    content = idx.read_document(_write(tmp_path, "doc.txt", TEXT.encode("utf-8-sig")))

    assert content == TEXT
    assert not content.startswith("﻿")


@pytest.mark.parametrize("encoding", ["utf-16", "utf-32"])
def test_reads_what_a_byte_order_mark_declares(make_index, tmp_path, encoding):
    # The mark is the file's own statement of its encoding: UTF-16/32 aren't valid UTF-8
    # either, and decoding them with the single-byte fallback would produce mojibake
    # instead of text.
    idx = make_index()
    assert idx.read_document(_write(tmp_path, "doc.txt", TEXT.encode(encoding))) == TEXT


@pytest.mark.parametrize("bom, encoding", [
    (codecs.BOM_UTF16_BE, "utf-16-be"),
    (codecs.BOM_UTF32_BE, "utf-32-be"),
])
def test_reads_big_endian_files_too(make_index, tmp_path, bom, encoding):
    # Python's "utf-16"/"utf-32" codecs write the mark for the machine's own byte order
    # (little-endian here), so a big-endian file has to be assembled by hand — and a
    # UTF-32-BE mark must not be read as the UTF-16-BE one it starts with.
    idx = make_index()
    assert idx.read_document(_write(tmp_path, "doc.txt", bom + TEXT.encode(encoding))) == TEXT


def test_falls_back_to_cp1252_and_says_so(make_index, tmp_path, caplog):
    # Regression test: a .txt exported by Windows/legacy systems raised UnicodeDecodeError,
    # so build_indices logged it as unreadable and left the document out of the index.
    idx = make_index()
    path = _write(tmp_path, "legado.txt", TEXT.encode("cp1252"))

    with caplog.at_level("WARNING"):
        content = idx.read_document(path)

    assert content == TEXT
    assert "legado.txt" in caplog.text and "cp1252" in caplog.text


def test_raises_when_no_encoding_can_decode_it(make_index, tmp_path):
    # 0x81 is undefined in cp1252, so the fallback can't quietly turn arbitrary bytes
    # into text: the file is reported instead.
    idx = make_index()
    path = _write(tmp_path, "binario.txt", b"\xff\x81\xfe texto \x90")

    with pytest.raises(UnicodeDecodeError):
        idx.read_document(path)


def test_build_indices_now_includes_a_cp1252_document(make_index, fake_chat_provider, tmp_path):
    data_dir = tmp_path / "data" / "doctype"
    data_dir.mkdir(parents=True)
    (data_dir / "utf8.txt").write_bytes("Documento em utf-8 sobre contratos".encode("utf-8"))
    (data_dir / "legado.txt").write_bytes("Documento legado sobre pagamentos à vista".encode("cp1252"))

    idx = make_index()
    fake_chat_provider.responses.append({"sections": []})
    idx.build_indices(document_type="doctype", base_data_dir=str(tmp_path / "data"), output_index_dir=str(tmp_path / "out"))

    _, metadata, _ = idx.indices["doctype"]["full"]
    assert {m["file"].rsplit("/", 1)[-1] for m in metadata} == {"utf8.txt", "legado.txt"}
    assert any("à vista" in m["content"][0] for m in metadata)
