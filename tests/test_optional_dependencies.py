import subprocess
import sys
import textwrap
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _TESTS_DIR.parent / "src"


def test_package_works_for_txt_files_without_the_ocr_extra(tmp_path):
    # Regression test: _ingestion.py imported utils_ocr at module level, and utils_ocr
    # imports pdfplumber/pytesseract/pdf2image/PIL — the optional [ocr] extra — so
    # `import faiss_index` itself failed without it, even though .txt files never
    # touch any of those. Runs in a subprocess so the modules can be made unimportable
    # without affecting the rest of the suite.
    (tmp_path / "data" / "doctype").mkdir(parents=True)
    (tmp_path / "data" / "doctype" / "doc.txt").write_text("contrato de locacao comercial", encoding="utf-8")
    script = textwrap.dedent(f"""
        import sys
        for name in ("pdfplumber", "pytesseract", "pdf2image", "PIL"):
            sys.modules[name] = None  # any import of these now raises ImportError
        sys.path[:0] = [{str(_SRC_DIR)!r}, {str(_TESTS_DIR)!r}]

        from faiss_index import FaissDocumentIndex
        from conftest import FakeChatProvider, FakeEmbeddingProvider

        chat = FakeChatProvider()
        chat.responses.append({{"sections": []}})
        idx = FaissDocumentIndex(base_path=".", embedding_provider=FakeEmbeddingProvider(),
                                 chat_provider=chat, use_mps=False)
        idx.build_indices("doctype", base_data_dir={str(tmp_path / "data")!r},
                          output_index_dir={str(tmp_path / "out")!r})
        print("chunks:", idx.indices["doctype"]["chunks"][0].ntotal)

        try:
            idx.read_document("scan.pdf")
        except ImportError as e:
            print("pdf:", e)
    """)

    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert "chunks: 1" in result.stdout
    assert 'pip install -e ".[ocr]"' in result.stdout
