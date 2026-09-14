import threading
import time
from pathlib import Path

from faiss_index import utils_ocr


# --- merge_ocr_results ------------------------------------------------------

def test_merge_ocr_results_no_problematic_pages_passes_through():
    assert utils_ocr.merge_ocr_results(["a", "b"], [], {}) == ["a", "b"]


def test_merge_ocr_results_problematic_page_at_end():
    assert utils_ocr.merge_ocr_results(["a", "b"], [2], {2: "c"}) == ["a", "b", "c"]


def test_merge_ocr_results_problematic_page_at_start():
    assert utils_ocr.merge_ocr_results(["b"], [0], {0: "a"}) == ["a", "b"]


def test_merge_ocr_results_problematic_page_between_valid_pages_does_not_drop_content():
    # Regression test: a problematic page between two valid ones used to overwrite/drop
    # the valid page's text, because it was reinserted by original page index into a
    # list that only holds the valid pages, compacted (see extract_text_from_pdf).
    result = utils_ocr.merge_ocr_results(["t0", "t2"], [1, 3], {1: "ocr1", 3: "ocr3"})
    assert result == ["t0", "ocr1", "t2", "ocr3"]


def test_merge_ocr_results_multiple_problematic_pages_in_a_row():
    result = utils_ocr.merge_ocr_results(["t0", "t3"], [1, 2], {1: "ocr1", 2: "ocr2"})
    assert result == ["t0", "ocr1", "ocr2", "t3"]


def test_merge_ocr_results_drops_problematic_page_when_ocr_finds_nothing():
    assert utils_ocr.merge_ocr_results(["t0"], [1], {1: ""}) == ["t0"]


def test_merge_ocr_results_drops_problematic_page_with_no_ocr_entry_at_all():
    assert utils_ocr.merge_ocr_results(["t0"], [1], {}) == ["t0"]


# --- _extract_text_from_vlm_response ----------------------------------------

def test_extract_text_from_vlm_response_falls_back_to_stripped_raw_text():
    assert utils_ocr._extract_text_from_vlm_response("  hello world  ") == "hello world"


def test_extract_text_from_vlm_response_extracts_detection_tags_in_order():
    content = (
        "some preamble noise\n"
        "<|det|>text [10, 20, 100, 40]<|/det|>Primeira linha\n"
        "<|det|>text [10, 50, 100, 70]<|/det|>Segunda linha"
    )
    assert utils_ocr._extract_text_from_vlm_response(content) == "Primeira linha\nSegunda linha"


def test_extract_text_from_vlm_response_ignores_empty_detection_matches():
    content = "<|det|>text [0,0,1,1]<|/det|>   <|det|>text [1,1,2,2]<|/det|>real text"
    assert utils_ocr._extract_text_from_vlm_response(content) == "real text"


# --- extract_text_from_pdf ---------------------------------------------------

_VALID_TEXT = (
    "Página com texto suficientemente longo para ultrapassar o limite mínimo "
    "de caracteres exigido pela função de extração."
)


class _FakePage:
    def __init__(self, text=None, raise_error=False):
        self._text = text
        self._raise_error = raise_error

    def extract_text(self):
        if self._raise_error:
            raise RuntimeError("boom")
        return self._text


class _FakePdf:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_extract_text_from_pdf_classifies_problematic_pages(monkeypatch):
    pages = [
        _FakePage(_VALID_TEXT),                    # valid
        _FakePage(None),                            # no text -> problematic
        _FakePage("curto"),                          # too short -> problematic
        _FakePage("(cid:12)(cid:34) texto quebrado"),  # broken-font marker -> problematic
        _FakePage(raise_error=True),                 # extraction error -> problematic
    ]
    monkeypatch.setattr(utils_ocr.pdfplumber, "open", lambda file_path: _FakePdf(pages))

    all_text, problematic_pages = utils_ocr.extract_text_from_pdf("fake.pdf")

    assert all_text == [_VALID_TEXT]
    assert problematic_pages == [1, 2, 3, 4]


# --- process_pdf_file (orchestration) ----------------------------------------

def test_process_pdf_file_skips_ocr_when_no_problematic_pages(monkeypatch):
    monkeypatch.setattr(utils_ocr, "extract_text_from_pdf", lambda file_path: (["a", "b"], []))
    calls = []
    monkeypatch.setattr(utils_ocr, "process_problematic_pages", lambda *a, **k: calls.append((a, k)))

    result = utils_ocr.process_pdf_file("fake.pdf", ocr_dpi=300, max_workers=None)

    assert result == "a\nb"
    assert calls == []


def test_process_pdf_file_runs_ocr_fallback_for_problematic_pages(monkeypatch):
    monkeypatch.setattr(utils_ocr, "extract_text_from_pdf", lambda file_path: (["a"], [1]))
    monkeypatch.setattr(
        utils_ocr, "process_problematic_pages",
        lambda file_path, all_text, problematic_pages, ocr_dpi, max_workers: all_text + ["ocr-b"],
    )

    result = utils_ocr.process_pdf_file("fake.pdf", ocr_dpi=300, max_workers=None)

    assert result == "a\nocr-b"


# --- run_ocr_on_images --------------------------------------------------------

def test_run_ocr_on_images_runs_tasks_concurrently(monkeypatch):
    # Regression test: run_ocr_on_images used to call .submit(...).result() on each
    # image within the same comprehension step, which blocks the main thread on one
    # task's result before submitting the next — serializing everything despite the
    # ThreadPoolExecutor. A barrier only releases once ALL n workers have reached it,
    # so this proves the n tasks were actually in flight at the same time: submitted
    # one-at-a-time-and-awaited would leave the later tasks never scheduled, and the
    # barrier would time out.
    n = 4
    barrier = threading.Barrier(n, timeout=2)

    def fake_process_image(img):
        barrier.wait()
        return f"text-{img}"

    monkeypatch.setattr(utils_ocr, "process_image", fake_process_image)

    images = {i: i for i in range(n)}
    result = utils_ocr.run_ocr_on_images(images, max_workers=n)

    assert result == {i: f"text-{i}" for i in range(n)}


# --- _get_vlm_provider --------------------------------------------------------

def test_get_vlm_provider_initializes_only_once_under_concurrent_access(monkeypatch):
    # Regression test: _get_vlm_provider's lazy init used to be a plain
    # "if _vlm_provider is None: _vlm_provider = ...", with no lock — since
    # run_ocr_on_images now genuinely runs OCR concurrently (see the fix above), several
    # threads calling this at once could each see None and construct their own
    # provider (each opening its own HTTP client) before any of them assigned it back.
    # The sleep here widens that race window so the bug reproduces reliably.
    utils_ocr._vlm_provider = None
    construction_count = 0
    count_lock = threading.Lock()

    class FakeProvider:
        def __init__(self, model, vision_model):
            nonlocal construction_count
            time.sleep(0.05)
            with count_lock:
                construction_count += 1

    monkeypatch.setattr(utils_ocr, "OpenAICompatibleChatProvider", FakeProvider)
    monkeypatch.setattr(utils_ocr.config, "OCR_VLM_MODEL", "fake-vlm-model")

    threads = [threading.Thread(target=utils_ocr._get_vlm_provider) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert construction_count == 1
    assert isinstance(utils_ocr._vlm_provider, FakeProvider)

    utils_ocr._vlm_provider = None  # don't leak state into other tests


# --- process_problematic_pages (orchestration, real merge_ocr_results) -------

def test_process_problematic_pages_orchestrates_convert_ocr_and_merge(monkeypatch):
    monkeypatch.setattr(
        utils_ocr, "convert_pdf_to_images",
        lambda file_path, problematic_pages, temp_dir, ocr_dpi: {1: "fake-image"},
    )
    monkeypatch.setattr(utils_ocr, "run_ocr_on_images", lambda images, max_workers: {1: "texto via ocr"})

    result = utils_ocr.process_problematic_pages(
        "fake.pdf", all_text=["t0"], problematic_pages=[1], ocr_dpi=300, max_workers=None,
    )

    assert result == ["t0", "texto via ocr"]


# --- extract_text_from_file_ocr_fallback (orchestration) ---------------------

def test_extract_text_from_file_ocr_fallback_pdf_goes_straight_to_process_pdf_file(monkeypatch):
    monkeypatch.setattr(utils_ocr, "process_pdf_file", lambda file_path, ocr_dpi, max_workers: "conteúdo")
    assert utils_ocr.extract_text_from_file_ocr_fallback("caso.PDF") == "conteúdo"


def test_extract_text_from_file_ocr_fallback_converts_non_pdf_then_cleans_up_temp_file(monkeypatch, tmp_path):
    docx_path = tmp_path / "caso.docx"
    docx_path.write_text("dummy")
    converted_paths = []

    def fake_convert_any_to_pdf(file_path, output_dir):
        pdf_path = Path(output_dir) / "caso.pdf"
        pdf_path.write_text("dummy pdf")
        converted_paths.append(pdf_path)
        return str(pdf_path)

    monkeypatch.setattr(utils_ocr, "convert_any_to_pdf", fake_convert_any_to_pdf)
    monkeypatch.setattr(utils_ocr, "process_pdf_file", lambda file_path, ocr_dpi, max_workers: "convertido")

    result = utils_ocr.extract_text_from_file_ocr_fallback(str(docx_path))

    assert result == "convertido"
    assert converted_paths[0].parent != tmp_path  # converted outside the source's folder
    assert not converted_paths[0].exists()        # and cleaned up afterwards
    assert list(tmp_path.iterdir()) == [docx_path]


def test_extract_text_from_file_ocr_fallback_keeps_same_named_pdf_next_to_source(monkeypatch, tmp_path):
    # Regression test: the conversion used to write into the source file's own folder.
    # LibreOffice names its output <basename>.pdf (overwriting anything already there),
    # and the cleanup then deleted that path — so indexing "caso.docx" destroyed an
    # unrelated, user-owned "caso.pdf" in the same folder.
    docx_path = tmp_path / "caso.docx"
    docx_path.write_text("dummy docx")
    user_pdf = tmp_path / "caso.pdf"
    user_pdf.write_text("the user's own pdf")

    def fake_libreoffice(command, check):
        # Mirrors `libreoffice --headless --convert-to pdf <file> --outdir <dir>`.
        source, out_dir = command[4], command[6]
        (Path(out_dir) / (Path(source).stem + ".pdf")).write_text("converted from docx")

    monkeypatch.setattr(utils_ocr, "run", fake_libreoffice)
    monkeypatch.setattr(
        utils_ocr, "process_pdf_file",
        lambda file_path, ocr_dpi, max_workers: Path(file_path).read_text(),
    )

    result = utils_ocr.extract_text_from_file_ocr_fallback(str(docx_path))

    assert result == "converted from docx"
    assert user_pdf.read_text() == "the user's own pdf"


def test_extract_text_from_file_ocr_fallback_returns_empty_string_on_error(monkeypatch):
    def boom(file_path, ocr_dpi, max_workers):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(utils_ocr, "process_pdf_file", boom)

    assert utils_ocr.extract_text_from_file_ocr_fallback("caso.pdf") == ""
