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
    pdf_path = tmp_path / "caso.pdf"
    pdf_path.write_text("dummy pdf")  # stand-in temp file for os.path.exists()/os.remove() to act on

    monkeypatch.setattr(utils_ocr, "convert_any_to_pdf", lambda file_path, output_dir: str(pdf_path))
    monkeypatch.setattr(utils_ocr, "process_pdf_file", lambda file_path, ocr_dpi, max_workers: "convertido")

    result = utils_ocr.extract_text_from_file_ocr_fallback(str(docx_path))

    assert result == "convertido"
    assert not pdf_path.exists()


def test_extract_text_from_file_ocr_fallback_returns_empty_string_on_error(monkeypatch):
    def boom(file_path, ocr_dpi, max_workers):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(utils_ocr, "process_pdf_file", boom)

    assert utils_ocr.extract_text_from_file_ocr_fallback("caso.pdf") == ""
