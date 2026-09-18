# Installation and setup

## Installation

Required dependencies:

```bash
pip install numpy faiss-cpu "openai>=1.40" nltk python-dotenv scikit-learn psutil rank-bm25
```

That's enough to import the package and index `.txt` files; reading `.pdf`/`.doc`/`.docx`
needs the `ocr` extra (see [Required setup](#required-setup)).

Alternatively, from this repository (editable install, with the OCR extra):

```bash
pip install -e ".[ocr]"
```

`faiss-cpu` only exists for CPU (FAISS has no Metal/MPS backend); on Linux with CUDA,
`faiss-gpu` enables `load_indices(..., use_gpu=True)`.

## Required setup

1. **NLTK stopwords** (used by `clean_text`/`remove_stop_words`):

   ```python
   import nltk
   nltk.download("stopwords")
   ```

2. **OpenAI key** (only for the default provider — see
   [Plugging in a custom provider](providers.md#plugging-in-a-custom-provider) to use a different
   backend instead): via the `OPENAI_API_KEY` environment variable, or passed directly
   to the constructor (`openai_key=...`). A `.env` is only read when the application
   loads it, or when `FAISS_INDEX_DOTENV_PATH` names it — see
   [Before using it: configuration](#before-using-it-configuration).

3. **Reading PDF/DOC/DOCX**: depends on `extract_text_from_file_ocr_fallback`
   (`utils_ocr.py`), which in turn needs `pdfplumber`, `pytesseract`, `pdf2image`,
   `Pillow`, and the `libreoffice` binary on the PATH (to convert `.doc`/`.docx`
   before OCR) — install the Python side with `pip install -e ".[ocr]"`. `.txt` files
   are read directly and don't need any of this: `utils_ocr` is only imported when a
   `.pdf`/`.doc`/`.docx` is actually read (raising an `ImportError` pointing at the
   extra if it's missing).
   `pytesseract` can be swapped for a vision-capable chat model via
   `FAISS_INDEX_OCR_VLM_MODEL` — see
   [The `config.py` and `constants.py` modules](configuration.md#the-configpy-and-constantspy-modules).
   None of this is needed if your own pipeline already extracts the text: see
   [Indexing text extracted elsewhere](indexing.md#indexing-text-extracted-elsewhere).

   Tesseract also needs the **`por` language data** (`por.traineddata`), which
   `apt install tesseract-ocr` does *not* include — install `tesseract-ocr-por`
   (Debian/Ubuntu) or `tesseract-lang` (Homebrew), or point `TESSDATA_PREFIX` at a
   tessdata directory that has it. Without it, a document with scanned pages raises
   `utils_ocr.OCRUnavailableError` instead of quietly losing those pages: OCR that
   can't run at all is a setup problem, and swallowing it means indexing documents
   that look complete but aren't. `FAISS_INDEX_OCR_VLM_MODEL` and `text_extractor`
   both bypass this check entirely — neither uses tesseract.

### Before using it: configuration

Before any of the uses below:

1. Install the dependencies ([Installation](#installation)) and run
   `nltk.download("stopwords")` ([Required setup](#required-setup)).
2. Make sure `OPENAI_API_KEY` is set — in the environment, or `openai_key=...` on the
   constructor. Importing this package does **not** load a `.env` on its own: call
   `load_dotenv()` in your application before importing it, or point
   `FAISS_INDEX_DOTENV_PATH` at the file (see below).
3. If your project's directory structure isn't the default (`./data`,
   `../faiss_index`, etc.), or you need a non-default `.env`/NLTK data location,
   adjust the environment variables described in
   [The `config.py` and `constants.py` modules](configuration.md#the-configpy-and-constantspy-modules)
   (`FAISS_INDEX_DOTENV_PATH`, `NLTK_DATA_PATH`, `FAISS_INDEX_BASE_DATA_DIR`,
   `FAISS_INDEX_OUTPUT_INDEX_DIR`, `FAISS_INDEX_PATH_INDICES`) — and, to force the
   log language, `FAISS_INDEX_LANG` ([Log language](configuration.md#log-language)).

   **Important:** `config.py` reads these environment variables exactly once, when
   the module is imported (`import faiss_index` already triggers `import config`).
   Set them *before* the import — via `export` in the shell, a `.env` loaded
   earlier, or `os.environ[...] = ...` as the first lines of your script — never
   after:

   ```python
   import os
   os.environ["FAISS_INDEX_OUTPUT_INDEX_DIR"] = "/data/production/faiss_index"

   from faiss_index import FaissDocumentIndex  # only now does config.py read the variable above
   ```
