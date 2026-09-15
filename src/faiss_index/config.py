"""
Environment/installation-dependent configuration for FaissDocumentIndex: .env loading,
NLTK data, and default indexing/performance/path parameter values. Unlike
`constants.py` (fixed protocol values), everything here is meant to be overridden via
environment variable or explicit parameter, without assuming any specific project's
directory structure.
"""
import os
import logging

import nltk
from dotenv import find_dotenv, load_dotenv

from .constants import STOPWORDS_LANGUAGE
from .i18n import _

logger = logging.getLogger(__name__)

# --- .env -------------------------------------------------------------------------
# FAISS_INDEX_DOTENV_PATH lets you point to a specific .env; if absent, we look for a
# .env starting from the current directory and walking up the tree, without assuming
# any specific project's directory depth. usecwd=True matters: by default find_dotenv
# starts from the directory of the module that calls it — this file, i.e. wherever the
# package is installed — so an application's own .env was never found (and running
# from a source checkout loaded this library repo's .env instead).
_dotenv_path = os.environ.get('FAISS_INDEX_DOTENV_PATH')
if _dotenv_path:
    load_dotenv(dotenv_path=_dotenv_path)
else:
    load_dotenv(find_dotenv(usecwd=True))

# --- NLTK data ------------------------------------------------------------------
# NLTK_DATA_PATH lets you point to a specific NLTK data directory. If absent, we just
# use NLTK's own default paths (e.g.: ~/nltk_data, via nltk.download()).
NLTK_DATA_PATH = os.environ.get('NLTK_DATA_PATH')
if NLTK_DATA_PATH:
    nltk.data.path.append(NLTK_DATA_PATH)

STOPWORDS_PT = nltk.corpus.stopwords.words(STOPWORDS_LANGUAGE)

logger.info(_("NLTK corpus %(corpus)s") % {"corpus": nltk.corpus.stopwords})
logger.info(_("Loaded stopwords: %(count)s words.") % {"count": len(STOPWORDS_PT)})

# --- Model/indexing/performance defaults ----------------------------------------
# Mirror the FaissDocumentIndex constructor defaults; they live here so they can be
# adjusted in one place (or overridden via environment variable) without editing the
# class signature.
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_SECTION_EXTRACTION_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_BATCH_SIZE = 100
DEFAULT_INDEX_TYPE = "auto"
DEFAULT_AUTO_INDEX_THRESHOLDS = (10_000, 80_000)
DEFAULT_IVF_NPROBE = 8
DEFAULT_PQ_M = 8
DEFAULT_PQ_NBITS = 8

# --- Path defaults ---------------------------------------------------------------
DEFAULT_BASE_DATA_DIR = os.environ.get('FAISS_INDEX_BASE_DATA_DIR', './data')
DEFAULT_OUTPUT_INDEX_DIR = os.environ.get('FAISS_INDEX_OUTPUT_INDEX_DIR', '../faiss_index')
# Used by generate_search_by_type to auto-load an index not yet in memory. Previously
# fixed at '../src/utils/mounted_volume/faiss_index' (an assumption about a specific
# project structure); now configurable via FAISS_INDEX_PATH_INDICES.
DEFAULT_PATH_INDICES = os.environ.get('FAISS_INDEX_PATH_INDICES', '../faiss_index')

# --- OCR backend ---------------------------------------------------------------------
# By default, image OCR (utils_ocr.process_image) runs locally via pytesseract. Setting
# FAISS_INDEX_OCR_VLM_MODEL switches it to a vision-capable chat model instead, called
# through the OpenAI-compatible client (OPENAI_API_KEY/OPENAI_BASE_URL env vars, read by
# the OpenAI SDK itself) — e.g. "Unlimited-OCR" served locally via LM Studio/oMLX/vLLM.
OCR_VLM_MODEL = os.environ.get('FAISS_INDEX_OCR_VLM_MODEL')
