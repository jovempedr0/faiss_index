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
from dotenv import load_dotenv

from .constants import STOPWORDS_LANGUAGE
from .i18n import _

logger = logging.getLogger(__name__)

# --- .env -------------------------------------------------------------------------
# Only when asked for: set FAISS_INDEX_DOTENV_PATH to the .env this library should
# load. Importing a library shouldn't put variables into the process — this used to
# search for a .env from the working directory upwards and load it, which fills in
# whatever that file holds (not just this library's settings) for every other library
# in the process too, from a file nobody here chose. An application that wants that
# calls load_dotenv() itself, before importing this package.
_dotenv_path = os.environ.get('FAISS_INDEX_DOTENV_PATH')
if _dotenv_path:
    load_dotenv(dotenv_path=_dotenv_path)

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
# Exact search up to 100k vectors: it costs 3.80 ms/query there and the same memory an
# ivf_flat would take (391 MiB), so the old 10k limit traded ~5 points of recall for
# ~3 ms. ivf_sq8 starts at 250k, where its quarter-size codes (489 MiB against 1957 at
# 500k) start to matter. See fixtures/eval/benchmark_index_defaults.py.
DEFAULT_AUTO_INDEX_THRESHOLDS = (100_000, 250_000)
# 8 kept only 92.5-94.3% of the exact top-10 at 100k vectors once the benchmark corpus
# was given realistic (less tightly clustered) geometry; 32 kept >=98.7% across every
# geometry tested, for 1.88 ms/query instead of 0.52 at 500k — against 18.95 ms for exact
# search. See fixtures/eval/benchmark_index_defaults.py.
DEFAULT_IVF_NPROBE = 32
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
