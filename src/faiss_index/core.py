import logging
import os
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from . import config
from . import constants
# FAISS_HAS_GPU_SUPPORT isn't used in this file anymore (only by _lifecycle.py) but is
# re-exported here since it used to live in this module.
from ._gpu_support import FAISS_HAS_GPU_SUPPORT  # noqa: F401
from ._index_backend import FaissIndexBackendMixin
from ._ingestion import DocumentIngestionMixin
from ._lifecycle import IndexLifecycleMixin
from ._search import SearchMixin
from ._sections import SectionSchemaMixin
from .i18n import _
from .providers import EmbeddingProvider, ChatProvider, RerankProvider, StructureProvider, OpenAICompatibleEmbeddingProvider, OpenAICompatibleChatProvider

logger = logging.getLogger(__name__)


class FaissDocumentIndex(
    SectionSchemaMixin,
    DocumentIngestionMixin,
    FaissIndexBackendMixin,
    IndexLifecycleMixin,
    SearchMixin,
):
    """
    Class for FAISS indexing and search over documents of any type (contracts,
    reports, evidence, statements of defense, etc.). The document type is always
    received as a parameter on the methods (document_type) — never fixed on the
    class.

    Its behavior is implemented across a handful of mixins, one per concern, each in
    its own module and composed here — the class itself still exposes every method as
    if it were defined directly on it (mixin methods freely call each other via
    `self`, regardless of which module they live in):
      - `SectionSchemaMixin` (`_sections.py`): the LLM-calibrated section schema.
      - `DocumentIngestionMixin` (`_ingestion.py`): reading files and turning them
        into (embeddings, metadata) for the "full"/"sections"/"chunks" strategies.
      - `FaissIndexBackendMixin` (`_index_backend.py`): FAISS index construction
        (flat/IVFFlat/IVFPQ) and the low-level dense-search path.
      - `IndexLifecycleMixin` (`_lifecycle.py`): build/save/load/unload and
        adding new documents to already-loaded indices.
      - `SearchMixin` (`_search.py`): dense/hybrid search, reranking, strategy
        comparison/scoring, and the `generate_search*` shortcuts.

    The split into sections (the "sections" strategy) doesn't assume any fixed
    structure: the section schema for each document_type is calibrated via an LLM
    from sample documents (see `register_document_type`) and cached per type.

    The FAISS index type is chosen by corpus size when `index_type="auto"` (the
    default): Flat (exact) for small corpora, IVFFlat/IVFPQ (approximate, faster
    and lighter on memory) for medium/large corpora. See `_build_faiss_index`.
    """

    SUPPORTED_FILE_EXTENSIONS = constants.SUPPORTED_FILE_EXTENSIONS

    def __init__(self, base_path: str,
                 openai_key: Optional[str] = None,
                 embedding_model: str = config.DEFAULT_EMBEDDING_MODEL,
                 embedding_dim: Optional[int] = None,
                 section_extraction_model: str = config.DEFAULT_SECTION_EXTRACTION_MODEL,
                 embedding_provider: Optional[EmbeddingProvider] = None,
                 chat_provider: Optional[ChatProvider] = None,
                 rerank_provider: Optional[RerankProvider] = None,
                 structure_provider: Optional[StructureProvider] = None,
                 text_extractor: Optional[Callable[[str], str]] = None,
                 embedding_batch_size: int = config.DEFAULT_EMBEDDING_BATCH_SIZE,
                 num_threads: Optional[int] = None,
                 index_type: str = config.DEFAULT_INDEX_TYPE,
                 auto_index_thresholds: Tuple[int, int] = config.DEFAULT_AUTO_INDEX_THRESHOLDS,
                 ivf_nlist: Optional[int] = None,
                 ivf_nprobe: int = config.DEFAULT_IVF_NPROBE,
                 pq_m: int = config.DEFAULT_PQ_M,
                 pq_nbits: int = config.DEFAULT_PQ_NBITS,
                 use_mps: Optional[bool] = None,
                 embedding_query_prefix: str = "",
                 embedding_document_prefix: str = ""):
        """
        Parameters:
            base_path (str): Base path where the documents are stored.
            openai_key (Optional[str]): OpenAI API key. If None, looked up from .env.
                To use an OpenAI-compatible API other than OpenAI's own (a local server —
                LM Studio, oMLX, vLLM, etc.), set the `OPENAI_BASE_URL` environment
                variable before instantiating; the OpenAI SDK reads it automatically
                (no separate `base_url` parameter here).
            embedding_model (str): Embedding model to use. Defaults to "text-embedding-3-large".
            embedding_dim (Optional[int]): Embedding vector dimension. Required only for
                models outside `constants.EMBEDDING_DIMENSIONS` (e.g., a local/non-OpenAI
                embedding model served through an OpenAI-compatible API — see `openai_key`).
                If None and `embedding_model` is a known OpenAI model, the dimension is
                looked up automatically.
            section_extraction_model (str): OpenAI chat model used to calibrate the
                section schema per document type. Defaults to "gpt-4o-mini".
            embedding_provider (Optional[EmbeddingProvider]): Custom embeddings backend
                (see `providers.py`). If None (the default), an
                `OpenAICompatibleEmbeddingProvider` is built from `embedding_model`/
                `embedding_dim`/`openai_key` — the OpenAI-compatible path described above.
                Pass your own object (any type with a `.dimension` attribute and an
                `.embed(texts) -> List[List[float]]` method) to use a backend that
                doesn't speak the OpenAI protocol at all.
            chat_provider (Optional[ChatProvider]): Custom chat backend used to calibrate
                section schemas (see `providers.py`). If None (the default), an
                `OpenAICompatibleChatProvider` is built from `section_extraction_model`/
                `openai_key`. Pass your own object (any type with a
                `.complete_structured(prompt, json_schema) -> dict` method) otherwise.
            rerank_provider (Optional[RerankProvider]): Backend for `rerank_results`
                (see [Plugging in a custom provider](#plugging-in-a-custom-provider)).
                Unlike `embedding_provider`/`chat_provider`, there's no default here —
                reranking is a genuinely new capability, not something already built
                into the OpenAI-compatible path. `providers.CrossEncoderRerankProvider`
                is the built-in option (needs the optional `sentence-transformers`
                dependency: `pip install -e ".[rerank]"`).
            structure_provider (Optional[StructureProvider]): If set, the `sections`
                strategy is built from this provider's `extract_sections(file_path)`
                instead of the LLM-calibrated pattern schema (`register_document_type`/
                `extract_sections`) — no calibration needed, each document's own
                structure defines its sections. `providers.DoclingStructureProvider`
                is the built-in option (needs the optional `docling` dependency:
                `pip install -e ".[docling]"`). No default — like `rerank_provider`,
                this is an opt-in capability, not part of the OpenAI-compatible path.
            text_extractor (Optional[Callable[[str], str]]): If set, `read_document`
                calls it for every file instead of reading/OCR'ing it here — the way to
                index text a pipeline of your own already extracted (another OCR engine,
                a cache, a database). It receives the file path `build_indices` found
                and returns that document's text; raising is fine (`build_indices` logs
                the file as unreadable and moves on). Discovery still only picks up
                `SUPPORTED_FILE_EXTENSIONS` files, and `metadata["file"]` still holds
                the real path, so nothing else in the pipeline changes.
            embedding_batch_size (int): How many texts are sent per call to the
                embeddings API. Defaults to 100.
            num_threads (Optional[int]): Number of threads FAISS should use for search.
                If None, uses all available cores (os.cpu_count()).
            index_type (str): "auto" (chosen by size), or fixed: "flat", "ivf_flat", "ivf_pq".
            auto_index_thresholds (Tuple[int, int]): (flat_limit, ivf_flat_limit) used
                when index_type="auto". Defaults to (10_000, 80_000).
            ivf_nlist (Optional[int]): Number of clusters for IVFFlat/IVFPQ. If None, uses
                approximately sqrt(n), adjusted down if there aren't enough vectors.
            ivf_nprobe (int): How many clusters are visited per search in IVFFlat/IVFPQ
                (higher = more accurate and slower). Defaults to 8.
            pq_m (int): Number of sub-quantizers for IVFPQ. Must divide embedding_dim.
            pq_nbits (int): Bits per sub-quantizer for IVFPQ. Defaults to 8.
            use_mps (Optional[bool]): Deprecated, no effect (emits a DeprecationWarning
                when passed). It used to route "flat" index searches through torch on
                Apple Silicon's GPU, which measured 2.5-7x slower than FAISS's own CPU
                search at every index size tested (1.3k-200k vectors), so that path was
                removed. The CUDA path (faiss-gpu) is `use_gpu` in `load_indices`.
            embedding_query_prefix (str): Prepended to every search query before it's
                embedded. Defaults to "" (none).
            embedding_document_prefix (str): Prepended to every indexed text (chunk,
                section) before it's embedded. Defaults to "" (none). Some retrieval
                embedding models are trained with asymmetric prefixes and search better
                with them — e.g. jina-embeddings-v5 retrieval: "Query: "/"Document: ";
                E5: "query: "/"passage: "; OpenAI's text-embedding-3 models use none.
                The document prefix an index was built with is saved alongside it, and
                `load_indices` warns when it differs from this instance's — changing it
                means rebuilding the indices.

        Returns:
            None
        """
        self.base_path = base_path
        self.indices = {}
        self.embedding_model = embedding_model
        self.section_extraction_model = section_extraction_model
        self.section_schemas: Dict[str, Dict[str, List[str]]] = {}
        # Lazily built (from already-loaded metadata) and cached per document_type/strategy
        # the first time evaluate_strategy_hybrid is called for that combination.
        self._bm25_indices: Dict[str, Dict[str, BM25Okapi]] = {}
        # Embedding rows appended by add_new_documents since each strategy was built/loaded,
        # per document_type/strategy — written out (and cleared) by save_indices.
        self._unsaved_embeddings: Dict[str, Dict[str, List[np.ndarray]]] = {}
        self.embedding_batch_size = embedding_batch_size
        self.embedding_query_prefix = embedding_query_prefix
        self.embedding_document_prefix = embedding_document_prefix
        self.index_type = index_type
        self.auto_index_thresholds = auto_index_thresholds
        self.ivf_nlist = ivf_nlist
        self.ivf_nprobe = ivf_nprobe
        self.pq_m = pq_m
        self.pq_nbits = pq_nbits

        if use_mps is not None:
            warnings.warn(
                "use_mps is deprecated and has no effect: searching 'flat' indices via torch/MPS measured "
                "slower than FAISS's CPU search at every index size tested, so that path was removed.",
                DeprecationWarning, stacklevel=2,
            )

        faiss.omp_set_num_threads(num_threads or os.cpu_count() or 1)

        if not openai_key:
            openai_key = os.getenv("OPENAI_API_KEY")
        if not embedding_provider or not chat_provider:
            # Only the OpenAI-compatible default path needs a key — a fully custom
            # embedding_provider + chat_provider pair doesn't touch OPENAI_API_KEY at all.
            if not openai_key:
                raise ValueError("OPENAI_API_KEY not found. Set it in .env or pass it as a parameter")

        self.embedding_provider = embedding_provider or OpenAICompatibleEmbeddingProvider(
            model=embedding_model, api_key=openai_key, dimension=embedding_dim
        )
        self.chat_provider = chat_provider or OpenAICompatibleChatProvider(
            model=section_extraction_model, api_key=openai_key
        )
        self.rerank_provider = rerank_provider
        self.structure_provider = structure_provider
        self.text_extractor = text_extractor
        self.embedding_dim = self.embedding_provider.dimension

        logger.info(_("Initialized FaissDocumentIndex with model %(model)s and dimension %(dim)s") % {"model": getattr(self.embedding_provider, "model", embedding_model), "dim": self.embedding_dim})
