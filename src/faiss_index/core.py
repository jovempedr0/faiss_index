import logging
import os
from typing import Dict, List, Optional, Tuple

import faiss
from rank_bm25 import BM25Okapi

from . import config
from . import constants
# torch is used directly below (MPS availability check); FAISS_HAS_GPU_SUPPORT isn't
# used in this file anymore (only by _lifecycle.py/_index_backend.py) but is
# re-exported here since it used to live in this module.
from ._gpu_support import FAISS_HAS_GPU_SUPPORT, torch  # noqa: F401
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
        (flat/IVFFlat/IVFPQ) and the low-level dense-search path (incl. MPS).
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
                 embedding_batch_size: int = config.DEFAULT_EMBEDDING_BATCH_SIZE,
                 num_threads: Optional[int] = None,
                 index_type: str = config.DEFAULT_INDEX_TYPE,
                 auto_index_thresholds: Tuple[int, int] = config.DEFAULT_AUTO_INDEX_THRESHOLDS,
                 ivf_nlist: Optional[int] = None,
                 ivf_nprobe: int = config.DEFAULT_IVF_NPROBE,
                 pq_m: int = config.DEFAULT_PQ_M,
                 pq_nbits: int = config.DEFAULT_PQ_NBITS,
                 use_mps: bool = config.DEFAULT_USE_MPS,
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
            use_mps (bool): If True (default) and `torch` with MPS support is available
                (Apple Silicon GPU), search on "flat" indices runs on MPS via torch instead
                of FAISS CPU. "ivf_flat"/"ivf_pq" indices keep using FAISS's native search
                (FAISS has no Metal backend, and redoing the brute-force approximate search
                on MPS would negate the benefit of having chosen it). Has no effect outside
                Apple Silicon macOS, or without `torch` installed — falls back to FAISS CPU
                normally. Independent of `use_gpu` in `load_indices`/`_move_index_to_gpu`,
                which is the CUDA GPU path (faiss-gpu), nonexistent on macOS.
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
        self.embedding_batch_size = embedding_batch_size
        self.embedding_query_prefix = embedding_query_prefix
        self.embedding_document_prefix = embedding_document_prefix
        self.index_type = index_type
        self.auto_index_thresholds = auto_index_thresholds
        self.ivf_nlist = ivf_nlist
        self.ivf_nprobe = ivf_nprobe
        self.pq_m = pq_m
        self.pq_nbits = pq_nbits

        self.mps_device = None
        if use_mps and torch is not None:
            try:
                if torch.backends.mps.is_available():
                    self.mps_device = torch.device("mps")
                    logger.info(_("MPS (Apple Silicon GPU) available — search on 'flat' indices will be accelerated via torch."))
            except Exception as e:
                logger.warning(_("Failed to check MPS availability: %(error)s. Using FAISS CPU.") % {"error": e})

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
        self.embedding_dim = self.embedding_provider.dimension

        logger.info(_("Initialized FaissDocumentIndex with model %(model)s and dimension %(dim)s") % {"model": getattr(self.embedding_provider, "model", embedding_model), "dim": self.embedding_dim})
