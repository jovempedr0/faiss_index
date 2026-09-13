import os
import gc
import re
import numpy as np
import faiss
import json
import nltk
import time
import openai
import logging
import psutil

from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from utils_ocr import extract_text_from_file_ocr_fallback

import config
import constants
from i18n import _

try:
    # Optional dependency: only used to accelerate search on "flat" indices via
    # GPU on Apple Silicon (MPS). FAISS has no Metal backend — CUDA (faiss-gpu)
    # remains the only native GPU path for FAISS itself.
    import torch
except ImportError:
    torch = None

# faiss-cpu (the only variant installable on macOS) doesn't expose these attributes —
# only faiss-gpu (Linux/CUDA) has them. Checking the installed build's capability, instead
# of trying StandardGpuResources()/index_cpu_to_gpu() and relying on the except to find out
# it doesn't exist, avoids a "failed to move to GPU" warning on every index loaded on any
# machine without CUDA (macOS included).
FAISS_HAS_GPU_SUPPORT = all(
    hasattr(faiss, attr) for attr in ("StandardGpuResources", "index_cpu_to_gpu", "GpuIndex")
)

logger = logging.getLogger(__name__)

def remove_stop_words(sentence:str) -> str:
	"""
	Remove Portuguese stop words from a sentence.

	Parameters:
		sentence (str): The sentence to remove stop words from.

	Returns:
		str: The sentence without stop words.
	"""
	words = sentence.split()

	filtered_words = [word for word in words if word not in config.STOPWORDS_PT]

	return ' '.join(filtered_words)


def clean_text(text:str) -> list[str]:
    """
    Cleans the text by removing punctuation, extra whitespace, and lowercasing it.

    Parameters:
        text (str): The text to clean.

    Returns:
        str: The cleaned text, without punctuation or extra whitespace, and lowercased.
    """
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)     # Remove extra whitespace
    text = text.lower()                  # Lowercase
    text = remove_stop_words(text)  # Uses the .text attribute to get the string
    return [text.strip()]


class FaissDocumentIndex:
    SUPPORTED_FILE_EXTENSIONS = constants.SUPPORTED_FILE_EXTENSIONS

    def __init__(self, base_path: str,
                 openai_key: Optional[str] = None,
                 embedding_model: str = config.DEFAULT_EMBEDDING_MODEL,
                 embedding_dim: Optional[int] = None,
                 section_extraction_model: str = config.DEFAULT_SECTION_EXTRACTION_MODEL,
                 embedding_batch_size: int = config.DEFAULT_EMBEDDING_BATCH_SIZE,
                 num_threads: Optional[int] = None,
                 index_type: str = config.DEFAULT_INDEX_TYPE,
                 auto_index_thresholds: Tuple[int, int] = config.DEFAULT_AUTO_INDEX_THRESHOLDS,
                 ivf_nlist: Optional[int] = None,
                 ivf_nprobe: int = config.DEFAULT_IVF_NPROBE,
                 pq_m: int = config.DEFAULT_PQ_M,
                 pq_nbits: int = config.DEFAULT_PQ_NBITS,
                 use_mps: bool = config.DEFAULT_USE_MPS):
        """
        Class for FAISS indexing and search over documents of any type (contracts,
        reports, evidence, statements of defense, etc.). The document type is always
        received as a parameter on the methods (document_type) — never fixed on the
        class.

        The split into sections (the "sections" strategy) doesn't assume any fixed
        structure: the section schema for each document_type is calibrated via an LLM
        from sample documents (see `register_document_type`) and cached per type.

        The FAISS index type is chosen by corpus size when `index_type="auto"` (the
        default): Flat (exact) for small corpora, IVFFlat/IVFPQ (approximate, faster
        and lighter on memory) for medium/large corpora. See `_build_faiss_index`.

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

        Returns:
            None
        """
        self.base_path = base_path
        self.indices = {}
        self.embedding_model = embedding_model
        self.section_extraction_model = section_extraction_model
        self.section_schemas: Dict[str, Dict[str, List[str]]] = {}
        self.embedding_batch_size = embedding_batch_size
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
        if not openai_key:
            raise ValueError("OPENAI_API_KEY not found. Set it in .env or pass it as a parameter")
        # Instance-level client (instead of mutating openai.api_key globally) — avoids two
        # instances with different keys, or concurrent threads, stepping on each other.
        self.client = openai.OpenAI(api_key=openai_key)

        if embedding_dim is not None:
            self.embedding_dim = embedding_dim
        elif embedding_model in constants.EMBEDDING_DIMENSIONS:
            self.embedding_dim = constants.EMBEDDING_DIMENSIONS[embedding_model]
        else:
            raise ValueError(
                f"Unknown embedding dimension for model '{embedding_model}'. "
                "Pass embedding_dim explicitly for models outside constants.EMBEDDING_DIMENSIONS "
                "(e.g., a local/non-OpenAI embedding model)."
            )

        logger.info(_("Initialized FaissDocumentIndex with model %(model)s and dimension %(dim)s") % {"model": embedding_model, "dim": self.embedding_dim})


    def is_index_loaded(self, document_types: list[str], strategies: list[str], require_gpu: bool) -> bool:
        """
        Checks whether the indices for all combinations of document types and
        strategies have already been loaded into memory.

        Parameters:
            document_types (list[str]): The list of document types to check.
            strategies (list[str]): The list of strategies to check.
            require_gpu (bool): If True, also checks whether the index is being
                                 searched with GPU acceleration (CUDA or MPS).

        Returns:
            bool: True if ALL requested combinations are loaded (and on GPU, if requested),
                  False if any of them fails.
        """
        if not document_types or not strategies:
            logger.warning(_("The 'document_types' and 'strategies' lists cannot be empty."))
            return False

        for document_type in document_types:
            for strategy in strategies:
                if not self._is_single_index_loaded(document_type, strategy, require_gpu):
                    return False

        num_combinations = len(document_types) * len(strategies)
        logger.info(_("All %(num_combinations)s requested index combinations are already loaded.") % {"num_combinations": num_combinations})
        return True

    def _is_single_index_loaded(self, document_type: str, strategy: str, require_gpu: bool) -> bool:
        """Checks whether a single document type/strategy combination is loaded correctly."""
        if document_type not in self.indices:
            logger.debug(_("Index for document type '%(document_type)s' not found.") % {"document_type": document_type})
            return False

        if strategy not in self.indices[document_type]:
            logger.debug(_("Strategy '%(strategy)s' for document type '%(document_type)s' not found.") % {"strategy": strategy, "document_type": document_type})
            return False

        index, metadata, embeddings = self.indices[document_type][strategy]
        if index is None or metadata is None or embeddings is None:
            logger.debug(_("Data for '%(document_type)s/%(strategy)s' is not loaded (None).") % {"document_type": document_type, "strategy": strategy})
            return False

        # "GPU" covers two independent paths: index moved to GPU via CUDA
        # (faiss-gpu, nonexistent on macOS) OR search accelerated via MPS (self._should_use_mps).
        is_gpu_index = (
            (FAISS_HAS_GPU_SUPPORT and isinstance(index, faiss.GpuIndex))
            or self._should_use_mps(index)
        )
        if require_gpu and not is_gpu_index:
            logger.debug(_("Index for '%(document_type)s/%(strategy)s' is not on GPU, but was expected to be.") % {"document_type": document_type, "strategy": strategy})
            return False

        return True


    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generates embeddings for a list of texts using the OpenAI embedding model,
        in batches (`self.embedding_batch_size` texts per call) to reduce the number of
        network round-trips relative to one call per text.

        Parameters:
            texts (List[str]): List of strings with the texts to generate embeddings for.

        Returns:
            np.ndarray: A numpy array with the embeddings generated for each text, in the
                        same order as `texts` (1 row per text — texts that end up empty
                        after cleaning get a zero vector, to preserve alignment with
                        metadata in the create_embeddings_* callers).
        """
        cleaned_texts = []
        for text in texts:
            # Remove null bytes and control characters that would invalidate the JSON
            text_limpo = (
                text.replace('\x00', '')
                    .encode('utf-8', errors='ignore')
                    .decode('utf-8')
                    .strip()
            )
            cleaned_texts.append(text_limpo[:constants.MAX_EMBEDDING_INPUT_CHARS])

        embeddings: List[Optional[List[float]]] = [None] * len(cleaned_texts)

        for batch_start in range(0, len(cleaned_texts), self.embedding_batch_size):
            batch_indices = range(batch_start, min(batch_start + self.embedding_batch_size, len(cleaned_texts)))
            batch_indices_with_text = [i for i in batch_indices if cleaned_texts[i]]
            batch_inputs = [cleaned_texts[i] for i in batch_indices_with_text]

            if batch_inputs:
                response = self.client.embeddings.create(input=batch_inputs, model=self.embedding_model)
                for data in response.data:
                    embeddings[batch_indices_with_text[data.index]] = data.embedding

            for i in batch_indices:
                if embeddings[i] is None:
                    logger.warning(_("Text became empty after cleaning; using a zero vector to preserve alignment with metadata."))
                    embeddings[i] = [0.0] * self.embedding_dim

        return np.array(embeddings)


    def read_document(self, file_path: str) -> str:
        """
        Reads the content of a document file.
        If the file is a .txt, reads it directly as UTF-8 text. Otherwise (.pdf/.doc/.docx),
        uses `extract_text_from_file_ocr_fallback` to extract the text (with OCR fallback).

        Parameters:
            file_path (str): Path to the document file.

        Returns:
            str: Text content extracted from the file.
        """
        if file_path.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        if file_path.lower().endswith(self.SUPPORTED_FILE_EXTENSIONS):
            return extract_text_from_file_ocr_fallback(file_path)


    def extract_sections(self, text: str, document_type: str) -> Dict[str, str]:
        """
        Extracts sections from a text based on the calibrated schema for the document type.
        This method doesn't assume any fixed structure: it uses the section schema (section
        name -> text patterns that mark its start) calibrated via LLM and stored in
        `self.section_schemas[document_type]` (see `register_document_type`).

        Parameters:
            text (str): The full document text to process.
            document_type (str): Document type whose section schema will be used.

        Returns:
            Dict[str, str]: A dictionary where the keys are "completo" (full text),
                             "cabecalho" (text before the first recognized section) and
                             the schema's section names, and the values are the
                             corresponding texts.

        Raises:
            ValueError: If no section schema is calibrated for `document_type`.
        """
        schema = self.section_schemas.get(document_type)
        if not schema:
            raise ValueError(
                f"No section schema calibrated for document type '{document_type}'. "
                "Call register_document_type(document_type, sample_texts=...) before using the 'sections' strategy."
            )

        sections = {"completo": text, "cabecalho": ""}
        sections.update({section_name: "" for section_name in schema})

        lines = text.split('\n')
        current_section = "cabecalho"

        for line in lines:
            lower = line.lower().strip()

            for section_name, patterns in schema.items():
                if any(pattern in lower for pattern in patterns):
                    current_section = section_name
                    break

            sections[current_section] += line + "\n"

        return {k: v.strip() for k, v in sections.items() if v.strip()}


    def create_embeddings_full(self, docs: List[Tuple[str, str]]) -> Tuple[np.ndarray, List]:
        """
        Generates embeddings for a list of full documents and returns the embeddings
        together with the metadata.

        Parameters:
            docs (List[Tuple[str, str]]): List of tuples, each containing the file name
                and the document's full text.

        Returns:
            Tuple[np.ndarray, List]: A tuple containing:
                - A numpy array with the texts' embeddings.
                - A list of metadata dicts, including the file name and the type ("full").
        """
        texts = [doc[1] for doc in docs]
        embeddings = self.get_embeddings(texts)
        # Each entry stores ONLY that document's own text (a 1-element list).
        # Previously the whole corpus (`texts`) was stored in every entry, which made
        # retrieval degenerate — see processar_resultados_busca.
        metadata = [{"file": doc[0], "type": "full", "content": [doc[1]], "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())} for doc in docs]
        return embeddings, metadata


    def create_embeddings_sections(self, docs: List[Tuple[str, str]], document_type: str) -> Tuple[np.ndarray, List]:
        """
        Creates embeddings for sections extracted from documents, using the section
        schema calibrated for `document_type` (see `extract_sections`/`register_document_type`).

        Parameters:
            docs (List[Tuple[str, str]]): List of tuples with the file path and the document's content.
            document_type (str): Document type whose section schema will be used.

        Returns:
            Tuple[np.ndarray, List]:
                - A numpy array with the extracted sections' embeddings.
                - A list of metadata for each section, including the file path, the
                  section name, and the type.
        """
        all_texts = []
        all_metadata = []

        for file_path, content in docs:
            sections = self.extract_sections(content, document_type)
            for section_name, section_text in sections.items():
                if section_text and section_name != "completo":
                    all_texts.append(section_text)
                    all_metadata.append({
                        "file": file_path,
                        "section_name": section_name,
                        "type": "section",
                        "section_text": section_text,
                        "content": content,
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
                    })

        embeddings = self.get_embeddings(all_texts)
        return embeddings, all_metadata


    def create_embeddings_chunks(self, docs: List[Tuple[str, str]], chunk_size: int = constants.DEFAULT_CHUNK_SIZE_WORDS) -> Tuple[np.ndarray, List]:
        """
        Splits documents into chunks, generates embeddings for each chunk, and returns
        the embeddings together with the metadata.

        Parameters:
            docs (List[Tuple[str, str]]): List of tuples with the file path and the document's content.
            chunk_size (int, optional): Size of each chunk in number of words. Defaults to 500.

        Returns:
            Tuple[np.ndarray, List]:
                - A numpy array with the chunks' embeddings.
                - A list of metadata dicts for each chunk, including the file path, chunk
                  index, and type.
        """
        all_texts = []
        all_metadata = []

        for file_path, content in docs:
            words = content.split()
            chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size//2)]

            for i, chunk in enumerate(chunks):
                if chunk:
                    all_texts.append(chunk)
                    all_metadata.append({
                        "file": file_path,
                        "chunk_index": i,
                        "type": "chunk",
                        "chunk_text": chunk,
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
                    })

        embeddings = self.get_embeddings(all_texts)
        return embeddings, all_metadata

    MAX_SECTION_SAMPLE_DOCS = constants.MAX_SECTION_SAMPLE_DOCS
    MAX_SECTION_SAMPLE_CHARS = constants.MAX_SECTION_SAMPLE_CHARS

    def register_document_type(
        self,
        document_type: str,
        sample_texts: List[str],
        force_recalibrate: bool = False
    ) -> Dict[str, List[str]]:
        """
        Calibrates, via LLM, the section schema of a document type from sample texts,
        and stores the result in cache (`self.section_schemas[document_type]`).
        `build_indices` calls this method automatically when there's no cached or
        on-disk schema for the type; it can also be called manually to recalibrate.

        Parameters:
            document_type (str): Document type to calibrate.
            sample_texts (List[str]): Sample document texts of that type.
            force_recalibrate (bool): If True, recalibrates even if a schema is already cached.

        Returns:
            Dict[str, List[str]]: The calibrated schema (section name -> text patterns).
            Empty dict if the LLM couldn't identify any section.
        """
        if document_type in self.section_schemas and not force_recalibrate:
            logger.info(_("Section schema for '%(document_type)s' is already cached. Skipping recalibration.") % {"document_type": document_type})
            return self.section_schemas[document_type]

        if not sample_texts:
            raise ValueError("At least one sample text must be provided to calibrate the section schema.")

        logger.info(
            _("Calibrating section schema for '%(document_type)s' via LLM (%(model)s) with %(num_samples)s sample document(s)...")
            % {"document_type": document_type, "model": self.section_extraction_model, "num_samples": len(sample_texts)}
        )
        schema = self._infer_section_schema_via_llm(document_type, sample_texts)

        if not schema:
            logger.warning(
                _("The LLM did not identify any section for '%(document_type)s'. The 'sections' strategy will be skipped for this document type.")
                % {"document_type": document_type}
            )

        self.section_schemas[document_type] = schema
        return schema

    def _infer_section_schema_via_llm(self, document_type: str, sample_texts: List[str]) -> Dict[str, List[str]]:
        """
        Calls an OpenAI chat model to infer the recurring structural sections across
        the sample texts and the patterns that mark the start of each one.
        """
        samples = sample_texts[: self.MAX_SECTION_SAMPLE_DOCS]
        samples_block = "\n\n---\n\n".join(text[: self.MAX_SECTION_SAMPLE_CHARS] for text in samples)

        prompt = (
            f"You will analyze {len(samples)} sample document(s) of type '{document_type}'. "
            "Identify the structural sections that repeat across them (for example: header, "
            "facts, grounds, requests, conclusion — but adapt the names to the actual "
            "document type, don't assume it's a legal document). For each section, list "
            "2 to 6 short words or phrases, in the document's language, that usually "
            "appear on the line that marks the start of that section.\n\n"
            f"Sample documents:\n{samples_block}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.section_extraction_model,
                messages=[{"role": "user", "content": prompt}],
                response_format=constants.SECTION_SCHEMA_RESPONSE_FORMAT,
            )
            parsed = json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(_("Failed to calibrate section schema via LLM for '%(document_type)s': %(error)s") % {"document_type": document_type, "error": e}, exc_info=True)
            return {}

        schema: Dict[str, List[str]] = {}
        for section in parsed.get("sections", []):
            name = section.get("name", "").strip().lower().replace(" ", "_")
            patterns = [p.strip().lower() for p in section.get("patterns", []) if p.strip()]
            if name and patterns:
                schema[name] = patterns

        return schema

    def _section_schema_path(self, document_type: str, directory: Path) -> Path:
        return Path(directory) / f"{document_type}_section_schema.json"

    def _save_section_schema(self, document_type: str, output_dir: Path) -> None:
        """Persists to disk the calibrated section schema for `document_type`, if any."""
        schema = self.section_schemas.get(document_type)
        if not schema:
            return
        path = self._section_schema_path(document_type, output_dir)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(schema, f, ensure_ascii=False, indent=4)
        logger.info(_("Section schema for '%(document_type)s' saved to '%(path)s'.") % {"document_type": document_type, "path": path})

    def _load_section_schema(self, document_type: str, document_dir: Path) -> None:
        """Loads from disk, if it exists, the calibrated section schema for `document_type`."""
        path = self._section_schema_path(document_type, document_dir)
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.section_schemas[document_type] = json.load(f)
            logger.info(_("Section schema for '%(document_type)s' loaded from '%(path)s'.") % {"document_type": document_type, "path": path})
        except Exception as e:
            logger.warning(_("Failed to load section schema from '%(path)s': %(error)s") % {"path": path, "error": e})

    MIN_TRAINING_POINTS_PER_CLUSTER = constants.MIN_TRAINING_POINTS_PER_CLUSTER

    def _select_index_kind(self, n: int) -> str:
        """Decides which FAISS index kind to build for `n` vectors."""
        if self.index_type != "auto":
            return self.index_type
        small_max, medium_max = self.auto_index_thresholds
        if n <= small_max:
            return "flat"
        elif n <= medium_max:
            return "ivf_flat"
        return "ivf_pq"

    def _compute_nlist(self, n: int) -> int:
        """
        Number of clusters for IVFFlat/IVFPQ: uses `self.ivf_nlist` if given, otherwise
        ~sqrt(n), always respecting a minimum number of training points per cluster
        (otherwise FAISS trains poorly).
        """
        desired = self.ivf_nlist or max(1, int(np.sqrt(n)))
        max_supported = max(1, n // self.MIN_TRAINING_POINTS_PER_CLUSTER)
        return max(1, min(desired, max_supported))

    def _should_use_mps(self, index) -> bool:
        """
        Decides whether search on this index should run via MPS (torch) instead of
        FAISS CPU. Only applies to "flat" indices — IVF/IVFPQ indices (identified here
        by the presence of the `nprobe` attribute, which only exists on IVF variants)
        keep using FAISS's native approximate search, which is already the right
        strategy for their corpus size.
        """
        return self.mps_device is not None and index is not None and not hasattr(index, "nprobe")

    def _mps_flat_search(self, index: faiss.Index, query_embedding: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Exhaustive search equivalent to an IndexFlatL2's, running on MPS (Apple Silicon
        GPU) via torch instead of FAISS CPU. Reconstructs the stored vectors from the
        index itself (without keeping a separate embeddings array) and computes the
        squared L2 distance — the same metric `faiss.IndexFlatL2.search` uses.

        Parameters:
            index (faiss.Index): "flat" index (no `nprobe`) to read the vectors from.
            query_embedding (np.ndarray): Query vector(s).
            k (int): Number of nearest neighbors to return.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (distances, indices), in the same format
            `faiss.Index.search` returns.
        """
        n = index.ntotal
        if n == 0:
            return np.zeros((query_embedding.shape[0], 0), dtype="float32"), np.zeros((query_embedding.shape[0], 0), dtype="int64")

        k = min(k, n)
        stored = np.asarray(index.reconstruct_n(0, n), dtype="float32")

        stored_t = torch.from_numpy(stored).to(self.mps_device)
        query_t = torch.from_numpy(query_embedding.astype("float32")).to(self.mps_device)

        sq_distances = torch.cdist(query_t, stored_t, p=2) ** 2
        top_distances, top_indices = torch.topk(sq_distances, k, dim=1, largest=False)

        return top_distances.detach().cpu().numpy(), top_indices.detach().cpu().numpy()

    def _build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Creates (and trains, if needed) a FAISS index suited to the corpus size:
        "flat" (exact) for small corpora, "ivf_flat"/"ivf_pq" (approximate, faster and
        lighter on memory) for medium/large corpora — see `index_type` and
        `auto_index_thresholds` on the constructor.

        Parameters:
            embeddings (np.ndarray): Vectors to index.

        Returns:
            faiss.Index: The already-trained (if applicable) index with the vectors added.
        """
        n = embeddings.shape[0]
        vectors = embeddings.astype('float32')
        kind = self._select_index_kind(n)

        if kind == "flat":
            index = faiss.IndexFlatL2(self.embedding_dim)
            index.add(vectors)
            logger.info(_("'flat' index created with %(n)s vectors (exact search).") % {"n": n})
            return index

        nlist = self._compute_nlist(n)
        quantizer = faiss.IndexFlatL2(self.embedding_dim)

        if kind == "ivf_flat":
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
        elif kind == "ivf_pq":
            index = faiss.IndexIVFPQ(quantizer, self.embedding_dim, nlist, self.pq_m, self.pq_nbits)
        else:
            raise ValueError(f"Unknown index type: {kind}")

        index.train(vectors)
        index.add(vectors)
        index.nprobe = self.ivf_nprobe
        # Enables index.reconstruct(i) (used in evaluate_strategy) for IVF* indices.
        index.make_direct_map()
        logger.info(
            _("'%(kind)s' index created with %(n)s vectors, nlist=%(nlist)s, nprobe=%(nprobe)s (approximate search).")
            % {"kind": kind, "n": n, "nlist": nlist, "nprobe": self.ivf_nprobe}
        )
        return index

    def _create_and_save_index(self, strategy_name: str, embeddings: np.ndarray, metadata: list, document_type: str, output_dir: Path) -> Tuple[Optional[faiss.Index], list, Optional[np.ndarray]]:
        """
        Creates and saves a FAISS index for a specific strategy.

        Parameters:
            strategy_name (str): Strategy name (e.g., "full", "sections", "chunks").
            embeddings (np.ndarray): Numpy array with the embeddings.
            metadata (list): List of metadata corresponding to the embeddings.
            document_type (str): Document type used to name the files.
            output_dir (Path): Directory where the index files will be saved.

        Returns:
            tuple: A tuple with the FAISS index, the metadata, and the embeddings.
        """
        if embeddings.shape[0] == 0:
            logger.warning(_("No embeddings generated for strategy '%(strategy_name)s'. Skipping.") % {"strategy_name": strategy_name})
            return None, [], None

        logger.info(_("Building and saving index for strategy: '%(strategy_name)s'...") % {"strategy_name": strategy_name})
        index = self._build_faiss_index(embeddings)

        faiss.write_index(index, str(output_dir / f"{document_type}_{strategy_name}.index"))
        with open(output_dir / f"{document_type}_{strategy_name}_metadata.json", "w", encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        np.save(output_dir / f"{document_type}_{strategy_name}_embeddings.npy", embeddings)

        logger.info(_("Index for '%(strategy_name)s' saved successfully.") % {"strategy_name": strategy_name})
        return index, metadata, embeddings


    def build_indices(
        self,
        document_type: str,
        base_data_dir: str = config.DEFAULT_BASE_DATA_DIR,
        output_index_dir: str = config.DEFAULT_OUTPUT_INDEX_DIR,
        doc_limit: Optional[int] = None
    ):
        """
        Builds and saves FAISS indices for the different document indexing strategies.

        Parameters:
            document_type (str): Document type to build the indices for.
            base_data_dir (str): Base directory where the document data is stored.
            output_index_dir (str): Directory where the FAISS indices will be saved.
            doc_limit (Optional[int]): Maximum number of documents to process. If None, processes all.

        Returns:
            None
        """
        base_dir = Path(base_data_dir) / document_type
        # Subfolder per document_type, to match the layout `load_indices` expects.
        output_dir = Path(output_index_dir) / document_type
        output_dir.mkdir(parents=True, exist_ok=True)

        file_paths = sorted(p for p in base_dir.rglob('*') if p.suffix in self.SUPPORTED_FILE_EXTENSIONS)

        if doc_limit is not None and doc_limit > 0:
            logger.info(_("Found %(total)s matching file(s). Limit of %(doc_limit)s applied.") % {"total": len(file_paths), "doc_limit": doc_limit})
            file_paths = file_paths[:doc_limit]

        docs = []
        for file_path in file_paths:
            try:
                content = self.read_document(str(file_path))
                if content and content.strip():
                    docs.append((str(file_path), content))
            except Exception as e:
                logger.error(_("Error reading %(file_path)s: %(error)s") % {"file_path": file_path, "error": e}, exc_info=True)

        if not docs:
            logger.warning(_("No documents found for document type: %(document_type)s") % {"document_type": document_type})
            return

        logger.info(_("Processing %(count)s documents of %(document_type)s...") % {"count": len(docs), "document_type": document_type})

        if document_type not in self.section_schemas:
            self._load_section_schema(document_type, output_dir)
        if document_type not in self.section_schemas:
            sample_texts = [content for _, content in docs[: self.MAX_SECTION_SAMPLE_DOCS]]
            self.register_document_type(document_type, sample_texts)
        self._save_section_schema(document_type, output_dir)

        embeddings_map = {
            "full": self.create_embeddings_full(docs),
            "chunks": self.create_embeddings_chunks(docs)
        }
        if self.section_schemas.get(document_type):
            embeddings_map["sections"] = self.create_embeddings_sections(docs, document_type)
        else:
            logger.warning(_("No section schema for '%(document_type)s'. The 'sections' strategy will not be built.") % {"document_type": document_type})

        self.indices[document_type] = {}

        # Iterates over the strategies and applies the creation/save logic
        for strategy, (embeddings, meta) in embeddings_map.items():
            result = self._create_and_save_index(strategy, embeddings, meta, document_type, output_dir)
            self.indices[document_type][strategy] = result

        logger.info(_("Index construction for '%(document_type)s' complete.") % {"document_type": document_type})


    def evaluate_strategy(self, query: str, document_type: str, strategy: str, k: int = 10) -> Dict:
        """
        Evaluates a search strategy on a FAISS index for a specific document type.

        Parameters:
            query (str): Query text for similarity search.
            document_type (str): Document type to select the corresponding index.
            strategy (str): Strategy used for indexing and search.
            k (int, optional): Number of nearest neighbors to return. Defaults to 10.

        Returns:
            Dict: A dictionary with:
                - "strategy": Strategy used.
                - "search_time": Time spent searching (in seconds).
                - "results": List of results, each with:
                    - "rank": Result position.
                    - "distance": FAISS distance of the result.
                    - "metadata": Metadata associated with the result.
                    - "similarity": Cosine similarity between the query and the result.
                - "avg_distance": Average distance of the returned results.
                - "std_distance": Standard deviation of the results' distances.
        Returns an empty dict if the document type doesn't exist in the indices.
        """
        if document_type not in self.indices or strategy not in self.indices[document_type]:
            return {}

        index, metadata, _embeddings = self.indices[document_type][strategy]
        query_embedding = self.get_embeddings([query])

        start_time = time.time()
        if self._should_use_mps(index):
            distances, indices = self._mps_flat_search(index, query_embedding, k)
        else:
            distances, indices = index.search(query_embedding.astype('float32'), k)
        search_time = time.time() - start_time

        # When k > ntotal, FAISS's native search (not the MPS one, which already caps k
        # at ntotal) fills the missing slots with index -1 and a "sentinel" distance.
        # Discard those slots instead of letting -1 become Python negative indexing
        # (metadata[-1]/index.reconstruct(-1)) and contaminate the distance statistics.
        valid_mask = indices[0] >= 0
        valid_distances = distances[0][valid_mask]

        results = []
        for dist, idx in zip(distances[0][valid_mask], indices[0][valid_mask]):
            # Reconstructs the vector from the FAISS index itself (instead of keeping a
            # separate embeddings array in memory) — works the same for "flat" and "ivf_*"
            # (index.make_direct_map() is called when building/loading the index).
            stored_vector = np.asarray(index.reconstruct(int(idx))).reshape(1, -1)
            results.append({
                "rank": len(results) + 1,
                "distance": float(dist),
                "doc_index": int(idx),
                "metadata": metadata[idx],
                "similarity": float(cosine_similarity(query_embedding, stored_vector)[0][0])
            })

        return {
            "strategy": strategy,
            "search_time": search_time,
            "results": results,
            "avg_distance": float(np.mean(valid_distances)) if len(valid_distances) else 0.0,
            "std_distance": float(np.std(valid_distances)) if len(valid_distances) else 0.0
        }


    def compare_strategies(self, queries: List[str], document_type: str, strategies_compare: list[str], k: int = 15) -> Dict:
        """
        Compares different search strategies for a list of queries, evaluating each one's performance.

        Parameters:
            queries (List[str]): List of query strings to evaluate.
            document_type (str): Document type or context to use when evaluating the strategies.
            strategies_compare (list[str]): Strategies to compare (e.g.: ["full", "chunks"]).
            k (int, optional): Number of results to return per query. Defaults to 15.

        Returns:
            Dict: A dictionary with, for each query, the results of the different strategies evaluated.
        """
        comparisons = {}

        for query in queries:
            print(_("\nQuery: %(query)s...") % {"query": query[:50]})
            comparisons[query] = {}

            for strategy in strategies_compare:
                results = self.evaluate_strategy(query, document_type, strategy, k)
                comparisons[query][strategy] = results

                if results:
                    print(_("  %(strategy)s: time=%(time)ss, avg_dist=%(dist)s") % {
                        "strategy": strategy,
                        "time": f"{results['search_time']:.3f}",
                        "dist": f"{results['avg_distance']:.3f}",
                    })

        return comparisons


    def calculate_heuristic_score(self, comparison_results: Dict, keywords: Optional[List[str]] = None) -> Dict:
        """
        Computes a heuristic score to compare the search strategies based on multiple criteria.

        Parameters:
            comparison_results (Dict): Comparison results between strategies, from `compare_strategies`.
            keywords (Optional[List[str]]): List of keywords to check for presence in the results.

        Returns:
            Dict: A dictionary with the mean score and standard deviation for each strategy,
                considering the defined criteria.
        """
        scores: Dict[str, List[float]] = defaultdict(list)

        for query, strategies in comparison_results.items():
            for strategy, results in strategies.items():
                if results:
                    final_score = self._calculate_single_strategy_score(results, keywords)
                    scores[strategy].append(final_score)

        return {
            strategy: {
                "mean_score": np.mean(score_list),
                "std_score": np.std(score_list),
                "scores": score_list
            }
            for strategy, score_list in scores.items() if score_list
        }

    @staticmethod
    def _extract_metadata_text(metadata: Dict) -> str:
        """
        Returns the text of a metadata item, whichever strategy produced it
        ("full" stores it in `content`, "section" in `section_text`, "chunk" in `chunk_text`).
        """
        if "chunk_text" in metadata:
            return metadata["chunk_text"]
        if "section_text" in metadata:
            return metadata["section_text"]
        content = metadata.get("content", "")
        return " ".join(content) if isinstance(content, list) else content

    def _calculate_single_strategy_score(self, results: Dict, keywords: Optional[List[str]]) -> float:
        """
        Computes the heuristic score for a single strategy/result.
        """
        # --- Criterion 1: Speed (lower is better)
        speed_score = 1 / (1 + results['search_time'] * 10)

        # --- Criterion 2: Average distance (lower is better)
        distance_score = 1 / (1 + results['avg_distance'])

        # --- Criterion 3: Variance (more stable is better)
        variance_score = 1 / (1 + results['std_distance'])

        # --- Criterion 4: File diversity among the top-k results
        unique_files = len({r['metadata']['file'] for r in results['results']})
        diversity_score = unique_files / len(results['results']) if results['results'] else 0

        # --- Criterion 5 (new): Keyword presence
        keyword_score_value = 0.0
        if keywords:
            keyword_scores = []
            for r in results['results']:
                text = self._extract_metadata_text(r['metadata'])
                count = sum(1 for kw in keywords if kw.lower() in text.lower())
                keyword_scores.append(count / len(keywords))
            keyword_score_value = np.mean(keyword_scores) if keyword_scores else 0.0

        # --- Final score with adjusted weights
        final_score = (
            0.1 * speed_score +
            0.3 * distance_score +
            0.1 * variance_score +
            0.3 * diversity_score +
            0.2 * keyword_score_value
        )
        return final_score


    def generate_search(self, received_query:list[str], keywords:List[str], document_type:str, strategies_compare:list[str]) -> Tuple[List, Dict]:
        """
        Runs a search based on a received query, keywords, and document type.

        Parameters:
            received_query (list[str]): List containing the received query.
            keywords (List[str]): Keywords to help the search.
            document_type (str): Document type to steer the search strategy.
            strategies_compare (list[str]): Strategies to compare (e.g.: ["full", "chunks"]).

        Returns:
            tuple: A tuple with the search results and the heuristic scores.
        """

        cleaned_query = clean_text(received_query[0])
        results = self.compare_strategies(cleaned_query, document_type, strategies_compare, k=20)
        scores = self.calculate_heuristic_score(results, keywords)

        return results, scores


    def generate_search_by_type(self, received_query: str, document_type: str, strategy: str, require_gpu: bool) -> List[str]:
        """
        Runs a search based on a received query, using a specific strategy.

        Parameters:
            received_query (str): The query string to search for.
            document_type (str): Document type to steer the search strategy.
            strategy (str): Indexing strategy to use (e.g.: 'chunks').
            require_gpu (bool): If True, requires the FAISS index to be loaded on GPU.

        Returns:
            List[str]: A list of text chunks corresponding to the search results.
        """
        is_loaded = self.is_index_loaded(document_types=[document_type], strategies=[strategy], require_gpu=require_gpu)

        if is_loaded:
            logger.info(_("Index for '%(document_type)s/%(strategy)s' found. Running search") % {"document_type": document_type, "strategy": strategy})
        else:
            logger.info(_("Index for '%(document_type)s/%(strategy)s' not found. Loading now...") % {"document_type": document_type, "strategy": strategy})

            self.load_indices(path_indices=config.DEFAULT_PATH_INDICES,
                            document_types=[document_type],
                            strategies=[strategy])

            is_now_loaded = self.is_index_loaded(document_types=[document_type], strategies=[strategy], require_gpu=require_gpu)
            logger.info(_("Index for '%(document_type)s/%(strategy)s' loaded - '%(is_now_loaded)s'. Running search...") % {"document_type": document_type, "strategy": strategy, "is_now_loaded": is_now_loaded})

        cleaned_query_list = clean_text(received_query)
        if not cleaned_query_list:
            return []

        cleaned_query_str = cleaned_query_list[0]

        results = self.evaluate_strategy(cleaned_query_str, document_type, strategy, k=5)

        chunks = [res['metadata']['chunk_text'] for res in results.get('results', [])]

        return chunks


    def _log_memory_usage(self, stage: str):
        """Helper function to log current memory usage."""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        # Converts bytes to GB for easier reading
        rss_gb = mem_info.rss / (1024 ** 3)
        logger.info(_("===> Memory Usage (%(stage)s): %(rss_gb)s GB") % {"stage": stage, "rss_gb": f"{rss_gb:.2f}"})

    def load_indices(
        self,
        path_indices: str,
        document_types: list[str],
        strategies: list[str],
        use_gpu: bool = True
    ) -> dict:
        """
        Loads FAISS indices, metadata, and embeddings in a memory-optimized way.
        If use_gpu is True, tries to move the indices to the GPU.

        Parameters:
            path_indices (str): Path to the indices' base directory.
            document_types (list[str]): Document types to load.
            strategies (list[str]): Strategies to load (e.g.: ['full', 'chunks']).
            use_gpu (bool): If True, tries to move the FAISS index to the GPU.

        Returns:
            dict: Dictionary with the loaded data.
        """
        self._log_memory_usage(_("Start of `load_indices`"))
        loaded_data = {}

        for document_type in document_types:
            self.indices.setdefault(document_type, {})
            loaded_data.setdefault(document_type, {})

            document_dir = os.path.join(path_indices, document_type)
            if not os.path.exists(document_dir):
                logger.error(_("Error: Directory '%(document_dir)s' for document type '%(document_type)s' not found.") % {"document_dir": document_dir, "document_type": document_type})
                continue

            logger.info(_("Loading data from '%(document_dir)s'...") % {"document_dir": document_dir})
            self._load_section_schema(document_type, Path(document_dir))

            for strategy in strategies:
                self._load_single_strategy(
                    document_type, document_dir, strategy, use_gpu, loaded_data
                )

        self._log_memory_usage(_("End of `load_indices`"))
        return loaded_data

    def _move_index_to_gpu(self, index_cpu, strategy):
        if not FAISS_HAS_GPU_SUPPORT:
            # Build without CUDA (faiss-cpu, the only variant installable on macOS) —
            # there's nothing to try here. On Apple Silicon, real acceleration is via MPS
            # at search time (see `use_mps` on the constructor and `_should_use_mps`), not on load.
            logger.debug(
                _("   FAISS installed without CUDA GPU support — keeping '%(strategy)s' on CPU.") % {"strategy": strategy}
            )
            return index_cpu
        try:
            if not hasattr(self, "gpu_resources") or self.gpu_resources is None:
                logger.info(_("   Initializing GPU resources for FAISS..."))
                self.gpu_resources = faiss.StandardGpuResources()
            logger.info(_("   Moving index '%(strategy)s' to GPU...") % {"strategy": strategy})
            index_gpu = faiss.index_cpu_to_gpu(self.gpu_resources, 0, index_cpu)
            self._log_memory_usage(_("After moving index '%(strategy)s' to GPU") % {"strategy": strategy})
            logger.info(_("Successfully moved index to GPU."))
            return index_gpu
        except Exception as gpu_error:
            logger.warning(_("    Failed to move index to GPU: %(error)s. Using CPU.") % {"error": gpu_error})
            return index_cpu

    def _load_single_strategy(
        self, document_type, document_dir, strategy, use_gpu, loaded_data
    ):
        logger.info(_("-> Loading strategy '%(strategy)s'...") % {"strategy": strategy})

        index_path = os.path.join(document_dir, f"{document_type}_{strategy}.index")
        metadata_path = os.path.join(document_dir, f"{document_type}_{strategy}_metadata.json")
        embeddings_path = os.path.join(document_dir, f"{document_type}_{strategy}_embeddings.npy")

        if not all(os.path.exists(p) for p in [index_path, metadata_path, embeddings_path]):
            logger.warning(
                _("   Warning: Files for strategy '%(strategy)s' of type '%(document_type)s' not found. Skipping.")
                % {"strategy": strategy, "document_type": document_type}
            )
            return

        try:
            self._log_memory_usage(_("Before loading index '%(strategy)s'") % {"strategy": strategy})
            index_cpu = faiss.read_index(str(index_path))
            if hasattr(index_cpu, "nprobe"):
                index_cpu.nprobe = self.ivf_nprobe
            if hasattr(index_cpu, "make_direct_map"):
                try:
                    index_cpu.make_direct_map()
                except Exception:
                    pass  # may already have a direct map (e.g.: reconstruction after a previous `add`)
            self._log_memory_usage(_("After loading index '%(strategy)s' to CPU") % {"strategy": strategy})
            final_index = (
                self._move_index_to_gpu(index_cpu, strategy) if use_gpu else index_cpu
            )

            self._log_memory_usage(_("Before loading metadata '%(strategy)s'") % {"strategy": strategy})
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            self._log_memory_usage(_("After loading metadata '%(strategy)s'") % {"strategy": strategy})

            embeddings = np.load(embeddings_path, mmap_mode="r")

            loaded_tuple = (final_index, metadata, embeddings)
            self.indices[document_type][strategy] = loaded_tuple
            loaded_data[document_type][strategy] = loaded_tuple

            device = (
                "GPU"
                if use_gpu and FAISS_HAS_GPU_SUPPORT and isinstance(final_index, faiss.GpuIndex)
                else "CPU"
            )
            logger.info(
                _("   Successfully loaded '%(strategy)s': %(ntotal)s vectors on device '%(device)s'.")
                % {"strategy": strategy, "ntotal": final_index.ntotal, "device": device}
            )

        except Exception as e:
            logger.error(
                _("   An error occurred while loading '%(strategy)s': %(error)s") % {"strategy": strategy, "error": e},
                exc_info=True,
            )
        finally:
            gc.collect()

    def unload_indices(self, document_type: str, strategy: str = None) -> None:
        """
        Deallocates (removes the references to) one or more sets of indices from memory.

        Parameters:
            document_type (str): The document type whose indices should be removed.
            strategy (str, optional): The specific strategy to remove.
                                      If None, removes all indices for the document_type.

        Returns:
            None
        """
        if document_type not in self.indices:
            logger.info(_("Info: No index loaded for document type '%(document_type)s'.") % {"document_type": document_type})
            return

        # Case 1: Deallocate a specific strategy
        if strategy:
            if strategy in self.indices[document_type]:
                # Removes the reference to the (index, metadata, embeddings) tuple
                del self.indices[document_type][strategy]
                logger.info(_("Success: Strategy '%(strategy)s' for '%(document_type)s' has been unloaded.") % {"strategy": strategy, "document_type": document_type})

                # If the strategies dict becomes empty, also remove the document_type
                if not self.indices[document_type]:
                    del self.indices[document_type]
                    logger.info(_("Info: No other strategy remains, removing entry for '%(document_type)s'.") % {"document_type": document_type})
            else:
                logger.warning(_("Warning: Strategy '%(strategy)s' not found for '%(document_type)s'.") % {"strategy": strategy, "document_type": document_type})

        # Case 2: Deallocate all strategies for the document_type
        else:
            # Removes the reference to the whole strategies dict
            del self.indices[document_type]
            logger.info(_("Success: All indices for '%(document_type)s' have been unloaded.") % {"document_type": document_type})

        gc.collect()


    def add_new_documents(self, document_type: str, new_docs: List[Tuple[str, str]]) -> None:
        """
        Adds new documents to an existing FAISS index for a specific document type.
        This updates the index, the metadata, and the embeddings with the new documents provided.

        Parameters:
            document_type (str): The document type the new documents belong to.
            new_docs (List[Tuple[str, str]]): A list of tuples, each with the file path
                and the document's content.

        Returns:
            None
        """
        if document_type not in self.indices:
            raise ValueError(f"Index for document type '{document_type}' not found. Load the index before adding documents.")

        for strategy, (index, metadata, embeddings) in self.indices[document_type].items():
            logger.info(_("Adding new documents to strategy '%(strategy)s' for '%(document_type)s'...") % {"strategy": strategy, "document_type": document_type})

            if strategy == "full":
                new_embeddings, new_metadata = self.create_embeddings_full(new_docs)
            elif strategy == "sections":
                new_embeddings, new_metadata = self.create_embeddings_sections(new_docs, document_type)
            elif strategy == "chunks":
                new_embeddings, new_metadata = self.create_embeddings_chunks(new_docs)
            else:
                continue

            # Adds the new embeddings to the FAISS index (index.make_direct_map(), already
            # called on index creation/load, keeps index.reconstruct(i) working for the new vectors)
            index.add(new_embeddings.astype('float32'))

            # Updates the metadata. We don't np.vstack the embeddings array: it may be
            # memory-mapped from disk (mmap_mode="r") and vstack would force materializing
            # all of it into RAM. evaluate_strategy already reads vectors via
            # index.reconstruct(), so this array doesn't need to stay up to date for search to work correctly.
            metadata.extend(new_metadata)

            # Updates the tuple in the indices structure (embeddings kept as-is)
            self.indices[document_type][strategy] = (index, metadata, embeddings)

            logger.info(_("New documents added to strategy '%(strategy)s' for '%(document_type)s'. Total now: %(ntotal)s vectors.") % {"strategy": strategy, "document_type": document_type, "ntotal": index.ntotal})


    def unload_all_indices(self) -> None:
        """
        Checks whether any indices are loaded in memory and unloads all of them.

        This clears the `self.indices` dictionary, removing all references to the
        index, metadata, and embeddings objects, allowing the garbage collector to
        free the memory used.

        Returns:
            None
        """
        if not self.indices:
            logger.info(_("Info: No index is currently loaded."))
            return

        num_document_types = len(self.indices)
        logger.info(_("Info: %(num_document_types)s document type(s) found. Unloading all indices...") % {"num_document_types": num_document_types})

        # Removes the reference to the whole indices dict
        self.indices.clear()

        # Calls the garbage collector to free memory as quickly as possible
        gc.collect()

        logger.info(_("Success: All indices have been unloaded and memory has been freed."))
