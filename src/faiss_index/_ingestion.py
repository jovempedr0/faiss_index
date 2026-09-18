"""
Mixin for FaissDocumentIndex: turning raw document files into (embeddings, metadata)
pairs — reading files (with OCR fallback), calling the embedding provider, and the
three indexing strategies ("full", "sections", "chunks"). Composed into the class in
`core.py`; not meant to be imported directly by users.
"""
import codecs
import logging
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import constants
from ._text_cleaning import word_windows
from .i18n import _

logger = logging.getLogger(__name__)

# Byte-order marks, longest first: UTF-32-LE's starts with UTF-16-LE's, so checking
# UTF-16 first would read a UTF-32 file two bytes at a time. Each codec here consumes
# the mark it matched, keeping it out of the text.
_BYTE_ORDER_MARKS = (
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)


def _decode_text_file(raw: bytes, file_path: str) -> str:
    """
    Decodes a .txt file's bytes: by its byte-order mark when it has one (the file's own
    statement of its encoding, not a guess), else as UTF-8, else as
    `constants.TEXT_FALLBACK_ENCODING`.

    The fallback is what recovers text exported by Windows/legacy systems, which UTF-8
    can't decode at all — but being single-byte it also decodes bytes that mean something
    else in another encoding, so it's tried last and logged when it's what produced the
    text. If it fails too, the UnicodeDecodeError propagates: `build_indices` reports
    that file and carries on, which beats indexing mojibake.
    """
    for bom, encoding in _BYTE_ORDER_MARKS:
        if raw.startswith(bom):
            return raw.decode(encoding)

    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode(constants.TEXT_FALLBACK_ENCODING)
        logger.warning(
            _("%(file_path)s is not valid UTF-8 and has no byte-order mark; read as "
              "%(encoding)s. Check its accented characters if they look wrong.")
            % {"file_path": file_path, "encoding": constants.TEXT_FALLBACK_ENCODING}
        )
        return text


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Scales vectors (along the last axis) to unit L2 norm; all-zero vectors stay zero."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def _mean_pool(vectors: np.ndarray) -> np.ndarray:
    """
    One vector standing for several embeddings of pieces of the same text: the mean of
    the L2-normalized rows, L2-normalized again — see `create_embeddings_full` for why
    both normalizations matter.
    """
    return _l2_normalize(_l2_normalize(vectors).mean(axis=0))


class DocumentIngestionMixin:

    def get_embeddings(self, texts: List[str], prefix: str = "") -> np.ndarray:
        """
        Generates embeddings for a list of texts using the OpenAI embedding model,
        in batches (`self.embedding_batch_size` texts per call) to reduce the number of
        network round-trips relative to one call per text.

        Parameters:
            texts (List[str]): List of strings with the texts to generate embeddings for.
            prefix (str): Prepended to each text that isn't empty after cleaning — how
                `embedding_query_prefix`/`embedding_document_prefix` are applied.

        Returns:
            np.ndarray: A float32 numpy array with the embeddings generated for each text,
                        in the same order as `texts` (1 row per text — texts that end up
                        empty after cleaning get a zero vector, to preserve alignment
                        with metadata in the create_embeddings_* callers). float32 is
                        what FAISS indexes anyway; float64 only doubled memory and the
                        saved `.npy` size.
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
            cleaned_texts.append((prefix + text_limpo)[:constants.MAX_EMBEDDING_INPUT_CHARS] if text_limpo else "")

        embeddings: List[Optional[List[float]]] = [None] * len(cleaned_texts)

        for batch_start in range(0, len(cleaned_texts), self.embedding_batch_size):
            batch_indices = range(batch_start, min(batch_start + self.embedding_batch_size, len(cleaned_texts)))
            batch_indices_with_text = [i for i in batch_indices if cleaned_texts[i]]
            batch_inputs = [cleaned_texts[i] for i in batch_indices_with_text]

            if batch_inputs:
                batch_embeddings = self.embedding_provider.embed(batch_inputs)
                for i, embedding in zip(batch_indices_with_text, batch_embeddings):
                    embeddings[i] = embedding

            for i in batch_indices:
                if embeddings[i] is None:
                    logger.warning(_("Text became empty after cleaning; using a zero vector to preserve alignment with metadata."))
                    embeddings[i] = [0.0] * self.embedding_dim

        return np.array(embeddings, dtype=np.float32)

    def read_document(self, file_path: str) -> str:
        """
        Reads the content of a document file.
        If the file is a .txt, reads it directly (see `_decode_text_file` for how its
        encoding is determined). Otherwise (.pdf/.doc/.docx), uses
        `extract_text_from_file_ocr_fallback` to extract the text (with OCR fallback).

        A `text_extractor` given to the constructor replaces all of that: it's called for
        every path, so text extracted by a pipeline of your own (another OCR engine, a
        cache, a database) can be indexed without this library reading the file at all.

        Parameters:
            file_path (str): Path to the document file.

        Returns:
            str: Text content extracted from the file.

        Raises:
            ValueError: If `file_path`'s extension isn't in `SUPPORTED_FILE_EXTENSIONS`
                (not checked when a `text_extractor` handles the file).
            ImportError: If it's a .pdf/.doc/.docx and the optional OCR dependencies
                (`pip install -e ".[ocr]"`) aren't installed.
        """
        if self.text_extractor is not None:
            # Before the extension check too: what this library can parse says nothing
            # about what someone else's extractor can (build_indices still only *finds*
            # SUPPORTED_FILE_EXTENSIONS files, so this only widens direct calls).
            return self.text_extractor(file_path)

        if file_path.lower().endswith('.txt'):
            with open(file_path, 'rb') as f:
                return _decode_text_file(f.read(), file_path)
        if file_path.lower().endswith(self.SUPPORTED_FILE_EXTENSIONS):
            # Imported here rather than at module level: utils_ocr needs the optional
            # [ocr] extra (pdfplumber/pytesseract/pdf2image/Pillow), which shouldn't be
            # required just to import the package or to index .txt files.
            try:
                from .utils_ocr import extract_text_from_file_ocr_fallback
            except ImportError as e:
                raise ImportError(
                    _("Reading .pdf/.doc/.docx files needs the optional OCR dependencies. "
                      "Install them with: pip install -e \".[ocr]\"")
                ) from e
            return extract_text_from_file_ocr_fallback(file_path)
        raise ValueError(
            f"Unsupported file extension for '{file_path}'. "
            f"Supported extensions: {self.SUPPORTED_FILE_EXTENSIONS}."
        )

    def create_embeddings_full(
        self, docs: List[Tuple[str, str]], chunks: Optional[Tuple[np.ndarray, List]] = None
    ) -> Tuple[np.ndarray, List]:
        """
        Generates one embedding per full document and returns the embeddings together
        with the metadata.

        A document's vector is the mean of its chunks' embeddings (see
        `create_embeddings_chunks`), L2-normalized before and after averaging — so it
        covers the whole text, not just the first `MAX_EMBEDDING_INPUT_CHARS` characters
        a single embedding call would see. Normalizing the mean back to unit length
        keeps `IndexFlatL2` ranking equivalent to cosine similarity: a mean of unit
        vectors gets shorter the more varied the chunks are, and that shorter length
        alone would otherwise pull long, heterogeneous documents up the ranking.

        Parameters:
            docs (List[Tuple[str, str]]): List of tuples, each containing the file name
                and the document's full text.
            chunks (Optional[Tuple[np.ndarray, List]]): What `create_embeddings_chunks(docs)`
                already returned for these same `docs`, to pool from instead of embedding
                the chunks again. If None, they're computed here.

        Returns:
            Tuple[np.ndarray, List]: A tuple containing:
                - A numpy array with one vector per document (a zero vector for a
                  document with no words, which has no chunks to pool).
                - A list of metadata dicts, including the file name and the type ("full").
        """
        chunk_embeddings, chunk_metadata = chunks if chunks is not None else self.create_embeddings_chunks(docs)

        rows_by_file: Dict[str, List[int]] = defaultdict(list)
        for row, chunk_meta in enumerate(chunk_metadata):
            rows_by_file[chunk_meta["file"]].append(row)

        embeddings = np.zeros((len(docs), self.embedding_dim), dtype=np.float32)
        for i, (file_path, _content) in enumerate(docs):
            rows = rows_by_file.get(file_path)
            if rows:
                embeddings[i] = _mean_pool(chunk_embeddings[rows])

        # Each entry stores ONLY that document's own text (a 1-element list).
        # Previously the whole corpus (every doc's text) was stored in every entry, which made
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
        return self._build_section_embeddings(
            docs, lambda file_path, content: self.extract_sections(content, document_type)
        )

    def create_embeddings_sections_via_structure(self, docs: List[Tuple[str, str]]) -> Tuple[np.ndarray, List]:
        """
        Creates embeddings for sections extracted via `self.structure_provider`
        instead of the LLM-calibrated pattern schema — each document's own real structure (heading hierarchy)
        defines its sections, so no calibration is needed for the document_type.

        Parameters:
            docs (List[Tuple[str, str]]): List of tuples with the file path and the
                document's content (the content itself isn't used for extraction here
                — the provider re-reads the file directly to see its real layout).

        Returns:
            Tuple[np.ndarray, List]: Same shape as `create_embeddings_sections`.
        """
        return self._build_section_embeddings(
            docs, lambda file_path, content: self.structure_provider.extract_sections(file_path)
        )

    def _build_section_embeddings(
        self, docs: List[Tuple[str, str]], extract_fn: Callable[[str, str], Dict[str, str]]
    ) -> Tuple[np.ndarray, List]:
        """
        Shared implementation behind `create_embeddings_sections` and
        `create_embeddings_sections_via_structure` — the two differ only in how a
        document's sections are extracted (`extract_fn(file_path, content) -> sections`),
        not in how the resulting sections become embeddings/metadata.
        """
        all_texts = []
        all_metadata = []

        for file_path, content in docs:
            sections = extract_fn(file_path, content)
            for section_name, section_text in sections.items():
                if section_text and section_name != "completo":
                    all_texts.append(section_text)
                    # No copy of the whole document here: with one entry per section it
                    # multiplied the metadata file several times over. The full text
                    # lives once per file in the "full" strategy's metadata (same "file").
                    all_metadata.append({
                        "file": file_path,
                        "section_name": section_name,
                        "type": "section",
                        "section_text": section_text,
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
                    })

        return self._embed_pooled_windows(all_texts), all_metadata

    def _embed_pooled_windows(self, texts: List[str]) -> np.ndarray:
        """
        One vector per text, covering all of it: the text is split into the same
        `DEFAULT_CHUNK_SIZE_WORDS`-word windows as "chunks", and its vector is the
        pooled mean of their (document-prefixed) embeddings — `_mean_pool`, as in
        "full". Embedding a section in a single call only represented its first
        `MAX_EMBEDDING_INPUT_CHARS` characters, and sections run far past that (35 of
        113 in a real legal corpus, up to ~420k characters). Short sections go through
        the same path (one window each) rather than being embedded as-is, which
        measured the same or slightly better on that corpus and keeps a single rule.
        A text with no words gets a zero vector.
        """
        windows: List[str] = []
        rows_by_text: List[range] = []
        for text in texts:
            text_windows = word_windows(text, constants.DEFAULT_CHUNK_SIZE_WORDS)
            rows_by_text.append(range(len(windows), len(windows) + len(text_windows)))
            windows.extend(text_windows)

        window_embeddings = self.get_embeddings(windows, prefix=self.embedding_document_prefix)

        embeddings = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        for i, rows in enumerate(rows_by_text):
            if rows:
                embeddings[i] = _mean_pool(window_embeddings[rows.start:rows.stop])
        return embeddings

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
            chunks = word_windows(content, chunk_size)

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

        embeddings = self.get_embeddings(all_texts, prefix=self.embedding_document_prefix)
        return embeddings, all_metadata
