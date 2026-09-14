"""
Provider abstraction for FaissDocumentIndex: small, single-purpose Protocols that
decouple embeddings/chat/vision calls from any specific SDK. `FaissDocumentIndex`
and `utils_ocr.py` depend only on these Protocols; the OpenAI-compatible
implementations below are the default (and work against any server that speaks the
OpenAI wire protocol — the real OpenAI API, or a local one like LM Studio/oMLX/vLLM),
but any object implementing the same methods can be passed in instead.
"""
import base64
import json
import os
from typing import Dict, List, Optional, Protocol

import openai

from . import constants
from .i18n import _


class EmbeddingProvider(Protocol):
    dimension: int

    def embed(self, texts: List[str]) -> List[List[float]]: ...


class ChatProvider(Protocol):
    def complete_structured(self, prompt: str, json_schema: dict) -> dict: ...


class VisionProvider(Protocol):
    def describe_image(self, image_png_bytes: bytes, prompt: str) -> str: ...


class RerankProvider(Protocol):
    def rerank(self, query: str, candidates: List[str]) -> List[float]: ...


class StructureProvider(Protocol):
    def extract_sections(self, file_path: str) -> Dict[str, str]: ...


def _resolve_embedding_dimension(model: str, dimension: Optional[int]) -> int:
    if dimension is not None:
        return dimension
    if model in constants.EMBEDDING_DIMENSIONS:
        return constants.EMBEDDING_DIMENSIONS[model]
    raise ValueError(
        _("Unknown embedding dimension for model '%(model)s'. Pass dimension explicitly "
          "for models outside constants.EMBEDDING_DIMENSIONS (e.g., a local/non-OpenAI "
          "embedding model).") % {"model": model}
    )


class OpenAICompatibleEmbeddingProvider:
    """
    Talks to any OpenAI-compatible embeddings endpoint — the real OpenAI API, or a
    local server (LM Studio, oMLX, vLLM, etc.) reached via `base_url` (or the
    `OPENAI_BASE_URL` environment variable, read automatically by the OpenAI SDK
    when `base_url` isn't passed).
    """

    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None, dimension: Optional[int] = None):
        self.model = model
        self.dimension = _resolve_embedding_dimension(model, dimension)
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def embed(self, texts: List[str]) -> List[List[float]]:
        response = self._client.embeddings.create(input=texts, model=self.model)
        # Reorders by the API's own `.index` rather than trusting response order.
        ordered = sorted(response.data, key=lambda data: data.index)
        return [data.embedding for data in ordered]


class OpenAICompatibleChatProvider:
    """
    Talks to any OpenAI-compatible chat completions endpoint. Implements both
    ChatProvider (structured-output completion, used to calibrate section schemas)
    and VisionProvider (image description, used by the OCR fallback) — `vision_model`
    is only required if `describe_image` is actually called.
    """

    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None, vision_model: Optional[str] = None):
        self.model = model
        self.vision_model = vision_model
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def complete_structured(self, prompt: str, json_schema: dict) -> dict:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "strict": True,
                    "schema": json_schema,
                },
            },
        )
        return json.loads(response.choices[0].message.content)

    def describe_image(self, image_png_bytes: bytes, prompt: str) -> str:
        if not self.vision_model:
            raise ValueError(
                _("No vision_model configured for this ChatProvider — describe_image() needs "
                  "one (equivalent to FAISS_INDEX_OCR_VLM_MODEL).")
            )
        b64_image = base64.b64encode(image_png_bytes).decode("utf-8")
        response = self._client.chat.completions.create(
            model=self.vision_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                ],
            }],
        )
        return response.choices[0].message.content


class CrossEncoderRerankProvider:
    """
    Reranks (query, candidate) pairs with a sentence-transformers CrossEncoder — a
    model trained to score a pair jointly, generally far more accurate than comparing
    independently-computed embeddings, at the cost of being run at query time over the
    candidate set (can't be precomputed like embeddings). Optional dependency: install
    with `pip install -e ".[rerank]"`.
    """

    def __init__(self, model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                _("CrossEncoderRerankProvider needs the 'sentence-transformers' package. "
                  "Install it with: pip install -e \".[rerank]\"")
            ) from e
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[str]) -> List[float]:
        return [float(score) for score in self._model.predict([(query, c) for c in candidates])]


class DoclingStructureProvider:
    """
    Extracts sections from a document's real structure (heading hierarchy, via
    Docling's layout-aware parsing) instead of the LLM-calibrated text-pattern
    matching in FaissDocumentIndex.extract_sections. No schema calibration needed —
    each document's own headings define its sections. Optional dependency: install
    with `pip install -e ".[docling]"`.

    Note: works on the original file (needs real layout, not already-flattened text),
    so it re-reads/re-parses the file independently of `read_document`/`utils_ocr.py` —
    when this provider is configured, each file is processed twice (once for
    full/chunks via the usual pipeline, once here for sections).

    `do_ocr` defaults to False: Docling's own OCR engine (RapidOCR, torch-based)
    segfaulted in testing (Python 3.14 + torch on macOS) — likely an environment/ABI
    issue, not a Docling bug, but disabling it by default avoids crashing the whole
    process for what's usually unnecessary anyway (a digitally-generated PDF already
    has embedded text; `utils_ocr.py`'s own OCR fallback already covers scanned pages
    for `full`/`chunks` regardless). A scanned page Docling can't read without OCR
    just won't contribute to `sections` — set `do_ocr=True` to use Docling's OCR
    instead, if your environment handles it fine.
    """

    _HEADING_LABEL_HINTS = ("section_header", "title")

    def __init__(self, do_ocr: bool = False):
        # faiss (already imported by the time this class is reachable — core.py
        # imports it unconditionally) and torch (pulled in by docling) each bundle
        # their own OpenMP runtime; loading both in one process segfaulted in testing
        # unless these are set before torch initializes. Setting them here (before the
        # docling/torch import below) is late enough to still take effect.
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.datamodel.base_models import InputFormat
        except ImportError as e:
            raise ImportError(
                _("DoclingStructureProvider needs the 'docling' package. "
                  "Install it with: pip install -e \".[docling]\"")
            ) from e
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = do_ocr
        self._converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

    def extract_sections(self, file_path: str) -> Dict[str, str]:
        doc = self._converter.convert(file_path).document

        sections: Dict[str, str] = {"completo": doc.export_to_text(), "cabecalho": ""}
        current_section = "cabecalho"

        for item, _level in doc.iterate_items():
            text = getattr(item, "text", None)
            if not text or not text.strip():
                continue
            label = str(getattr(item, "label", "")).lower()
            if any(hint in label for hint in self._HEADING_LABEL_HINTS):
                current_section = text.strip().lower().replace(" ", "_")
                sections.setdefault(current_section, "")
            else:
                sections[current_section] = sections.get(current_section, "") + text + "\n"

        return {k: v.strip() for k, v in sections.items() if v.strip()}
