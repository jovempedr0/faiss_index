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
from typing import List, Optional, Protocol

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
