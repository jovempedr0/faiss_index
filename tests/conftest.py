import sys
from pathlib import Path

import nltk
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)


def _deterministic_vector(text: str, dim: int):
    seed = sum(ord(c) for c in text) or 1
    return [float((seed * (i + 1)) % 97) for i in range(dim)]


class FakeEmbeddingProvider:
    """Implements providers.EmbeddingProvider: deterministic, no network calls."""

    def __init__(self, dimension: int = 8):
        self.dimension = dimension
        self.embedding_calls = []  # list of batch sizes, one entry per embed() call

    def embed(self, texts):
        self.embedding_calls.append(len(texts))
        return [_deterministic_vector(text, self.dimension) for text in texts]


class FakeChatProvider:
    """Implements providers.ChatProvider: a FIFO queue of canned structured responses."""

    def __init__(self):
        self.responses = []

    def complete_structured(self, prompt, json_schema):
        if not self.responses:
            raise RuntimeError("No queued response for FakeChatProvider")
        return self.responses.pop(0)


class FakeRerankProvider:
    """Implements providers.RerankProvider: scores driven by a lookup dict (default 0.0)."""

    def __init__(self, scores_by_candidate: dict = None):
        self.scores_by_candidate = scores_by_candidate or {}
        self.calls = []  # list of (query, candidates) tuples, one per rerank() call

    def rerank(self, query, candidates):
        self.calls.append((query, candidates))
        return [self.scores_by_candidate.get(c, 0.0) for c in candidates]


class FakeStructureProvider:
    """
    Implements providers.StructureProvider: sections driven by a lookup dict keyed by
    file path (default: {} — no sections for a path not registered).
    """

    def __init__(self, sections_by_file: dict = None):
        self.sections_by_file = sections_by_file or {}
        self.calls = []  # list of file_path values, one per extract_sections() call

    def extract_sections(self, file_path):
        self.calls.append(file_path)
        return self.sections_by_file.get(file_path, {})


@pytest.fixture
def fake_embedding_provider():
    return FakeEmbeddingProvider()


@pytest.fixture
def fake_chat_provider():
    return FakeChatProvider()


@pytest.fixture
def fake_rerank_provider():
    return FakeRerankProvider()


@pytest.fixture
def fake_structure_provider():
    return FakeStructureProvider()


@pytest.fixture
def make_index(fake_embedding_provider, fake_chat_provider, tmp_path):
    from faiss_index import FaissDocumentIndex

    def _make(embedding_dim: int = 8, **kwargs):
        fake_embedding_provider.dimension = embedding_dim
        return FaissDocumentIndex(
            base_path=str(tmp_path),
            embedding_provider=fake_embedding_provider,
            chat_provider=fake_chat_provider,
            **kwargs,
        )

    return _make
