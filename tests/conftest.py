import json
import sys
from pathlib import Path

import nltk
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)


def _deterministic_vector(text: str, dim: int):
    seed = sum(ord(c) for c in text) or 1
    return [float((seed * (i + 1)) % 97) for i in range(dim)]


class _FakeEmbeddingData:
    def __init__(self, embedding, index):
        self.embedding = embedding
        self.index = index


class _FakeEmbeddingResponse:
    def __init__(self, data):
        self.data = data


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeChatResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeEmbeddings:
    def __init__(self, outer):
        self._outer = outer

    def create(self, input, model):
        self._outer.embedding_calls.append(len(input))
        data = [
            _FakeEmbeddingData(_deterministic_vector(text, self._outer.embedding_dim), i)
            for i, text in enumerate(input)
        ]
        return _FakeEmbeddingResponse(data)


class _FakeCompletions:
    def __init__(self, outer):
        self._outer = outer

    def create(self, model, messages, response_format=None):
        if not self._outer.chat_responses:
            raise RuntimeError("No queued chat response for FakeOpenAIClient")
        content = self._outer.chat_responses.pop(0)
        return _FakeChatResponse(content)


class _FakeChat:
    def __init__(self, outer):
        self.completions = _FakeCompletions(outer)


class FakeOpenAIClient:
    """
    Stand-in for openai.OpenAI: deterministic embeddings (same text -> same
    vector, distinct texts -> distinct vectors), and a FIFO queue of canned
    chat-completion responses for structured-output calls (section-schema
    calibration).
    """

    def __init__(self, embedding_dim: int = 8):
        self.embedding_dim = embedding_dim
        self.embedding_calls = []  # list of batch sizes, one entry per API call
        self.chat_responses = []
        self.embeddings = _FakeEmbeddings(self)
        self.chat = _FakeChat(self)

    def queue_chat_response(self, schema_dict: dict):
        self.chat_responses.append(json.dumps(schema_dict))


@pytest.fixture
def fake_openai_client(monkeypatch):
    client = FakeOpenAIClient()
    monkeypatch.setattr("faiss_index.openai.OpenAI", lambda *a, **kw: client)
    return client


@pytest.fixture
def make_index(fake_openai_client, tmp_path):
    from faiss_index import FaissDocumentIndex

    def _make(embedding_dim: int = 8, **kwargs):
        fake_openai_client.embedding_dim = embedding_dim
        kwargs.setdefault("use_mps", False)
        return FaissDocumentIndex(
            base_path=str(tmp_path),
            openai_key="test-key",
            embedding_model="fake-embedding-model",
            embedding_dim=embedding_dim,
            **kwargs,
        )

    return _make
