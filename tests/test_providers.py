"""The OpenAI-compatible providers' own behaviour, with the SDK client stubbed out."""
from types import SimpleNamespace

from faiss_index.providers import OpenAICompatibleChatProvider


class _FakeCompletions:
    """Stands in for client.chat.completions: records kwargs, returns canned JSON."""

    def __init__(self, content='{"sections": []}'):
        self.calls = []
        self.content = content

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))])


def _provider_with_fake_client(**kwargs):
    provider = OpenAICompatibleChatProvider(model="modelo", api_key="x", **kwargs)
    completions = _FakeCompletions()
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return provider, completions


def test_complete_structured_asks_for_deterministic_output_by_default():
    # Regression test: no temperature was sent at all, so the server's default (sampling)
    # applied — two calibrations over the same corpus returned different section schemas,
    # which meant the same documents were cut differently depending on the day.
    provider, completions = _provider_with_fake_client()

    provider.complete_structured("prompt", {"type": "object"})

    assert completions.calls[0]["temperature"] == 0.0


def test_complete_structured_honors_an_explicit_temperature():
    provider, completions = _provider_with_fake_client(temperature=0.7)

    provider.complete_structured("prompt", {"type": "object"})

    assert completions.calls[0]["temperature"] == 0.7


def test_complete_structured_still_requests_the_schema_and_parses_the_response():
    provider, completions = _provider_with_fake_client()
    schema = {"type": "object", "properties": {}}

    parsed = provider.complete_structured("prompt", schema)

    assert parsed == {"sections": []}
    request = completions.calls[0]
    assert request["model"] == "modelo"
    assert request["messages"] == [{"role": "user", "content": "prompt"}]
    assert request["response_format"]["json_schema"]["schema"] is schema
    assert request["response_format"]["json_schema"]["strict"] is True
