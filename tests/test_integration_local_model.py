"""
Integration test against a real, live OpenAI-compatible server (local or otherwise) —
excluded from the default `pytest` run (see the `integration` marker in
pyproject.toml's `addopts`). Run explicitly with:

    pytest -m integration -v

Requires OPENAI_API_KEY/OPENAI_BASE_URL (via .env, picked up automatically) pointing
at a reachable server. Skips (doesn't fail) if the key is missing or the server isn't
reachable, so it's safe to run without special setup — you just won't get coverage.

The specific model names default to this project's own local dev setup (oMLX serving
jina-embeddings-v5-text-small-retrieval-mlx + Qwen3-14B-4bit) — override via
FAISS_INDEX_TEST_EMBEDDING_MODEL/FAISS_INDEX_TEST_EMBEDDING_DIM/FAISS_INDEX_TEST_CHAT_MODEL
if your local server serves different ones.
"""
import os

import openai
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def real_index(tmp_path_factory):
    from faiss_index import FaissDocumentIndex

    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set (see .env) — skipping local-model integration test")

    embedding_model = os.environ.get("FAISS_INDEX_TEST_EMBEDDING_MODEL", "jina-embeddings-v5-text-small-retrieval-mlx")
    embedding_dim = int(os.environ.get("FAISS_INDEX_TEST_EMBEDDING_DIM", "1024"))
    chat_model = os.environ.get("FAISS_INDEX_TEST_CHAT_MODEL", "Qwen3-14B-4bit")

    idx = FaissDocumentIndex(
        base_path=str(tmp_path_factory.mktemp("integration")),
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        section_extraction_model=chat_model,
    )
    try:
        idx.get_embeddings(["ping"])
    except openai.APIConnectionError:
        pytest.skip(f"Local model server not reachable at {os.environ.get('OPENAI_BASE_URL')}")
    return idx


def test_build_and_hybrid_search_against_real_local_server(real_index, tmp_path):
    idx = real_index
    document_type = "integration_doctype"
    data_dir = tmp_path / "data" / document_type
    data_dir.mkdir(parents=True)

    exact_term = "0001234-56.2024.8.14.0301"
    (data_dir / "doc1.txt").write_text(
        f"Contrato de prestacao de servicos entre as partes.\n"
        f"Processo numero {exact_term} em tramitacao.\n",
        encoding="utf-8",
    )
    (data_dir / "doc2.txt").write_text(
        "Relatorio financeiro anual da empresa, sem relacao com litigios.\n",
        encoding="utf-8",
    )

    idx.build_indices(
        document_type=document_type,
        base_data_dir=str(tmp_path / "data"),
        output_index_dir=str(tmp_path / "out"),
    )

    dense = idx.evaluate_strategy(exact_term, document_type, "full", k=2)
    assert dense["results"]

    hybrid = idx.evaluate_strategy_hybrid(exact_term, document_type, "full", k=2)
    assert hybrid["results"]
    assert hybrid["results"][0]["metadata"]["file"].endswith("doc1.txt")
