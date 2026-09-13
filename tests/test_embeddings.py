import numpy as np


def test_get_embeddings_returns_one_vector_per_text(make_index):
    idx = make_index(embedding_dim=4)
    result = idx.get_embeddings(["primeiro texto", "segundo texto"])
    assert result.shape == (2, 4)


def test_get_embeddings_same_text_gives_same_vector(make_index):
    idx = make_index(embedding_dim=4)
    result = idx.get_embeddings(["repetido", "repetido"])
    assert np.array_equal(result[0], result[1])


def test_get_embeddings_batches_according_to_batch_size(make_index, fake_embedding_provider):
    idx = make_index(embedding_dim=4, embedding_batch_size=2)
    idx.get_embeddings(["a", "b", "c", "d", "e"])
    # 5 texts, batch size 2 -> batches of [2, 2, 1]
    assert fake_embedding_provider.embedding_calls == [2, 2, 1]


def test_get_embeddings_strips_null_bytes(make_index):
    idx = make_index(embedding_dim=4)
    # get_embeddings only strips NUL bytes explicitly (faiss_index.py's get_embeddings);
    # other control characters survive the utf-8 encode/decode round-trip unchanged.
    with_null_bytes = "texto\x00com\x00lixo"
    cleaned_equivalent = "textocomlixo"
    result = idx.get_embeddings([with_null_bytes])
    expected = idx.get_embeddings([cleaned_equivalent])
    assert np.array_equal(result, expected)


def test_get_embeddings_empty_text_becomes_zero_vector(make_index):
    idx = make_index(embedding_dim=4)
    result = idx.get_embeddings(["   ", "texto real"])
    assert np.array_equal(result[0], np.zeros(4))
    assert not np.array_equal(result[1], np.zeros(4))


def test_get_embeddings_truncates_long_input(make_index):
    idx = make_index(embedding_dim=4)
    from faiss_index import constants

    short = "x" * constants.MAX_EMBEDDING_INPUT_CHARS
    long = short + "extra que deveria ser cortado"
    result_short = idx.get_embeddings([short])
    result_long = idx.get_embeddings([long])
    assert np.array_equal(result_short, result_long)
