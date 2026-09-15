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


def test_create_embeddings_full_is_the_normalized_mean_of_its_normalized_chunks(make_index):
    idx = make_index(embedding_dim=4)
    docs = [("doc.txt", " ".join(f"palavra{i}" for i in range(1000)))]  # 3 chunks (0, 250, 500)
    chunk_embeddings, chunk_metadata = idx.create_embeddings_chunks(docs)
    assert len(chunk_metadata) == 3

    full_embeddings, _metadata = idx.create_embeddings_full(docs)

    # The fake provider's vectors aren't unit-length, so both normalizations matter here.
    unit_chunks = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    expected = unit_chunks.mean(axis=0)
    expected /= np.linalg.norm(expected)
    assert np.allclose(full_embeddings[0], expected)
    assert np.isclose(np.linalg.norm(full_embeddings[0]), 1.0)


def test_create_embeddings_chunks_has_no_window_contained_in_the_previous_one(make_index):
    # Regression test: windows started every chunk_size // 2 words all the way to the end,
    # so each document's last window (or two, for odd chunk sizes) sat entirely inside the
    # previous one — a wasted embedding and a duplicate hit, on every document.
    idx = make_index(embedding_dim=4)
    for n_words, chunk_size, expected_windows in [
        (1000, 500, [(0, 500), (250, 750), (500, 1000)]),
        (600, 500, [(0, 500), (250, 600)]),
        (300, 500, [(0, 300)]),
        (12, 5, [(0, 5), (2, 7), (4, 9), (6, 11), (8, 12)]),
    ]:
        words = [f"w{i}" for i in range(n_words)]

        _, metadata = idx.create_embeddings_chunks([("doc.txt", " ".join(words))], chunk_size=chunk_size)

        assert [m["chunk_text"] for m in metadata] == [" ".join(words[a:b]) for a, b in expected_windows]
        assert [m["chunk_index"] for m in metadata] == list(range(len(expected_windows)))


def test_create_embeddings_full_covers_text_past_the_embedding_input_limit(make_index):
    # Regression test: "full" used to embed the whole document in one call, truncated to
    # MAX_EMBEDDING_INPUT_CHARS — two documents identical up to that point got the exact
    # same vector no matter what followed (on a real corpus, 20 of 23 documents were
    # longer than the limit, so only 8.4% of the text was represented at all).
    from faiss_index import constants

    idx = make_index(embedding_dim=4)
    shared_prefix = "prefixo " * 1100
    assert len(shared_prefix) > constants.MAX_EMBEDDING_INPUT_CHARS
    docs = [("a.txt", shared_prefix + "gato preto " * 150), ("b.txt", shared_prefix + "cachorro branco " * 150)]

    full_embeddings, _metadata = idx.create_embeddings_full(docs)

    assert not np.allclose(full_embeddings[0], full_embeddings[1])


def test_create_embeddings_full_reuses_given_chunks_without_embedding_again(make_index, fake_embedding_provider):
    idx = make_index(embedding_dim=4)
    docs = [("a.txt", "primeiro documento"), ("b.txt", "segundo documento")]
    chunks = idx.create_embeddings_chunks(docs)
    calls_before = list(fake_embedding_provider.embedding_calls)

    reused, _ = idx.create_embeddings_full(docs, chunks=chunks)

    assert fake_embedding_provider.embedding_calls == calls_before
    assert np.allclose(reused, idx.create_embeddings_full(docs)[0])


def test_create_embeddings_full_document_without_words_gets_zero_vector(make_index):
    idx = make_index(embedding_dim=4)

    full_embeddings, metadata = idx.create_embeddings_full([("vazio.txt", "   "), ("doc.txt", "texto real")])

    assert full_embeddings.shape == (2, 4)
    assert np.array_equal(full_embeddings[0], np.zeros(4))
    assert np.isclose(np.linalg.norm(full_embeddings[1]), 1.0)
    assert [m["file"] for m in metadata] == ["vazio.txt", "doc.txt"]


def test_get_embeddings_prefix_is_prepended_to_non_empty_texts_only(make_index):
    idx = make_index(embedding_dim=4)

    prefixed = idx.get_embeddings(["texto real", "   "], prefix="Query: ")

    assert np.array_equal(prefixed[0], idx.get_embeddings(["Query: texto real"])[0])
    assert np.array_equal(prefixed[1], np.zeros(4))  # still empty -> still a zero vector


def test_get_embeddings_truncates_long_input(make_index):
    idx = make_index(embedding_dim=4)
    from faiss_index import constants

    short = "x" * constants.MAX_EMBEDDING_INPUT_CHARS
    long = short + "extra que deveria ser cortado"
    result_short = idx.get_embeddings([short])
    result_long = idx.get_embeddings([long])
    assert np.array_equal(result_short, result_long)
