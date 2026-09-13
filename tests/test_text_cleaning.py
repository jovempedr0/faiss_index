from faiss_index import clean_text, remove_stop_words


def test_remove_stop_words_strips_portuguese_stopwords():
    result = remove_stop_words("o gato de a casa")
    assert "gato" in result
    assert "casa" in result
    assert " de " not in f" {result} "
    assert result.split() == [w for w in result.split() if w not in ("o", "de", "a")]


def test_clean_text_removes_punctuation_and_lowercases():
    result = clean_text("Olá, MUNDO!!  Tudo bem?")
    assert isinstance(result, list)
    assert len(result) == 1
    cleaned = result[0]
    assert cleaned == cleaned.lower()
    assert "," not in cleaned
    assert "!" not in cleaned
    assert "?" not in cleaned
    assert "  " not in cleaned


def test_clean_text_collapses_whitespace():
    result = clean_text("muitos    espaços\n\naqui")[0]
    assert "  " not in result
    assert "\n" not in result
