"""
Text normalization used by FaissDocumentIndex's BM25 tokenizing (`_search.py`'s
`_tokenize_for_bm25`) and by `generate_search_by_type`'s empty-query check — not for
the text sent to embedding/rerank models, which read natural language (stopword
removal would drop words like "não") — plus `word_windows`, the word-window split
both indexing and reranking cut long text with. Kept as plain functions in their own module (not methods
on the class) so `_search.py` can import them without a circular import back into
`core.py`.
"""
import re
import unicodedata

from . import config


def _fold_accents(text: str) -> str:
    """
    Strips diacritics ("não" -> "nao", "execução" -> "execucao"): NFKD-decomposes the
    text and drops the combining marks. The compatibility decomposition also splits
    ligatures PDF text extraction often leaves behind ("ﬁ" -> "fi").
    """
    return "".join(char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char))


# config.STOPWORDS_PT with accents folded, to filter tokens that were folded too — NLTK's
# list spells "não"/"é"/"já"/"você" with accents, so the folded "nao"/"e"/"ja"/"voce"
# would otherwise slip through as content words.
_FOLDED_STOPWORDS_PT = frozenset(_fold_accents(word) for word in config.STOPWORDS_PT)


def tokenize_accent_folded(text: str) -> list[str]:
    """
    `clean_text`'s tokens with accents folded (and stopwords removed in their folded
    form as well) — so a text typed with or without accents yields the same tokens.
    """
    return [token for token in clean_text(_fold_accents(text))[0].split() if token not in _FOLDED_STOPWORDS_PT]


def remove_stop_words(sentence:str) -> str:
	"""
	Remove Portuguese stop words from a sentence.

	Parameters:
		sentence (str): The sentence to remove stop words from.

	Returns:
		str: The sentence without stop words.
	"""
	words = sentence.split()

	filtered_words = [word for word in words if word not in config.STOPWORDS_PT]

	return ' '.join(filtered_words)


def clean_text(text:str) -> list[str]:
    """
    Cleans the text by removing punctuation, extra whitespace, and lowercasing it.

    Parameters:
        text (str): The text to clean.

    Returns:
        str: The cleaned text, without punctuation or extra whitespace, and lowercased.
    """
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)     # Remove extra whitespace
    text = text.lower()                  # Lowercase
    text = remove_stop_words(text)  # Uses the .text attribute to get the string
    return [text.strip()]


def word_windows(text: str, chunk_size: int) -> list[str]:
    """
    Splits `text` into windows of `chunk_size` words overlapping by half, the last one
    reaching the end of the text — the "chunks" strategy's windows, also pooled into
    each section's vector (`_embed_pooled_windows`) and used to cut a long candidate
    into passages a cross-encoder can read whole (`_passages_for_rerank`).
    """
    words = text.split()
    windows = []
    for start in range(0, len(words), chunk_size // 2):
        windows.append(' '.join(words[start:start + chunk_size]))
        if start + chunk_size >= len(words):
            # This window already reaches the end of the document: any later
            # start would only produce a window entirely contained in this one.
            break
    return windows
