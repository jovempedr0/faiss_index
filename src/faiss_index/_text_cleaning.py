"""
Text normalization used by FaissDocumentIndex's BM25 tokenizing (`_search.py`'s
`_tokenize_for_bm25`) and by `generate_search_by_type`'s empty-query check — not for
the text sent to embedding/rerank models, which read natural language (stopword
removal would drop words like "não"). Kept as plain functions in their own module (not methods
on the class) so `_search.py` can import them without a circular import back into
`core.py`.
"""
import re

from . import config


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
