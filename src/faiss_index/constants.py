"""
Internal constants for FaissDocumentIndex: fixed protocol/algorithm values that don't
vary by environment (unlike `config.py`, which handles environment/installation-
dependent configuration).
"""

# File extensions supported by `read_document`/`build_indices`.
SUPPORTED_FILE_EXTENSIONS = ('.txt', '.pdf', '.doc', '.docx')

# Embedding vector dimension per supported OpenAI model.
EMBEDDING_DIMENSIONS = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
}

# Language used to load the NLTK stopword list.
STOPWORDS_LANGUAGE = "portuguese"

# Encoding `read_document` falls back to for a .txt file that isn't valid UTF-8 and
# carries no byte-order mark — what Windows and older systems export Latin-script text
# as. Single-byte, so it decodes almost anything: it's a last resort, after the BOM and
# UTF-8, and using it is logged.
TEXT_FALLBACK_ENCODING = "cp1252"

# Max number of characters of a text sent to the embeddings API (get_embeddings).
MAX_EMBEDDING_INPUT_CHARS = 8000

# Default size (in words) of the windows used by create_embeddings_chunks.
DEFAULT_CHUNK_SIZE_WORDS = 500

# Section schema calibration via LLM (register_document_type).
MAX_SECTION_SAMPLE_DOCS = 5
MAX_SECTION_SAMPLE_CHARS = 4000
# Ceiling asked of the model in the calibration prompt. Each section is one vector per
# document, so a schema that splits the same text into more, narrower sections spreads a
# document's content thinner instead of describing it better.
MAX_SECTIONS_PER_SCHEMA = 8

# Reranking long texts. A cross-encoder reads a fixed window (512 tokens for the default
# model, roughly 350 words of Portuguese), so a candidate handed over whole is judged by
# its opening alone: reranking "full" documents that way took recall@5 from 79.7% to
# 31.2% on the real corpus. Long candidates are cut into these windows instead, and the
# candidate takes its best window's score. Scoring every window is what made the first
# attempt at this unaffordable (~1 min/query on "full"), so only the best
# RERANK_PASSAGES_PER_CANDIDATE by BM25 against the query are actually scored.
RERANK_PASSAGE_WORDS = 150
RERANK_PASSAGES_PER_CANDIDATE = 4

# Min number of training points per cluster when sizing nlist in IVFFlat/IVFSQ8/IVFPQ.
MIN_TRAINING_POINTS_PER_CLUSTER = 30

# Min number of training points for IVFSQ8's scalar quantizer, which learns each
# dimension's [min, max] range from them and clips any vector added later (e.g. via
# add_new_documents) to that range. Few points give ranges too narrow for later
# vectors: on real 1024-dim embeddings, training on 300 / 1000 / all 10.8k of them
# and then adding all 10.8k gave recall@10 98.2% / 98.9% / 99.6% (probing every list).
MIN_SQ8_TRAINING_POINTS = 1000

# JSON Schema (provider-agnostic — no OpenAI-specific envelope) describing the
# structured output requested in the chat call that infers the section schema of a
# document type (`_infer_section_schema_via_llm`, via `ChatProvider.complete_structured`).
SECTION_SCHEMA_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "Section name: one or two words in snake_case, lowercase letters and "
                            "underscores only. No slashes, no punctuation, no alternative names "
                            "(write 'header', never 'header_/_identifying_information')."
                        )
                    },
                    "patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Short words or phrases, in the document's language, that usually mark the start of the section."
                    }
                },
                "required": ["name", "patterns"],
                "additionalProperties": False
            }
        }
    },
    "required": ["sections"],
    "additionalProperties": False
}
