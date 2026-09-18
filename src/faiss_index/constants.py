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

# Max number of characters of a text sent to the embeddings API (get_embeddings).
MAX_EMBEDDING_INPUT_CHARS = 8000

# Default size (in words) of the windows used by create_embeddings_chunks.
DEFAULT_CHUNK_SIZE_WORDS = 500

# Section schema calibration via LLM (register_document_type).
MAX_SECTION_SAMPLE_DOCS = 5
MAX_SECTION_SAMPLE_CHARS = 4000

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
                        "description": "Short, descriptive section name, in snake_case."
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
