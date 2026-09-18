"""
Writes a labeled query set for `evaluate_retrieval`, from the chunks of an index that
was already built — one question per sampled chunk, written by the local chat model,
labeled with the file it came from and a passage out of the chunk's middle.

    python fixtures/eval/generate_labeled_queries.py [n_per_file] [output.json] [chunks_metadata.json]

The chunks argument is what makes an OCR A/B honest: questions written from one
extraction's text favour that extraction, so each index needs a set written from its
own chunks, and both sets get run against both indices (see measure_ocr_ab.py).

Kept out of git (see .gitignore): the questions quote real case documents.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "src")
from dotenv import load_dotenv

load_dotenv(str(Path(__file__).resolve().parents[2] / ".env"))

from faiss_index.providers import OpenAICompatibleChatProvider  # noqa: E402

DEFAULT_CHUNKS = "fixtures/faiss_index_output/casos/casos_chunks_metadata.json"
MODEL = "Qwen3-14B-4bit"
MIN_CHUNK_CHARS = 600
PASSAGE_WORDS = 30
QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {
            "type": "string",
            "description": (
                "A question in Portuguese that this excerpt answers and that someone "
                "searching a case archive would plausibly type. Do not quote the excerpt, "
                "do not mention 'the document' or 'the excerpt', and do not name the file."
            ),
        }
    },
    "required": ["question"],
    "additionalProperties": False,
}


def middle_passage(text: str, words: int = PASSAGE_WORDS) -> str:
    """A passage from the middle of the chunk — what `relevant_text` is checked against."""
    tokens = text.split()
    start = max(0, len(tokens) // 2 - words // 2)
    return " ".join(tokens[start:start + words])


def looks_copied(question: str, chunk: str, run_length: int = 12) -> bool:
    """True when the question lifts a long verbatim run from the chunk (BM25 would win for free)."""
    words = question.split()
    normalized = " ".join(chunk.split()).lower()
    return any(
        " ".join(words[i:i + run_length]).lower() in normalized
        for i in range(max(0, len(words) - run_length + 1))
    )


def main(n_per_file: int, output: str, chunks_path: str = DEFAULT_CHUNKS) -> None:
    chunks = json.load(open(chunks_path, encoding="utf-8"))
    usable = [c for c in chunks if len(c["chunk_text"]) >= MIN_CHUNK_CHARS]
    by_file: dict[str, list] = {}
    for chunk in usable:
        by_file.setdefault(chunk["file"], []).append(chunk)

    rng = random.Random(0)  # sampled deterministically, so the set can be rebuilt
    sampled = [c for file_chunks in by_file.values()
               for c in rng.sample(file_chunks, min(n_per_file, len(file_chunks)))]
    print(f"  {chunks_path}")
    print(f"  {len(by_file)} arquivos, {len(usable)} chunks utilizáveis -> {len(sampled)} amostrados")

    provider = OpenAICompatibleChatProvider(model=MODEL)
    labeled, discarded = [], 0
    for i, chunk in enumerate(sampled, start=1):
        prompt = (
            "Read this excerpt from a Brazilian court document and write ONE question, in "
            "Portuguese, that it answers. /no_think\n\n"
            f"Excerpt:\n{chunk['chunk_text'][:3000]}"
        )
        try:
            question = provider.complete_structured(prompt, QUESTION_SCHEMA)["question"].strip()
        except Exception as e:
            print(f"  [{i}/{len(sampled)}] falhou: {type(e).__name__}: {e}")
            discarded += 1
            continue

        if not 4 <= len(question.split()) <= 40 or looks_copied(question, chunk["chunk_text"]):
            discarded += 1
            continue

        labeled.append({
            "query": question,
            "relevant_files": [chunk["file"]],
            "relevant_text": middle_passage(chunk["chunk_text"]),
        })
        if i % 10 == 0:
            print(f"  [{i}/{len(sampled)}] {len(labeled)} mantidas, {discarded} descartadas")

    json.dump(labeled, open(output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n  {len(labeled)} perguntas gravadas em {output} ({discarded} descartadas)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3,
         sys.argv[2] if len(sys.argv) > 2 else "fixtures/eval/labeled_queries_casos.json",
         sys.argv[3] if len(sys.argv) > 3 else DEFAULT_CHUNKS)
