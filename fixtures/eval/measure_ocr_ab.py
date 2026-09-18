"""
Compares two OCR extractions of the same corpus without letting either one pick the
exam questions.

Questions written from an extraction's own text favour that extraction: they use its
spellings, its ligatures, its mistakes. So each index gets a question set generated
from its own chunks, and every set is run against both indices. A genuinely better
extraction wins on both sets; one that only wins on its own set won nothing.

`recall_at_k` and `mrr` are the fair numbers here — they ask which *file* came back,
and the file paths are identical in both indices. `passage_recall_at_k` is not
comparable across extractions: it checks a 30-word passage taken from one OCR's text
against the other OCR's text, so it measures how much the two agree, not retrieval.
It is printed only where the set and the index match.

    python fixtures/eval/measure_ocr_ab.py [output.json]

Needs the local oMLX server up (embeddings) and both indices built.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from dotenv import load_dotenv

load_dotenv(str(Path(__file__).resolve().parents[2] / ".env"))

from faiss_index import FaissDocumentIndex  # noqa: E402

STRATEGIES = ["full", "sections", "chunks"]
INDICES = {
    "GLM-OCR": "fixtures/faiss_index_output",
    "tesseract": "fixtures/faiss_index_output_tesseract",
}
QUERY_SETS = {
    "GLM-OCR": "fixtures/eval/labeled_queries_casos.json",
    "tesseract": "fixtures/eval/labeled_queries_casos_tesseract.json",
}


def load(path_indices: str) -> FaissDocumentIndex:
    idx = FaissDocumentIndex(
        base_path=".",
        embedding_model="jina-embeddings-v5-text-small-retrieval-mlx", embedding_dim=1024,
        section_extraction_model="Qwen3-14B-4bit",
        embedding_query_prefix="Query: ", embedding_document_prefix="Document: ",
    )
    idx.load_indices(path_indices=path_indices, document_types=["casos"],
                     strategies=STRATEGIES, use_gpu=False)
    return idx


def main(output: str) -> None:
    labeled = {name: json.load(open(path, encoding="utf-8")) for name, path in QUERY_SETS.items()}
    for name, queries in labeled.items():
        print(f"  perguntas escritas do corpus {name}: {len(queries)}")

    report = {}
    for index_name, path_indices in INDICES.items():
        idx = load(path_indices)
        for set_name, queries in labeled.items():
            scores = idx.evaluate_retrieval(queries, "casos", STRATEGIES, k=5, use_hybrid=True)
            for strategy, s in scores.items():
                report[f"{index_name}|{set_name}|{strategy}"] = s

    print(f"\n  busca híbrida, k=5. Cada coluna é um conjunto de perguntas; cada linha, um índice.")
    for strategy in STRATEGIES:
        print(f"\n  {strategy}")
        print(f"  {'índice':<12} " + " ".join(f"{'perguntas ' + s:>24}" for s in QUERY_SETS))
        for index_name in INDICES:
            cells = []
            for set_name in QUERY_SETS:
                s = report[f"{index_name}|{set_name}|{strategy}"]
                cell = f"{s['recall_at_k']:.1%} / MRR {s['mrr']:.3f}"
                if index_name == set_name and s["passage_recall_at_k"] is not None:
                    cell += f" / p{s['passage_recall_at_k']:.0%}"
                cells.append(f"{cell:>24}")
            print(f"  {index_name:<12} " + " ".join(cells))

    json.dump(report, open(output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n  gravado em {output}")
    print("  n pequeno dos dois lados: diferenças de 2-3 pontos são ruído.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fixtures/eval/ocr_ab_report.json")
