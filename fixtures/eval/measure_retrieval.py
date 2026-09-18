"""
Measures retrieval on the real corpus with `evaluate_retrieval`, for every strategy,
dense and hybrid — the numbers the docs quote.

    python fixtures/eval/measure_retrieval.py [labeled_queries.json] [output.json]

Needs the local oMLX server up (embeddings) and an index in fixtures/faiss_index_output.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from dotenv import load_dotenv

load_dotenv(str(Path(__file__).resolve().parents[2] / ".env"))

from faiss_index import FaissDocumentIndex  # noqa: E402

STRATEGIES = ["full", "sections", "chunks"]


def main(queries_path: str, output: str) -> None:
    labeled = json.load(open(queries_path, encoding="utf-8"))
    idx = FaissDocumentIndex(
        base_path=".",
        embedding_model="jina-embeddings-v5-text-small-retrieval-mlx", embedding_dim=1024,
        section_extraction_model="Qwen3-14B-4bit",
        embedding_query_prefix="Query: ", embedding_document_prefix="Document: ",
    )
    idx.load_indices(path_indices="fixtures/faiss_index_output", document_types=["casos"],
                     strategies=STRATEGIES, use_gpu=False)

    report = {}
    print(f"  {len(labeled)} perguntas rotuladas\n")
    print(f"  {'estratégia':<10} {'busca':<8} {'recall@5':>9} {'MRR':>7} {'passage@5':>10} {'chars/result':>13}")
    for use_hybrid in (False, True):
        scores = idx.evaluate_retrieval(labeled, "casos", STRATEGIES, k=5, use_hybrid=use_hybrid)
        for strategy, s in scores.items():
            label = "híbrida" if use_hybrid else "densa"
            report[f"{strategy}/{label}"] = s
            passage = f"{s['passage_recall_at_k']:.1%}" if s["passage_recall_at_k"] is not None else "-"
            print(f"  {strategy:<10} {label:<8} {s['recall_at_k']:>8.1%} {s['mrr']:>7.3f} {passage:>10} {s['avg_result_chars']:>13,.0f}")

    json.dump(report, open(output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n  gravado em {output}")
    print(f"  n={len(labeled)} é pequeno: diferenças de 2-3 pontos são ruído.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fixtures/eval/labeled_queries_casos.json",
         sys.argv[2] if len(sys.argv) > 2 else "fixtures/eval/retrieval_report.json")
