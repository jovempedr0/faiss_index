"""
Measures what reranking does to hybrid retrieval, per strategy, including its cost.

The cross-encoder reads 512 tokens, so on `full` (whole court documents, ~118k chars)
it only ever sees the opening. This script is how that was measured and how any fix
to it has to be measured: same queries, same index, quality *and* seconds per query.

    python fixtures/eval/measure_rerank.py [labeled_queries.json] [output.json]

Needs the local oMLX server up (embeddings) and an index in fixtures/faiss_index_output.
The cross-encoder runs offline from the HF cache.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
from dotenv import load_dotenv

load_dotenv(str(Path(__file__).resolve().parents[2] / ".env"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from faiss_index import FaissDocumentIndex, providers  # noqa: E402

STRATEGIES = ["full", "sections", "chunks"]


def main(queries_path: str, output: str) -> None:
    labeled = json.load(open(queries_path, encoding="utf-8"))
    idx = FaissDocumentIndex(
        base_path=".",
        embedding_model="jina-embeddings-v5-text-small-retrieval-mlx", embedding_dim=1024,
        section_extraction_model="Qwen3-14B-4bit",
        embedding_query_prefix="Query: ", embedding_document_prefix="Document: ",
        rerank_provider=providers.CrossEncoderRerankProvider(),
    )
    idx.load_indices(path_indices="fixtures/faiss_index_output", document_types=["casos"],
                     strategies=STRATEGIES, use_gpu=False)

    report = {}
    print(f"\n  {len(labeled)} perguntas rotuladas, busca híbrida, k=5\n")
    print(f"  {'estratégia':<10} {'rerank':<8} {'recall@5':>9} {'MRR':>7} {'passage@5':>10} {'s/query':>9}")
    for strategy in STRATEGIES:
        for rerank in (False, True):
            start = time.perf_counter()
            scores = idx.evaluate_retrieval(labeled, "casos", [strategy], k=5,
                                            use_hybrid=True, rerank=rerank)[strategy]
            scores["seconds_per_query"] = (time.perf_counter() - start) / len(labeled)
            report[f"{strategy}/{'rerank' if rerank else 'sem'}"] = scores
            passage = (f"{scores['passage_recall_at_k']:.1%}"
                       if scores["passage_recall_at_k"] is not None else "-")
            print(f"  {strategy:<10} {'sim' if rerank else 'não':<8} {scores['recall_at_k']:>8.1%} "
                  f"{scores['mrr']:>7.3f} {passage:>10} {scores['seconds_per_query']:>9.2f}")

    json.dump(report, open(output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n  gravado em {output}")
    print(f"  n={len(labeled)} é pequeno: diferenças de 2-3 pontos são ruído.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fixtures/eval/labeled_queries_casos.json",
         sys.argv[2] if len(sys.argv) > 2 else "fixtures/eval/rerank_report.json")
