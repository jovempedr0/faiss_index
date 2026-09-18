"""
Sweeps the two numbers that decide what reranking costs and what it recovers: how many
passages of each candidate get scored (`constants.RERANK_PASSAGES_PER_CANDIDATE`) and how
many candidates retrieval hands over (`max(k * 4, 20)` in `_retrieve`). Both were chosen
by argument, not measurement.

The reranker can only reorder what it is given, so the pool's own recall is the ceiling.
Measured on the real corpus, hybrid, 64 questions — recall@10 / @20 / @40 is 98.4% /
98.4% / 100% for `full`, 87.5% / 98.4% / 100% for `sections`, 90.6% / 95.3% / 98.4% for
`chunks` — while reranking with the shipped settings lands at 89.1% for all three. The
gap between 89.1% and the ceiling is the cross-encoder's own accuracy, which is what more
passages per candidate might buy.

    python fixtures/eval/tune_rerank.py [strategy ...]

Cost is roughly (pool x passages) cross-encoder pairs per query, so the grid's far corner
is 16x the near one. Needs the local oMLX server up (embeddings); the cross-encoder runs
offline from the HF cache.
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

from faiss_index import FaissDocumentIndex, constants, providers  # noqa: E402

# Overridable for a narrower confirmation run: FAISS_TUNE_POOLS=10,20 FAISS_TUNE_PASSAGES=2,4
POOLS = [int(v) for v in os.environ.get("FAISS_TUNE_POOLS", "10,20,40").split(",")]
PASSAGES = [int(v) for v in os.environ.get("FAISS_TUNE_PASSAGES", "2,4,8").split(",")]
K = 5


class TunableIndex(FaissDocumentIndex):
    """`_retrieve` with the candidate pool as a knob instead of `max(k * 4, 20)`."""

    candidate_pool = 20

    def _retrieve(self, query, document_type, strategy, k, use_hybrid, rerank):
        evaluate = self.evaluate_strategy_hybrid if use_hybrid else self.evaluate_strategy
        retrieval_k = self.candidate_pool if rerank else k
        items = evaluate(query, document_type, strategy, k=retrieval_k).get("results", [])
        return self.rerank_results(query, items, k=k) if rerank else items


def main(strategies: list) -> None:
    labeled = json.load(open("fixtures/eval/labeled_queries_casos.json", encoding="utf-8"))
    idx = TunableIndex(
        base_path=".",
        embedding_model="jina-embeddings-v5-text-small-retrieval-mlx", embedding_dim=1024,
        section_extraction_model="Qwen3-14B-4bit",
        embedding_query_prefix="Query: ", embedding_document_prefix="Document: ",
        rerank_provider=providers.CrossEncoderRerankProvider(),
    )
    idx.load_indices(path_indices="fixtures/faiss_index_output", document_types=["casos"],
                     strategies=strategies, use_gpu=False)

    shipped = (constants.RERANK_PASSAGES_PER_CANDIDATE, 20)
    report = {}
    for strategy in strategies:
        print(f"\n  {strategy} — {len(labeled)} perguntas, híbrida, k={K}\n")
        print(f"  {'pool':>5} {'passagens':>10} {'pares/query':>12} {'recall@5':>9} "
              f"{'MRR':>7} {'passage@5':>10} {'s/query':>9}")
        for pool in POOLS:
            for passages in PASSAGES:
                idx.candidate_pool = pool
                constants.RERANK_PASSAGES_PER_CANDIDATE = passages
                start = time.perf_counter()
                s = idx.evaluate_retrieval(labeled, "casos", [strategy], k=K,
                                           use_hybrid=True, rerank=True)[strategy]
                s["seconds_per_query"] = (time.perf_counter() - start) / len(labeled)
                s["pool"], s["passages"] = pool, passages
                report[f"{strategy}|{pool}|{passages}"] = s
                mark = "  <- atual" if (passages, pool) == shipped else ""
                passage = (f"{s['passage_recall_at_k']:.1%}"
                           if s["passage_recall_at_k"] is not None else "-")
                print(f"  {pool:>5} {passages:>10} {pool * passages:>12} "
                      f"{s['recall_at_k']:>8.1%} {s['mrr']:>7.3f} {passage:>10} "
                      f"{s['seconds_per_query']:>9.2f}{mark}")
        constants.RERANK_PASSAGES_PER_CANDIDATE = shipped[0]

    out = os.environ.get("FAISS_TUNE_OUTPUT", "fixtures/eval/rerank_tuning.json")
    json.dump(report, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n  gravado em {out}")
    print(f"  n={len(labeled)}: diferenças de 2-3 pontos são ruído; o custo não é.")


if __name__ == "__main__":
    main(sys.argv[1:] or ["full"])
