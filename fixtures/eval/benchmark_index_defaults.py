"""
Benchmarks the index defaults (`auto_index_thresholds` and `ivf_nprobe`) on corpora
larger than the real fixture one, which only has ~1.3k chunks.

The corpus is synthesized from the real chunk embeddings instead of drawn at random:
random vectors have no cluster structure, and cluster structure is exactly what decides
IVF recall, so a random corpus would make IVF look worse than it is. Each synthetic
vector is an interpolation between a real vector and one of its nearest real neighbours,
plus noise scaled by the real mean nearest-neighbour distance, renormalized (the real
embeddings are L2-normalized). Queries are real vectors held out of the corpus seeds.

Recall is measured against exact search on the same corpus, so it is self-consistent;
what it does not capture is the greater topical diversity a real 500k-document corpus
would have.

    python fixtures/eval/benchmark_index_defaults.py [output.json]

Takes a few minutes and a few GB of RAM at the largest size.
"""
import gc
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import faiss
import numpy as np

sys.path.insert(0, "src")
from faiss_index import constants  # noqa: E402

EMBEDDINGS = "fixtures/faiss_index_output/casos/casos_chunks_embeddings.npy"
DEFAULT_OUTPUT = "fixtures/eval/index_defaults_report.json"

# Both overridable for sensitivity runs: FAISS_BENCH_SIZES=100000 FAISS_BENCH_NOISE=1.5
CORPUS_SIZES = [int(s) for s in os.environ.get(
    "FAISS_BENCH_SIZES", "50000,100000,250000,500000").split(",")]
NPROBES = [1, 8, 16, 32, 64, 128]
N_QUERIES = 200
K = 10
NEIGHBOURS = 10
# How far synthetic vectors wander from the real pair they interpolate, as a fraction of
# the real mean nearest-neighbour distance. Lower makes tighter (easier) clusters.
NOISE_FRACTION = float(os.environ.get("FAISS_BENCH_NOISE", "0.5"))
SEED = 20260918


def synthesize(seeds: np.ndarray, n_target: int, rng: np.random.Generator) -> Tuple[np.ndarray, float]:
    """
    Grows `seeds` into `n_target` vectors by interpolating each seed towards one of its
    nearest neighbours and adding calibrated noise. Keeps the seeds themselves.
    """
    dim = seeds.shape[1]
    neighbour_index = faiss.IndexFlatL2(dim)
    neighbour_index.add(seeds)
    distances, neighbours = neighbour_index.search(seeds, NEIGHBOURS + 1)
    # Column 0 is the vector itself.
    mean_nn_distance = float(np.sqrt(distances[:, 1:]).mean())
    sigma = NOISE_FRACTION * mean_nn_distance / np.sqrt(dim)

    n_new = n_target - len(seeds)
    picks = rng.integers(0, len(seeds), size=n_new)
    partners = neighbours[picks, rng.integers(1, NEIGHBOURS + 1, size=n_new)]
    alpha = rng.random((n_new, 1), dtype=np.float32)
    new = (1.0 - alpha) * seeds[picks] + alpha * seeds[partners]
    new += rng.normal(0.0, sigma, size=new.shape).astype(np.float32)
    new /= np.linalg.norm(new, axis=1, keepdims=True)

    corpus = np.vstack([seeds, new]).astype(np.float32)
    rng.shuffle(corpus)
    return corpus, mean_nn_distance


def nlist_for(n: int) -> int:
    """Mirrors FaissIndexBackendMixin._compute_nlist with ivf_nlist unset."""
    desired = max(1, int(np.sqrt(n)))
    max_supported = max(1, n // constants.MIN_TRAINING_POINTS_PER_CLUSTER)
    return max(1, min(desired, max_supported))


def measure_latency(index: faiss.Index, queries: np.ndarray) -> float:
    """Mean ms per single-query search — the way the library searches, one query at a time."""
    start = time.perf_counter()
    for query in queries:
        index.search(query.reshape(1, -1), K)
    return (time.perf_counter() - start) * 1000.0 / len(queries)


def recall_against(index: faiss.Index, queries: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    _, found = index.search(queries, K)
    hits = sum(len(set(f) & set(t)) for f, t in zip(found, truth))
    top1 = float(np.mean(found[:, 0] == truth[:, 0]))
    return {"recall_at_10": hits / (len(queries) * K), "top1": top1}


def bytes_per_vector(kind: str, dim: int) -> int:
    """Payload bytes per vector, ignoring the (size-independent) centroid table."""
    return dim * 4 + 8 if kind == "ivf_flat" else dim + 8


def build(kind: str, corpus: np.ndarray, nlist: int) -> faiss.Index:
    dim = corpus.shape[1]
    quantizer = faiss.IndexFlatL2(dim)
    if kind == "ivf_flat":
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)
    else:
        index = faiss.IndexIVFScalarQuantizer(
            quantizer, dim, nlist, faiss.ScalarQuantizer.QT_8bit
        )
    index.train(corpus)
    index.add(corpus)
    return index


def main() -> None:
    output = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT
    rng = np.random.default_rng(SEED)

    real = np.load(EMBEDDINGS).astype(np.float32)
    order = rng.permutation(len(real))
    queries = np.ascontiguousarray(real[order[:N_QUERIES]])
    seeds = np.ascontiguousarray(real[order[N_QUERIES:]])
    dim = real.shape[1]
    print(f"{len(real)} real vectors, dim {dim}: {len(seeds)} seeds, {len(queries)} held-out queries")

    results: List[dict] = []
    for size in CORPUS_SIZES:
        corpus, mean_nn = synthesize(seeds, size, rng)
        nlist = nlist_for(size)
        print(f"\n=== {size:,} vectors (nlist={nlist}, seed mean NN distance {mean_nn:.4f}) ===")

        flat = faiss.IndexFlatL2(dim)
        flat.add(corpus)
        _, truth = flat.search(queries, K)
        flat_ms = measure_latency(flat, queries)
        print(f"flat (exact): {flat_ms:.2f} ms/query, {bytes_per_vector('ivf_flat', dim) * size / 2**20:.0f} MiB")
        results.append({"size": size, "kind": "flat", "nprobe": None, "recall_at_10": 1.0,
                        "top1": 1.0, "ms_per_query": flat_ms,
                        "mib": (dim * 4) * size / 2**20})
        del flat
        gc.collect()

        for kind in ("ivf_flat", "ivf_sq8"):
            index = build(kind, corpus, nlist)
            for nprobe in NPROBES:
                if nprobe > nlist:
                    continue
                index.nprobe = nprobe
                scores = recall_against(index, queries, truth)
                ms = measure_latency(index, queries)
                print(f"{kind:9s} nprobe={nprobe:4d}: recall@10 {scores['recall_at_10']:.3f} "
                      f"top1 {scores['top1']:.3f} {ms:.2f} ms/query")
                results.append({"size": size, "kind": kind, "nprobe": nprobe,
                                "ms_per_query": ms,
                                "mib": bytes_per_vector(kind, dim) * size / 2**20, **scores})
            del index
            gc.collect()

        del corpus
        gc.collect()

    with open(output, "w", encoding="utf-8") as handle:
        json.dump({"seed": SEED, "dim": dim, "n_queries": N_QUERIES, "k": K,
                   "noise_fraction": NOISE_FRACTION, "results": results}, handle, indent=2)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
