"""
Mixin for FaissDocumentIndex: dense (FAISS) and hybrid (dense + BM25 via Reciprocal
Rank Fusion) search, reranking, strategy comparison/scoring, and the high-level
`generate_search`/`generate_search_by_type` shortcuts. Composed into the class in
`core.py`; not meant to be imported directly by users.
"""
import logging
import time
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

from . import config
from ._text_cleaning import clean_text, tokenize_accent_folded
from .i18n import _

logger = logging.getLogger(__name__)


class SearchMixin:

    def _tokenize_for_bm25(self, text: str) -> List[str]:
        """Tokenizes text for BM25 indexing/querying via clean_text (lowercase,
        punctuation stripped, PT stopwords removed), with accents folded on both the
        corpus and the query — users often type Portuguese without them, and "execucao"
        must match "execução" (on a real legal corpus, accentless queries' BM25-only
        passage@5 went from 61% to 84%, the same as with accents). Only the lexical
        side normalizes like this — dense search and reranking receive the query
        unaltered."""
        return tokenize_accent_folded(text)

    def _get_bm25_index(self, document_type: str, strategy: str) -> BM25Okapi:
        """
        Lazily builds (and caches on the instance) a BM25Okapi index over the metadata
        already loaded for `document_type`/`strategy` — no separate on-disk format, no
        LLM/embedding calls: just tokenizing text that's already in memory.
        """
        cached = self._bm25_indices.get(document_type, {}).get(strategy)
        if cached is not None:
            return cached

        _, metadata, _embeddings = self.indices[document_type][strategy]
        corpus = [self._tokenize_for_bm25(self._extract_metadata_text(m)) for m in metadata]
        bm25 = BM25Okapi(corpus)

        self._bm25_indices.setdefault(document_type, {})[strategy] = bm25
        return bm25

    def evaluate_strategy(self, query: str, document_type: str, strategy: str, k: int = 10) -> Dict:
        """
        Evaluates a search strategy on a FAISS index for a specific document type.

        Parameters:
            query (str): Query text for similarity search.
            document_type (str): Document type to select the corresponding index.
            strategy (str): Strategy used for indexing and search.
            k (int, optional): Number of nearest neighbors to return. Defaults to 10.

        Returns:
            Dict: A dictionary with:
                - "strategy": Strategy used.
                - "search_time": Time spent searching (in seconds).
                - "results": List of results, each with:
                    - "rank": Result position.
                    - "distance": FAISS distance of the result.
                    - "metadata": Metadata associated with the result.
                    - "similarity": Cosine similarity between the query and the result.
                - "avg_distance": Average distance of the returned results.
                - "std_distance": Standard deviation of the results' distances.
        Returns an empty dict if the document type doesn't exist in the indices.
        """
        if document_type not in self.indices or strategy not in self.indices[document_type]:
            return {}

        index, metadata, _embeddings = self.indices[document_type][strategy]
        query_embedding = self.get_embeddings([query], prefix=self.embedding_query_prefix)

        start_time = time.time()
        valid_indices, valid_distances = self._dense_search_indices(index, query_embedding, k)
        search_time = time.time() - start_time

        results = []
        for dist, idx in zip(valid_distances, valid_indices):
            # Reconstructs the vector from the FAISS index itself (instead of keeping a
            # separate embeddings array in memory) — works the same for "flat" and "ivf_*"
            # (index.make_direct_map() is called when building/loading the index).
            stored_vector = np.asarray(index.reconstruct(int(idx))).reshape(1, -1)
            results.append({
                "rank": len(results) + 1,
                "distance": float(dist),
                "doc_index": int(idx),
                "metadata": metadata[idx],
                "similarity": float(cosine_similarity(query_embedding, stored_vector)[0][0])
            })

        return {
            "strategy": strategy,
            "search_time": search_time,
            "results": results,
            "avg_distance": float(np.mean(valid_distances)) if len(valid_distances) else 0.0,
            "std_distance": float(np.std(valid_distances)) if len(valid_distances) else 0.0
        }

    def evaluate_strategy_hybrid(
        self, query: str, document_type: str, strategy: str, k: int = 10,
        candidate_pool: Optional[int] = None, rrf_k: int = 60
    ) -> Dict:
        """
        Hybrid search: fuses dense (FAISS embeddings) and lexical (BM25) rankings via
        Reciprocal Rank Fusion — helps with exact terms (names, case/process numbers)
        that a purely dense search can miss, without needing to normalize/weigh two
        incompatible score scales (L2 distance vs. BM25 score).

        Parameters:
            query (str): Query text.
            document_type (str): Document type to select the corresponding index.
            strategy (str): Strategy used for indexing and search.
            k (int, optional): Number of fused results to return. Defaults to 10.
            candidate_pool (Optional[int]): How many candidates each of the dense/BM25
                searches contributes to the fusion before cutting to `k`. Defaults to
                max(k * 4, 50), capped at the corpus size.
            rrf_k (int): RRF's smoothing constant (higher = flatter weighting of rank
                position). Defaults to 60, the commonly used value.

        Returns:
            Dict: Same overall shape as `evaluate_strategy`, with "results" entries
            carrying "rrf_score" (used for ranking) plus "dense_rank"/"bm25_rank"
            (whichever lists the document appeared in) instead of "distance"/
            "similarity". Returns an empty dict if the document type/strategy doesn't
            exist in the indices.
        """
        if document_type not in self.indices or strategy not in self.indices[document_type]:
            return {}

        index, metadata, _embeddings = self.indices[document_type][strategy]
        pool = candidate_pool or max(k * 4, 50)
        pool = min(pool, index.ntotal)

        start_time = time.time()

        query_embedding = self.get_embeddings([query], prefix=self.embedding_query_prefix)
        dense_indices, _dense_distances = self._dense_search_indices(index, query_embedding, pool)

        bm25 = self._get_bm25_index(document_type, strategy)
        bm25_scores = bm25.get_scores(self._tokenize_for_bm25(query))
        ranked_bm25 = np.argsort(bm25_scores)[::-1]
        # Drop documents that share no term with the query (score 0) instead of letting
        # them fill up the pool: in a large corpus most documents score 0, and without
        # this filter they'd still receive a bm25_rank/RRF contribution just for landing
        # in an arbitrary (non-stable-sorted) slice of the tie — diluting the fusion's
        # whole point of surfacing genuine lexical matches.
        bm25_indices = ranked_bm25[bm25_scores[ranked_bm25] > 0][:pool]

        dense_ranks = {int(doc_idx): rank for rank, doc_idx in enumerate(dense_indices)}
        bm25_ranks = {int(doc_idx): rank for rank, doc_idx in enumerate(bm25_indices)}

        rrf_scores: Dict[int, float] = defaultdict(float)
        for doc_idx, rank in dense_ranks.items():
            rrf_scores[doc_idx] += 1.0 / (rrf_k + rank + 1)
        for doc_idx, rank in bm25_ranks.items():
            rrf_scores[doc_idx] += 1.0 / (rrf_k + rank + 1)

        search_time = time.time() - start_time

        ranked_doc_indices = sorted(rrf_scores, key=lambda doc_idx: rrf_scores[doc_idx], reverse=True)[:k]

        results = []
        for doc_idx in ranked_doc_indices:
            results.append({
                "rank": len(results) + 1,
                "rrf_score": rrf_scores[doc_idx],
                "doc_index": doc_idx,
                "metadata": metadata[doc_idx],
                "dense_rank": dense_ranks.get(doc_idx),
                "bm25_rank": bm25_ranks.get(doc_idx),
            })

        return {
            "strategy": strategy,
            "search_time": search_time,
            "results": results,
        }

    def rerank_results(self, query: str, results: List[Dict], k: Optional[int] = None) -> List[Dict]:
        """
        Reorders a search's results with `self.rerank_provider` — a cross-encoder (or
        any other model that scores a query/candidate pair jointly), generally more
        accurate than the embedding similarity used to retrieve them in the first
        place. The classic retrieve-then-rerank pattern: retrieve a larger candidate
        pool cheaply (dense and/or BM25), then spend the more expensive per-pair
        scoring only on those candidates.

        Composable with the "results" list from either `evaluate_strategy` or
        `evaluate_strategy_hybrid` — this method only needs each item's "metadata".

        Parameters:
            query (str): The same query the results were retrieved for.
            results (List[Dict]): A "results" list from `evaluate_strategy`/
                `evaluate_strategy_hybrid`.
            k (Optional[int]): How many reranked results to keep. Defaults to all of them.

        Returns:
            List[Dict]: The input results, reordered by `rerank_score` (descending)
            and truncated to `k`, each with "rank" recomputed and a new "rerank_score" key.

        Raises:
            ValueError: If no `rerank_provider` was configured on the constructor.
        """
        if self.rerank_provider is None:
            raise ValueError(
                "No rerank_provider configured. Pass one to the constructor (e.g. "
                "providers.CrossEncoderRerankProvider(), needs the optional "
                "'sentence-transformers' dependency: pip install -e \".[rerank]\")."
            )

        if not results:
            return []

        candidates = [self._extract_metadata_text(r["metadata"]) for r in results]
        scores = self.rerank_provider.rerank(query, candidates)

        reranked = sorted(zip(results, scores), key=lambda pair: pair[1], reverse=True)[:k]
        return [
            {**result, "rank": new_rank, "rerank_score": float(score)}
            for new_rank, (result, score) in enumerate(reranked, start=1)
        ]

    def compare_strategies(self, queries: List[str], document_type: str, strategies_compare: list[str], k: int = 15, use_hybrid: bool = False) -> Dict:
        """
        Compares different search strategies for a list of queries, evaluating each one's performance.

        Parameters:
            queries (List[str]): List of query strings to evaluate.
            document_type (str): Document type or context to use when evaluating the strategies.
            strategies_compare (list[str]): Strategies to compare (e.g.: ["full", "chunks"]).
            k (int, optional): Number of results to return per query. Defaults to 15.
            use_hybrid (bool): If True, evaluates each strategy with
                `evaluate_strategy_hybrid` (dense + BM25 via RRF) instead of
                `evaluate_strategy`. Defaults to False.

        Returns:
            Dict: A dictionary with, for each query, the results of the different strategies evaluated.
        """
        comparisons = {}
        evaluate = self.evaluate_strategy_hybrid if use_hybrid else self.evaluate_strategy

        for query in queries:
            logger.info(_("Query: %(query)s...") % {"query": query[:50]})
            comparisons[query] = {}

            for strategy in strategies_compare:
                results = evaluate(query, document_type, strategy, k)
                comparisons[query][strategy] = results

                if results:
                    if 'avg_distance' in results:
                        logger.info(_("  %(strategy)s: time=%(time)ss, avg_dist=%(dist)s") % {
                            "strategy": strategy,
                            "time": f"{results['search_time']:.3f}",
                            "dist": f"{results['avg_distance']:.3f}",
                        })
                    else:
                        avg_rrf = np.mean([r['rrf_score'] for r in results['results']]) if results['results'] else 0.0
                        logger.info(_("  %(strategy)s: time=%(time)ss, avg_rrf=%(rrf)s") % {
                            "strategy": strategy,
                            "time": f"{results['search_time']:.3f}",
                            "rrf": f"{avg_rrf:.4f}",
                        })

        return comparisons

    def evaluate_retrieval(
        self, labeled_queries: List[Dict], document_type: str, strategies: List[str],
        k: int = 5, use_hybrid: bool = False, rerank: bool = False
    ) -> Dict[str, Dict]:
        """
        Measures retrieval quality per strategy against queries whose right answers are
        known — the way to pick a strategy (or to compare dense/hybrid/rerank, embedding
        prefixes, models...) on your own documents.

        Parameters:
            labeled_queries (List[Dict]): One dict per query, with:
                - "query" (str): the query, as a user would type it.
                - "relevant_files" (List[str]): the file paths (as stored in
                  `metadata["file"]`) that answer it.
                - "relevant_text" (str, optional): a passage that answers it. A query is
                  a passage hit when some returned text contains this passage (compared
                  with whitespace normalized).
            document_type (str): Document type whose (already loaded) indices are searched.
            strategies (List[str]): Strategies to evaluate (e.g.: ["full", "sections", "chunks"]).
            k (int): Results retrieved per query. Defaults to 5.
            use_hybrid (bool), rerank (bool): Same retrieval options as `generate_search_by_type`.

        Returns:
            Dict[str, Dict]: For each strategy:
                - "recall_at_k": fraction of queries with a relevant file in the top `k`.
                - "mrr": mean reciprocal rank of the first relevant file (0 when absent).
                - "passage_recall_at_k": fraction of the queries that have "relevant_text"
                  whose passage is contained in a returned text; None if none has one.
                - "avg_result_chars": mean length of the returned texts — what the
                  strategy costs as LLM context ("full" finds the right file easily
                  precisely by returning whole documents).
                - "n_queries": number of queries evaluated.
        """
        normalize = lambda text: " ".join(text.split())
        report = {}
        for strategy in strategies:
            reciprocal_ranks, result_chars, passage_queries, passage_hits = [], [], 0, 0
            for labeled in labeled_queries:
                items = self._retrieve(labeled["query"], document_type, strategy, k, use_hybrid, rerank)
                texts = [self._extract_metadata_text(item["metadata"]) for item in items]
                relevant_files = set(labeled["relevant_files"])
                first_rank = next((rank for rank, item in enumerate(items, start=1) if item["metadata"]["file"] in relevant_files), None)
                reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
                result_chars.extend(len(text) for text in texts)
                if labeled.get("relevant_text"):
                    passage_queries += 1
                    passage = normalize(labeled["relevant_text"])
                    passage_hits += any(passage in normalize(text) for text in texts)
            report[strategy] = {
                "recall_at_k": float(np.mean([rr > 0 for rr in reciprocal_ranks])) if reciprocal_ranks else 0.0,
                "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
                "passage_recall_at_k": passage_hits / passage_queries if passage_queries else None,
                "avg_result_chars": float(np.mean(result_chars)) if result_chars else 0.0,
                "n_queries": len(labeled_queries),
            }
        return report

    def calculate_heuristic_score(self, comparison_results: Dict, keywords: Optional[List[str]] = None) -> Dict:
        """
        Computes a heuristic score to compare the search strategies based on multiple criteria.

        Deprecated: the score (speed, distance, variance, file diversity, keyword presence)
        doesn't measure relevance, and it structurally favors "full" — every "full" result
        is a different file (diversity is always 1.0) and whole documents contain keywords
        far more often. Use `evaluate_retrieval` with labeled queries instead.

        Parameters:
            comparison_results (Dict): Comparison results between strategies, from `compare_strategies`.
            keywords (Optional[List[str]]): List of keywords to check for presence in the results.

        Returns:
            Dict: A dictionary with the mean score and standard deviation for each strategy,
                considering the defined criteria.
        """
        warnings.warn(
            "calculate_heuristic_score is deprecated: its score doesn't measure relevance and structurally "
            "favors the 'full' strategy. Use evaluate_retrieval with labeled queries instead.",
            DeprecationWarning, stacklevel=2,
        )
        return self._heuristic_scores(comparison_results, keywords)

    def _heuristic_scores(self, comparison_results: Dict, keywords: Optional[List[str]]) -> Dict:
        """`calculate_heuristic_score` without the deprecation warning (also used by `generate_search`)."""
        scores: Dict[str, List[float]] = defaultdict(list)

        for query, strategies in comparison_results.items():
            for strategy, results in strategies.items():
                if results:
                    final_score = self._calculate_single_strategy_score(results, keywords)
                    scores[strategy].append(final_score)

        return {
            strategy: {
                "mean_score": np.mean(score_list),
                "std_score": np.std(score_list),
                "scores": score_list
            }
            for strategy, score_list in scores.items() if score_list
        }

    @staticmethod
    def _extract_metadata_text(metadata: Dict) -> str:
        """
        Returns the text of a metadata item, whichever strategy produced it
        ("full" stores it in `content`, "section" in `section_text`, "chunk" in `chunk_text`).
        """
        if "chunk_text" in metadata:
            return metadata["chunk_text"]
        if "section_text" in metadata:
            return metadata["section_text"]
        content = metadata.get("content", "")
        return " ".join(content) if isinstance(content, list) else content

    def _calculate_single_strategy_score(self, results: Dict, keywords: Optional[List[str]]) -> float:
        """
        Computes the heuristic score for a single strategy/result.
        """
        # --- Criterion 1: Speed (lower is better)
        speed_score = 1 / (1 + results['search_time'] * 10)

        # --- Criteria 2 & 3: Relevance signal — evaluate_strategy has avg_distance/
        # std_distance (L2, lower is better); evaluate_strategy_hybrid has neither
        # (it fuses two incompatible scales via RRF instead) but carries rrf_score
        # per result (already higher-is-better, naturally bounded).
        if 'avg_distance' in results:
            distance_score = 1 / (1 + results['avg_distance'])
            variance_score = 1 / (1 + results['std_distance'])
        else:
            rrf_scores = [r['rrf_score'] for r in results['results']]
            distance_score = float(np.mean(rrf_scores)) if rrf_scores else 0.0
            variance_score = 1 / (1 + float(np.std(rrf_scores))) if rrf_scores else 0.0

        # --- Criterion 4: File diversity among the top-k results
        unique_files = len({r['metadata']['file'] for r in results['results']})
        diversity_score = unique_files / len(results['results']) if results['results'] else 0

        # --- Criterion 5 (new): Keyword presence
        keyword_score_value = 0.0
        if keywords:
            keyword_scores = []
            for r in results['results']:
                text = self._extract_metadata_text(r['metadata'])
                count = sum(1 for kw in keywords if kw.lower() in text.lower())
                keyword_scores.append(count / len(keywords))
            keyword_score_value = np.mean(keyword_scores) if keyword_scores else 0.0

        # --- Final score with adjusted weights
        final_score = (
            0.1 * speed_score +
            0.3 * distance_score +
            0.1 * variance_score +
            0.3 * diversity_score +
            0.2 * keyword_score_value
        )
        return final_score

    def generate_search(self, received_query:list[str], keywords:List[str], document_type:str, strategies_compare:list[str], use_hybrid: bool = False) -> Tuple[Dict, Dict]:
        """
        Runs a search based on a received query, keywords, and document type.

        Parameters:
            received_query (list[str]): List containing the received query.
            keywords (List[str]): Keywords to help the search.
            document_type (str): Document type to steer the search strategy.
            strategies_compare (list[str]): Strategies to compare (e.g.: ["full", "chunks"]).
            use_hybrid (bool): Passed through to `compare_strategies` — evaluates with
                `evaluate_strategy_hybrid` (dense + BM25) instead of `evaluate_strategy`
                when True. Defaults to False.

        Returns:
            tuple: A tuple with the search results and the heuristic scores.

        Deprecated: the scores come from `calculate_heuristic_score` — see why there.
        Use `evaluate_retrieval` with labeled queries to compare strategies.
        """
        warnings.warn(
            "generate_search is deprecated: its scores come from calculate_heuristic_score, which "
            "doesn't measure relevance. Use evaluate_retrieval with labeled queries to compare strategies.",
            DeprecationWarning, stacklevel=2,
        )
        # Searched as-is, not clean_text'ed — see generate_search_by_type.
        results = self.compare_strategies([received_query[0]], document_type, strategies_compare, k=20, use_hybrid=use_hybrid)
        scores = self._heuristic_scores(results, keywords)

        return results, scores

    def generate_search_by_type(self, received_query: str, document_type: str, strategy: str, require_gpu: bool, k: int = 5, use_hybrid: bool = False, rerank: bool = False) -> List[str]:
        """
        Runs a search based on a received query, using a specific strategy.

        Parameters:
            received_query (str): The query string to search for.
            document_type (str): Document type to steer the search strategy.
            strategy (str): Indexing strategy to use (e.g.: 'chunks').
            require_gpu (bool): If True, requires the FAISS index to be loaded on GPU.
            k (int): How many texts to return. Defaults to 5.
            use_hybrid (bool): If True, retrieves with `evaluate_strategy_hybrid`
                (dense + BM25 via RRF) instead of `evaluate_strategy`. Defaults to False.
            rerank (bool): If True, retrieves a larger candidate pool
                (`max(k * 4, 20)`) and narrows it to `k` via `rerank_results` (needs
                `self.rerank_provider` configured on the constructor). Defaults to False.

        Returns:
            List[str]: The texts of the search results, best first — a chunk, a section,
                or a whole document, depending on `strategy`.
        """
        if self._is_single_index_loaded(document_type, strategy, require_gpu=False):
            logger.info(_("Index for '%(document_type)s/%(strategy)s' found. Running search") % {"document_type": document_type, "strategy": strategy})

            if require_gpu and not self._is_single_index_loaded(document_type, strategy, require_gpu=True):
                # Already in memory, just not GPU-accelerated: move that same index
                # instead of reloading it from disk, which would silently discard
                # anything added via add_new_documents since it was saved (and re-read
                # the files on every call). Without CUDA this keeps the index on CPU.
                index, metadata, embeddings = self.indices[document_type][strategy]
                self.indices[document_type][strategy] = (self._move_index_to_gpu(index, strategy), metadata, embeddings)
        else:
            logger.info(_("Index for '%(document_type)s/%(strategy)s' not found. Loading now...") % {"document_type": document_type, "strategy": strategy})

            self.load_indices(path_indices=config.DEFAULT_PATH_INDICES,
                            document_types=[document_type],
                            strategies=[strategy])

            is_now_loaded = self.is_index_loaded(document_types=[document_type], strategies=[strategy], require_gpu=require_gpu)
            logger.info(_("Index for '%(document_type)s/%(strategy)s' loaded - '%(is_now_loaded)s'. Running search...") % {"document_type": document_type, "strategy": strategy, "is_now_loaded": is_now_loaded})

        # clean_text always returns a 1-element list ([text.strip()]), even when the
        # text becomes empty after cleaning — so the emptiness has to be checked on
        # the string itself, not on the (always truthy) wrapping list.
        if not clean_text(received_query)[0]:
            return []

        # The query itself is searched as-is: embedding and cross-encoder models read
        # natural language (documents are embedded unaltered too), and clean_text's
        # stopword removal drops words like "não"/"sem" that invert a query's meaning.
        # Only the BM25 side normalizes it, inside _tokenize_for_bm25.
        result_items = self._retrieve(received_query.strip(), document_type, strategy, k, use_hybrid, rerank)
        return [self._extract_metadata_text(res['metadata']) for res in result_items]

    def _retrieve(self, query: str, document_type: str, strategy: str, k: int, use_hybrid: bool, rerank: bool) -> List[Dict]:
        """
        Retrieval path shared by `generate_search_by_type` and `evaluate_retrieval`:
        dense or hybrid search for `k` results — or, with `rerank`, a larger candidate
        pool (`max(k * 4, 20)`) narrowed down to `k` via `rerank_results`.
        """
        evaluate = self.evaluate_strategy_hybrid if use_hybrid else self.evaluate_strategy
        retrieval_k = max(k * 4, 20) if rerank else k
        result_items = evaluate(query, document_type, strategy, k=retrieval_k).get('results', [])
        if rerank:
            result_items = self.rerank_results(query, result_items, k=k)
        return result_items
