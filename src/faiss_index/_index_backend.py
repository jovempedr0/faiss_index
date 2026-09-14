"""
Mixin for FaissDocumentIndex: FAISS index construction (flat/IVFFlat/IVFPQ, chosen by
corpus size) and the low-level dense-search path, including the MPS (Apple Silicon
GPU) fallback for "flat" indices via `torch`. Composed into the class in `core.py`;
not meant to be imported directly by users.
"""
import logging
from typing import Tuple

import faiss
import numpy as np

from . import constants
from ._gpu_support import torch
from .i18n import _

logger = logging.getLogger(__name__)


class FaissIndexBackendMixin:

    MIN_TRAINING_POINTS_PER_CLUSTER = constants.MIN_TRAINING_POINTS_PER_CLUSTER

    def _select_index_kind(self, n: int) -> str:
        """Decides which FAISS index kind to build for `n` vectors."""
        if self.index_type != "auto":
            return self.index_type
        small_max, medium_max = self.auto_index_thresholds
        if n <= small_max:
            return "flat"
        elif n <= medium_max:
            return "ivf_flat"
        return "ivf_pq"

    def _compute_nlist(self, n: int) -> int:
        """
        Number of clusters for IVFFlat/IVFPQ: uses `self.ivf_nlist` if given, otherwise
        ~sqrt(n), always respecting a minimum number of training points per cluster
        (otherwise FAISS trains poorly).
        """
        desired = self.ivf_nlist or max(1, int(np.sqrt(n)))
        max_supported = max(1, n // self.MIN_TRAINING_POINTS_PER_CLUSTER)
        return max(1, min(desired, max_supported))

    def _should_use_mps(self, index) -> bool:
        """
        Decides whether search on this index should run via MPS (torch) instead of
        FAISS CPU. Only applies to "flat" indices — IVF/IVFPQ indices (identified here
        by the presence of the `nprobe` attribute, which only exists on IVF variants)
        keep using FAISS's native approximate search, which is already the right
        strategy for their corpus size.
        """
        return self.mps_device is not None and index is not None and not hasattr(index, "nprobe")

    def _mps_flat_search(self, index: faiss.Index, query_embedding: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Exhaustive search equivalent to an IndexFlatL2's, running on MPS (Apple Silicon
        GPU) via torch instead of FAISS CPU. Reconstructs the stored vectors from the
        index itself (without keeping a separate embeddings array) and computes the
        squared L2 distance — the same metric `faiss.IndexFlatL2.search` uses.

        Parameters:
            index (faiss.Index): "flat" index (no `nprobe`) to read the vectors from.
            query_embedding (np.ndarray): Query vector(s).
            k (int): Number of nearest neighbors to return.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (distances, indices), in the same format
            `faiss.Index.search` returns.
        """
        n = index.ntotal
        if n == 0:
            return np.zeros((query_embedding.shape[0], 0), dtype="float32"), np.zeros((query_embedding.shape[0], 0), dtype="int64")

        k = min(k, n)
        stored = np.asarray(index.reconstruct_n(0, n), dtype="float32")

        stored_t = torch.from_numpy(stored).to(self.mps_device)
        query_t = torch.from_numpy(query_embedding.astype("float32")).to(self.mps_device)

        sq_distances = torch.cdist(query_t, stored_t, p=2) ** 2
        top_distances, top_indices = torch.topk(sq_distances, k, dim=1, largest=False)

        return top_distances.detach().cpu().numpy(), top_indices.detach().cpu().numpy()

    def _build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Creates (and trains, if needed) a FAISS index suited to the corpus size:
        "flat" (exact) for small corpora, "ivf_flat"/"ivf_pq" (approximate, faster and
        lighter on memory) for medium/large corpora — see `index_type` and
        `auto_index_thresholds` on the constructor.

        Parameters:
            embeddings (np.ndarray): Vectors to index.

        Returns:
            faiss.Index: The already-trained (if applicable) index with the vectors added.
        """
        n = embeddings.shape[0]
        vectors = embeddings.astype('float32')
        kind = self._select_index_kind(n)

        if kind == "flat":
            index = faiss.IndexFlatL2(self.embedding_dim)
            index.add(vectors)
            logger.info(_("'flat' index created with %(n)s vectors (exact search).") % {"n": n})
            return index

        nlist = self._compute_nlist(n)
        quantizer = faiss.IndexFlatL2(self.embedding_dim)

        if kind == "ivf_pq":
            # IndexIVFPQ's product-quantizer training needs at least 2**pq_nbits points,
            # independent of nlist — a much higher floor than _compute_nlist's per-cluster
            # minimum (which only covers IndexIVFFlat's requirement). FAISS raises a hard
            # C++ error below this, not a warning, so this has to be checked beforehand.
            min_pq_training_points = 2 ** self.pq_nbits
            if n < min_pq_training_points:
                logger.warning(
                    _("Only %(n)s vectors available, but 'ivf_pq' needs at least %(min)s to "
                      "train its product quantizer (2**pq_nbits=%(pq_nbits)s) — falling back "
                      "to 'ivf_flat' for this corpus.")
                    % {"n": n, "min": min_pq_training_points, "pq_nbits": self.pq_nbits}
                )
                kind = "ivf_flat"

        if kind == "ivf_flat":
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
        elif kind == "ivf_pq":
            index = faiss.IndexIVFPQ(quantizer, self.embedding_dim, nlist, self.pq_m, self.pq_nbits)
        else:
            raise ValueError(f"Unknown index type: {kind}")

        index.train(vectors)
        index.add(vectors)
        index.nprobe = self.ivf_nprobe
        # Enables index.reconstruct(i) (used in evaluate_strategy) for IVF* indices.
        index.make_direct_map()
        logger.info(
            _("'%(kind)s' index created with %(n)s vectors, nlist=%(nlist)s, nprobe=%(nprobe)s (approximate search).")
            % {"kind": kind, "n": n, "nlist": nlist, "nprobe": self.ivf_nprobe}
        )
        return index

    def _dense_search_indices(self, index: faiss.Index, query_embedding: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs a dense search (FAISS or MPS, whichever applies — see `_should_use_mps`)
        for the `n` nearest neighbors of `query_embedding`.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (valid_indices, valid_distances) — parallel
            arrays, already filtered of the -1 "sentinel" slots FAISS's native search
            (not the MPS one, which already caps n at ntotal) fills in when n > ntotal.
        """
        if self._should_use_mps(index):
            distances, indices = self._mps_flat_search(index, query_embedding, n)
        else:
            distances, indices = index.search(query_embedding.astype('float32'), n)
        valid_mask = indices[0] >= 0
        return indices[0][valid_mask], distances[0][valid_mask]
