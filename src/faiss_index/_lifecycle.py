"""
Mixin for FaissDocumentIndex: building indices from documents on disk, persisting
them, loading/unloading them in memory (optionally onto GPU via CUDA — see
`_gpu_support.py`), and appending new documents to already-loaded indices. Composed
into the class in `core.py`; not meant to be imported directly by users.
"""
import gc
import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
import psutil

from . import config
from ._gpu_support import FAISS_HAS_GPU_SUPPORT
from .i18n import _

logger = logging.getLogger(__name__)


class IndexLifecycleMixin:

    def is_index_loaded(self, document_types: list[str], strategies: list[str], require_gpu: bool) -> bool:
        """
        Checks whether the indices for all combinations of document types and
        strategies have already been loaded into memory.

        Parameters:
            document_types (list[str]): The list of document types to check.
            strategies (list[str]): The list of strategies to check.
            require_gpu (bool): If True, also checks whether the index is being
                                 searched with GPU acceleration (CUDA or MPS).

        Returns:
            bool: True if ALL requested combinations are loaded (and on GPU, if requested),
                  False if any of them fails.
        """
        if not document_types or not strategies:
            logger.warning(_("The 'document_types' and 'strategies' lists cannot be empty."))
            return False

        for document_type in document_types:
            for strategy in strategies:
                if not self._is_single_index_loaded(document_type, strategy, require_gpu):
                    return False

        num_combinations = len(document_types) * len(strategies)
        logger.info(_("All %(num_combinations)s requested index combinations are already loaded.") % {"num_combinations": num_combinations})
        return True

    def _is_single_index_loaded(self, document_type: str, strategy: str, require_gpu: bool) -> bool:
        """Checks whether a single document type/strategy combination is loaded correctly."""
        if document_type not in self.indices:
            logger.debug(_("Index for document type '%(document_type)s' not found.") % {"document_type": document_type})
            return False

        if strategy not in self.indices[document_type]:
            logger.debug(_("Strategy '%(strategy)s' for document type '%(document_type)s' not found.") % {"strategy": strategy, "document_type": document_type})
            return False

        index, metadata, embeddings = self.indices[document_type][strategy]
        if index is None or metadata is None or embeddings is None:
            logger.debug(_("Data for '%(document_type)s/%(strategy)s' is not loaded (None).") % {"document_type": document_type, "strategy": strategy})
            return False

        # "GPU" covers two independent paths: index moved to GPU via CUDA
        # (faiss-gpu, nonexistent on macOS) OR search accelerated via MPS (self._should_use_mps).
        is_gpu_index = (
            (FAISS_HAS_GPU_SUPPORT and isinstance(index, faiss.GpuIndex))
            or self._should_use_mps(index)
        )
        if require_gpu and not is_gpu_index:
            logger.debug(_("Index for '%(document_type)s/%(strategy)s' is not on GPU, but was expected to be.") % {"document_type": document_type, "strategy": strategy})
            return False

        return True

    def _create_and_save_index(self, strategy_name: str, embeddings: np.ndarray, metadata: list, document_type: str, output_dir: Path) -> Tuple[Optional[faiss.Index], list, Optional[np.ndarray]]:
        """
        Creates and saves a FAISS index for a specific strategy.

        Parameters:
            strategy_name (str): Strategy name (e.g., "full", "sections", "chunks").
            embeddings (np.ndarray): Numpy array with the embeddings.
            metadata (list): List of metadata corresponding to the embeddings.
            document_type (str): Document type used to name the files.
            output_dir (Path): Directory where the index files will be saved.

        Returns:
            tuple: A tuple with the FAISS index, the metadata, and the embeddings.
        """
        if embeddings.shape[0] == 0:
            logger.warning(_("No embeddings generated for strategy '%(strategy_name)s'. Skipping.") % {"strategy_name": strategy_name})
            return None, [], None

        logger.info(_("Building and saving index for strategy: '%(strategy_name)s'...") % {"strategy_name": strategy_name})
        index = self._build_faiss_index(embeddings)

        faiss.write_index(index, str(output_dir / f"{document_type}_{strategy_name}.index"))
        with open(output_dir / f"{document_type}_{strategy_name}_metadata.json", "w", encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        np.save(output_dir / f"{document_type}_{strategy_name}_embeddings.npy", embeddings)

        logger.info(_("Index for '%(strategy_name)s' saved successfully.") % {"strategy_name": strategy_name})
        return index, metadata, embeddings

    def build_indices(
        self,
        document_type: str,
        base_data_dir: str = config.DEFAULT_BASE_DATA_DIR,
        output_index_dir: str = config.DEFAULT_OUTPUT_INDEX_DIR,
        doc_limit: Optional[int] = None
    ):
        """
        Builds and saves FAISS indices for the different document indexing strategies.

        Parameters:
            document_type (str): Document type to build the indices for.
            base_data_dir (str): Base directory where the document data is stored.
            output_index_dir (str): Directory where the FAISS indices will be saved.
            doc_limit (Optional[int]): Maximum number of documents to process. If None, processes all.

        Returns:
            None
        """
        base_dir = Path(base_data_dir) / document_type
        # Subfolder per document_type, to match the layout `load_indices` expects.
        output_dir = Path(output_index_dir) / document_type
        output_dir.mkdir(parents=True, exist_ok=True)

        file_paths = sorted(p for p in base_dir.rglob('*') if p.suffix in self.SUPPORTED_FILE_EXTENSIONS)

        if doc_limit is not None and doc_limit > 0:
            logger.info(_("Found %(total)s matching file(s). Limit of %(doc_limit)s applied.") % {"total": len(file_paths), "doc_limit": doc_limit})
            file_paths = file_paths[:doc_limit]

        docs = []
        for file_path in file_paths:
            try:
                content = self.read_document(str(file_path))
                if content and content.strip():
                    docs.append((str(file_path), content))
            except Exception as e:
                logger.error(_("Error reading %(file_path)s: %(error)s") % {"file_path": file_path, "error": e}, exc_info=True)

        if not docs:
            logger.warning(_("No documents found for document type: %(document_type)s") % {"document_type": document_type})
            return

        logger.info(_("Processing %(count)s documents of %(document_type)s...") % {"count": len(docs), "document_type": document_type})

        embeddings_map = {
            "full": self.create_embeddings_full(docs),
            "chunks": self.create_embeddings_chunks(docs)
        }

        if self.structure_provider is not None:
            # Each document's own structure defines its sections — no schema to
            # calibrate/save (the pattern-based path below is skipped entirely).
            embeddings_map["sections"] = self.create_embeddings_sections_via_structure(docs)
        else:
            if document_type not in self.section_schemas:
                self._load_section_schema(document_type, output_dir)
            if document_type not in self.section_schemas:
                sample_texts = [content for _, content in docs[: self.MAX_SECTION_SAMPLE_DOCS]]
                self.register_document_type(document_type, sample_texts)
            self._save_section_schema(document_type, output_dir)

            if self.section_schemas.get(document_type):
                embeddings_map["sections"] = self.create_embeddings_sections(docs, document_type)
            else:
                logger.warning(_("No section schema for '%(document_type)s'. The 'sections' strategy will not be built.") % {"document_type": document_type})

        self.indices[document_type] = {}
        # Stale otherwise: evaluate_strategy_hybrid's cached BM25 indices for this
        # document_type were built from the metadata being replaced right here.
        self._bm25_indices.pop(document_type, None)

        # Iterates over the strategies and applies the creation/save logic
        for strategy, (embeddings, meta) in embeddings_map.items():
            result = self._create_and_save_index(strategy, embeddings, meta, document_type, output_dir)
            self.indices[document_type][strategy] = result

        logger.info(_("Index construction for '%(document_type)s' complete.") % {"document_type": document_type})

    def _log_memory_usage(self, stage: str):
        """Helper function to log current memory usage."""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        # Converts bytes to GB for easier reading
        rss_gb = mem_info.rss / (1024 ** 3)
        logger.info(_("===> Memory Usage (%(stage)s): %(rss_gb)s GB") % {"stage": stage, "rss_gb": f"{rss_gb:.2f}"})

    def load_indices(
        self,
        path_indices: str,
        document_types: list[str],
        strategies: list[str],
        use_gpu: bool = True
    ) -> dict:
        """
        Loads FAISS indices, metadata, and embeddings in a memory-optimized way.
        If use_gpu is True, tries to move the indices to the GPU.

        Parameters:
            path_indices (str): Path to the indices' base directory.
            document_types (list[str]): Document types to load.
            strategies (list[str]): Strategies to load (e.g.: ['full', 'chunks']).
            use_gpu (bool): If True, tries to move the FAISS index to the GPU.

        Returns:
            dict: Dictionary with the loaded data.
        """
        self._log_memory_usage(_("Start of `load_indices`"))
        loaded_data = {}

        for document_type in document_types:
            self.indices.setdefault(document_type, {})
            loaded_data.setdefault(document_type, {})

            document_dir = os.path.join(path_indices, document_type)
            if not os.path.exists(document_dir):
                logger.error(_("Error: Directory '%(document_dir)s' for document type '%(document_type)s' not found.") % {"document_dir": document_dir, "document_type": document_type})
                continue

            logger.info(_("Loading data from '%(document_dir)s'...") % {"document_dir": document_dir})
            self._load_section_schema(document_type, Path(document_dir))

            for strategy in strategies:
                self._load_single_strategy(
                    document_type, document_dir, strategy, use_gpu, loaded_data
                )

        self._log_memory_usage(_("End of `load_indices`"))
        return loaded_data

    def _move_index_to_gpu(self, index_cpu, strategy):
        if not FAISS_HAS_GPU_SUPPORT:
            # Build without CUDA (faiss-cpu, the only variant installable on macOS) —
            # there's nothing to try here. On Apple Silicon, real acceleration is via MPS
            # at search time (see `use_mps` on the constructor and `_should_use_mps`), not on load.
            logger.debug(
                _("   FAISS installed without CUDA GPU support — keeping '%(strategy)s' on CPU.") % {"strategy": strategy}
            )
            return index_cpu
        try:
            if not hasattr(self, "gpu_resources") or self.gpu_resources is None:
                logger.info(_("   Initializing GPU resources for FAISS..."))
                self.gpu_resources = faiss.StandardGpuResources()
            logger.info(_("   Moving index '%(strategy)s' to GPU...") % {"strategy": strategy})
            index_gpu = faiss.index_cpu_to_gpu(self.gpu_resources, 0, index_cpu)
            self._log_memory_usage(_("After moving index '%(strategy)s' to GPU") % {"strategy": strategy})
            logger.info(_("Successfully moved index to GPU."))
            return index_gpu
        except Exception as gpu_error:
            logger.warning(_("    Failed to move index to GPU: %(error)s. Using CPU.") % {"error": gpu_error})
            return index_cpu

    def _load_single_strategy(
        self, document_type, document_dir, strategy, use_gpu, loaded_data
    ):
        logger.info(_("-> Loading strategy '%(strategy)s'...") % {"strategy": strategy})

        index_path = os.path.join(document_dir, f"{document_type}_{strategy}.index")
        metadata_path = os.path.join(document_dir, f"{document_type}_{strategy}_metadata.json")
        embeddings_path = os.path.join(document_dir, f"{document_type}_{strategy}_embeddings.npy")

        if not all(os.path.exists(p) for p in [index_path, metadata_path, embeddings_path]):
            logger.warning(
                _("   Warning: Files for strategy '%(strategy)s' of type '%(document_type)s' not found. Skipping.")
                % {"strategy": strategy, "document_type": document_type}
            )
            return

        try:
            self._log_memory_usage(_("Before loading index '%(strategy)s'") % {"strategy": strategy})
            index_cpu = faiss.read_index(str(index_path))
            if hasattr(index_cpu, "nprobe"):
                index_cpu.nprobe = self.ivf_nprobe
            if hasattr(index_cpu, "make_direct_map"):
                try:
                    index_cpu.make_direct_map()
                except Exception:
                    pass  # may already have a direct map (e.g.: reconstruction after a previous `add`)
            self._log_memory_usage(_("After loading index '%(strategy)s' to CPU") % {"strategy": strategy})
            final_index = (
                self._move_index_to_gpu(index_cpu, strategy) if use_gpu else index_cpu
            )

            self._log_memory_usage(_("Before loading metadata '%(strategy)s'") % {"strategy": strategy})
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            self._log_memory_usage(_("After loading metadata '%(strategy)s'") % {"strategy": strategy})

            embeddings = np.load(embeddings_path, mmap_mode="r")

            loaded_tuple = (final_index, metadata, embeddings)
            self.indices[document_type][strategy] = loaded_tuple
            loaded_data[document_type][strategy] = loaded_tuple
            # Stale otherwise: a strategy already loaded that gets reloaded here (no
            # unload_indices() in between) would keep evaluate_strategy_hybrid's cached
            # BM25 index pointing at the previous metadata list.
            self._bm25_indices.get(document_type, {}).pop(strategy, None)

            device = (
                "GPU"
                if use_gpu and FAISS_HAS_GPU_SUPPORT and isinstance(final_index, faiss.GpuIndex)
                else "CPU"
            )
            logger.info(
                _("   Successfully loaded '%(strategy)s': %(ntotal)s vectors on device '%(device)s'.")
                % {"strategy": strategy, "ntotal": final_index.ntotal, "device": device}
            )

        except Exception as e:
            logger.error(
                _("   An error occurred while loading '%(strategy)s': %(error)s") % {"strategy": strategy, "error": e},
                exc_info=True,
            )
        finally:
            gc.collect()

    def unload_indices(self, document_type: str, strategy: str = None) -> None:
        """
        Deallocates (removes the references to) one or more sets of indices from memory.

        Parameters:
            document_type (str): The document type whose indices should be removed.
            strategy (str, optional): The specific strategy to remove.
                                      If None, removes all indices for the document_type.

        Returns:
            None
        """
        if document_type not in self.indices:
            logger.info(_("Info: No index loaded for document type '%(document_type)s'.") % {"document_type": document_type})
            return

        # Case 1: Deallocate a specific strategy
        if strategy:
            if strategy in self.indices[document_type]:
                # Removes the reference to the (index, metadata, embeddings) tuple
                del self.indices[document_type][strategy]
                # Stale otherwise: a future load_indices() for the same document_type/
                # strategy could point at different metadata than what the BM25 index
                # (see evaluate_strategy_hybrid) was built from.
                self._bm25_indices.get(document_type, {}).pop(strategy, None)
                logger.info(_("Success: Strategy '%(strategy)s' for '%(document_type)s' has been unloaded.") % {"strategy": strategy, "document_type": document_type})

                # If the strategies dict becomes empty, also remove the document_type
                if not self.indices[document_type]:
                    del self.indices[document_type]
                    logger.info(_("Info: No other strategy remains, removing entry for '%(document_type)s'.") % {"document_type": document_type})
            else:
                logger.warning(_("Warning: Strategy '%(strategy)s' not found for '%(document_type)s'.") % {"strategy": strategy, "document_type": document_type})

        # Case 2: Deallocate all strategies for the document_type
        else:
            # Removes the reference to the whole strategies dict
            del self.indices[document_type]
            self._bm25_indices.pop(document_type, None)
            logger.info(_("Success: All indices for '%(document_type)s' have been unloaded.") % {"document_type": document_type})

        gc.collect()

    def add_new_documents(self, document_type: str, new_docs: List[Tuple[str, str]]) -> None:
        """
        Adds new documents to an existing FAISS index for a specific document type.
        This updates the index, the metadata, and the embeddings with the new documents provided.

        Parameters:
            document_type (str): The document type the new documents belong to.
            new_docs (List[Tuple[str, str]]): A list of tuples, each with the file path
                and the document's content.

        Returns:
            None
        """
        if document_type not in self.indices:
            raise ValueError(f"Index for document type '{document_type}' not found. Load the index before adding documents.")

        for strategy, (index, metadata, embeddings) in self.indices[document_type].items():
            logger.info(_("Adding new documents to strategy '%(strategy)s' for '%(document_type)s'...") % {"strategy": strategy, "document_type": document_type})

            if strategy == "full":
                new_embeddings, new_metadata = self.create_embeddings_full(new_docs)
            elif strategy == "sections":
                if self.structure_provider is not None:
                    new_embeddings, new_metadata = self.create_embeddings_sections_via_structure(new_docs)
                else:
                    new_embeddings, new_metadata = self.create_embeddings_sections(new_docs, document_type)
            elif strategy == "chunks":
                new_embeddings, new_metadata = self.create_embeddings_chunks(new_docs)
            else:
                continue

            # Adds the new embeddings to the FAISS index (index.make_direct_map(), already
            # called on index creation/load, keeps index.reconstruct(i) working for the new vectors)
            index.add(new_embeddings.astype('float32'))

            # Updates the metadata. We don't np.vstack the embeddings array: it may be
            # memory-mapped from disk (mmap_mode="r") and vstack would force materializing
            # all of it into RAM. evaluate_strategy already reads vectors via
            # index.reconstruct(), so this array doesn't need to stay up to date for search to work correctly.
            metadata.extend(new_metadata)

            # Updates the tuple in the indices structure (embeddings kept as-is)
            self.indices[document_type][strategy] = (index, metadata, embeddings)
            # Invalidates the cached BM25 index (see evaluate_strategy_hybrid) — it was
            # built from `metadata` before these documents were appended to it.
            self._bm25_indices.get(document_type, {}).pop(strategy, None)

            logger.info(_("New documents added to strategy '%(strategy)s' for '%(document_type)s'. Total now: %(ntotal)s vectors.") % {"strategy": strategy, "document_type": document_type, "ntotal": index.ntotal})

    def unload_all_indices(self) -> None:
        """
        Checks whether any indices are loaded in memory and unloads all of them.

        This clears the `self.indices` dictionary, removing all references to the
        index, metadata, and embeddings objects, allowing the garbage collector to
        free the memory used.

        Returns:
            None
        """
        if not self.indices:
            logger.info(_("Info: No index is currently loaded."))
            return

        num_document_types = len(self.indices)
        logger.info(_("Info: %(num_document_types)s document type(s) found. Unloading all indices...") % {"num_document_types": num_document_types})

        # Removes the reference to the whole indices dict
        self.indices.clear()
        self._bm25_indices.clear()

        # Calls the garbage collector to free memory as quickly as possible
        gc.collect()

        logger.info(_("Success: All indices have been unloaded and memory has been freed."))
