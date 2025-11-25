"""
Objaverse asset retriever using pre-computed CLIP and SBERT embeddings.
Adapted from SceneEval's objaverse_retriever.py.
"""

import os
from pathlib import Path
from typing import Optional

import compress_json
import compress_pickle
import numpy as np
import torch
import torch.nn.functional as F


class ObjaverseRetriever:
    """Retrieves Objaverse assets using CLIP+SBERT embeddings."""

    def __init__(
        self,
        objathor_dir: str,
        clip_model=None,
        clip_tokenizer=None,
        sbert_model=None,
        retrieval_threshold: float = 28.0,
        use_text: bool = True,
    ):
        """
        Initialize the retriever with pre-computed embeddings.

        Args:
            objathor_dir: Path to objathor data directory containing:
                - annotations.json.gz (or similar)
                - clip_features.pkl
                - sbert_features.pkl
            clip_model: CLIP model for text encoding
            clip_tokenizer: CLIP tokenizer
            sbert_model: Sentence-BERT model for text encoding
            retrieval_threshold: Minimum CLIP similarity score
            use_text: Whether to combine CLIP and SBERT scores
        """
        self.objathor_dir = Path(objathor_dir)
        self.retrieval_threshold = retrieval_threshold
        self.use_text = use_text

        # Load annotations
        annotations_path = self._find_annotations_file()
        self.database = compress_json.load(str(annotations_path))
        print(f"Loaded {len(self.database)} asset annotations")

        # Load pre-computed features
        features_dir = self.objathor_dir
        clip_features_path = features_dir / "clip_features.pkl"
        sbert_features_path = features_dir / "sbert_features.pkl"

        if not clip_features_path.exists():
            raise FileNotFoundError(f"CLIP features not found at {clip_features_path}")
        if not sbert_features_path.exists():
            raise FileNotFoundError(f"SBERT features not found at {sbert_features_path}")

        clip_features_dict = compress_pickle.load(str(clip_features_path))
        sbert_features_dict = compress_pickle.load(str(sbert_features_path))

        # Verify UIDs match
        assert clip_features_dict["uids"] == sbert_features_dict["uids"], \
            "CLIP and SBERT feature UIDs do not match"

        self.asset_ids = clip_features_dict["uids"]
        clip_features = clip_features_dict["img_features"].astype(np.float32)
        sbert_features = sbert_features_dict["text_features"].astype(np.float32)

        # Convert to tensors and normalize CLIP features
        self.clip_features = torch.from_numpy(clip_features)
        self.clip_features = F.normalize(self.clip_features, p=2, dim=-1)
        self.sbert_features = torch.from_numpy(sbert_features)

        print(f"Loaded features for {len(self.asset_ids)} assets")

        # Store models
        self.clip_model = clip_model
        self.clip_tokenizer = clip_tokenizer
        self.sbert_model = sbert_model

    def _find_annotations_file(self) -> Path:
        """Find the annotations file in the objathor directory."""
        possible_names = [
            "annotations.json.gz",
            "annotations.json",
            "objathor_annotations.json.gz",
            "objathor_annotations.json",
        ]
        for name in possible_names:
            path = self.objathor_dir / name
            if path.exists():
                return path
        raise FileNotFoundError(
            f"Could not find annotations file in {self.objathor_dir}. "
            f"Tried: {possible_names}"
        )

    def retrieve(
        self,
        queries: list[str],
        threshold: Optional[float] = None,
    ) -> list[tuple[str, float]]:
        """
        Retrieve assets matching text queries.

        Args:
            queries: List of text descriptions to search for
            threshold: CLIP similarity threshold (overrides default)

        Returns:
            List of (asset_id, score) tuples, sorted by score descending
        """
        if threshold is None:
            threshold = self.retrieval_threshold

        if self.clip_model is None or self.clip_tokenizer is None:
            raise RuntimeError("CLIP model and tokenizer must be set for retrieval")

        # Encode queries with CLIP
        with torch.no_grad():
            tokens = self.clip_tokenizer(queries)
            if hasattr(tokens, 'to'):
                tokens = tokens.to(next(self.clip_model.parameters()).device)
            query_feature_clip = self.clip_model.encode_text(tokens)
            query_feature_clip = F.normalize(query_feature_clip, p=2, dim=-1)
            query_feature_clip = query_feature_clip.cpu()

        # Compute CLIP similarities
        clip_similarities = 100 * torch.einsum(
            "ij, lkj -> ilk", query_feature_clip, self.clip_features
        )
        clip_similarities = torch.max(clip_similarities, dim=-1).values

        # Compute SBERT similarities if available
        if self.use_text and self.sbert_model is not None:
            query_feature_sbert = self.sbert_model.encode(
                queries, convert_to_tensor=True, show_progress_bar=False
            )
            query_feature_sbert = query_feature_sbert.cpu()
            sbert_similarities = query_feature_sbert @ self.sbert_features.T
            similarities = clip_similarities + sbert_similarities
        else:
            similarities = clip_similarities

        # Filter by threshold and collect results
        threshold_indices = torch.where(clip_similarities > threshold)

        unsorted_results = []
        for query_index, asset_index in zip(*threshold_indices):
            score = similarities[query_index, asset_index].item()
            unsorted_results.append((self.asset_ids[asset_index], score))

        # Sort by score descending
        results = sorted(unsorted_results, key=lambda x: x[1], reverse=True)
        return results

    def get_asset_metadata(self, asset_id: str) -> Optional[dict]:
        """Get metadata for a specific asset."""
        return self.database.get(asset_id)

    def compute_size_difference(
        self,
        target_size: tuple[float, float, float],
        candidates: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """
        Re-rank candidates by size similarity.

        Args:
            target_size: Target bounding box size (x, y, z) in cm
            candidates: List of (asset_id, score) tuples

        Returns:
            Re-ranked list of (asset_id, adjusted_score) tuples
        """
        candidate_sizes = []
        for uid, _ in candidates:
            metadata = self.database.get(uid, {})
            bbox = metadata.get("boundingBox", {})
            size = [
                bbox.get("x", 0) * 100,
                bbox.get("y", 0) * 100,
                bbox.get("z", 0) * 100,
            ]
            size.sort()
            candidate_sizes.append(size)

        candidate_sizes = torch.tensor(candidate_sizes)

        target_size_list = list(target_size)
        target_size_list.sort()
        target_size_tensor = torch.tensor(target_size_list)

        size_difference = abs(candidate_sizes - target_size_tensor).mean(axis=1) / 100
        size_difference = size_difference.tolist()

        candidates_with_size_difference = []
        for i, (uid, score) in enumerate(candidates):
            candidates_with_size_difference.append(
                (uid, score - size_difference[i] * 10)
            )

        # Sort by adjusted score
        candidates_with_size_difference = sorted(
            candidates_with_size_difference, key=lambda x: x[1], reverse=True
        )

        return candidates_with_size_difference
