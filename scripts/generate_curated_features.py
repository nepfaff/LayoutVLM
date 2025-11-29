#!/usr/bin/env python3
"""
Generate CLIP and SBERT features for curated assets.

Adapted from objathor's generate_holodeck_features.py to work with the
curated asset format (blender_renders/ with render_{angle}.png files
and data.json annotations).
"""

import json
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Union

import compress_pickle
import numpy as np
import PIL.Image as Image
import torch
import tqdm
from torch.utils.data import Dataset, DataLoader

try:
    import open_clip
except ImportError:
    raise ImportError(
        "open_clip is not installed. Run: pip install open_clip_torch"
    )

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "sentence_transformers is not installed. Run: pip install sentence_transformers"
    )


if torch.cuda.is_available():
    DEFAULT_DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = "mps"
else:
    DEFAULT_DEVICE = "cpu"


class CuratedAssetDataset(Dataset):
    """Dataset for curated assets with blender renders and data.json annotations."""

    def __init__(
        self,
        asset_dir: str,
        image_preprocessor,
        img_angles: tuple = (0.0, 90.0, 270.0),
    ):
        self.asset_dir = Path(asset_dir)
        self.image_preprocessor = image_preprocessor
        self.img_angles = img_angles

        # Find all valid asset directories (those with data.json and blender_renders)
        self.assets = []
        self.annotations = {}

        print("Scanning asset directories...")
        for entry in sorted(self.asset_dir.iterdir()):
            if not entry.is_dir():
                continue
            if entry.name.startswith('.'):
                continue
            if entry.name == 'features':
                continue

            data_json = entry / "data.json"
            renders_dir = entry / "blender_renders"

            if not data_json.exists():
                print(f"  Skipping {entry.name}: no data.json")
                continue
            if not renders_dir.exists():
                print(f"  Skipping {entry.name}: no blender_renders/")
                continue

            # Check renders exist for all angles
            missing_renders = []
            for angle in img_angles:
                render_path = renders_dir / f"render_{angle}.png"
                if not render_path.exists():
                    # Try jpg
                    render_path = renders_dir / f"render_{angle}.jpg"
                    if not render_path.exists():
                        missing_renders.append(angle)

            if missing_renders:
                print(f"  Skipping {entry.name}: missing renders for angles {missing_renders}")
                continue

            # Load and validate data.json
            try:
                with open(data_json) as f:
                    data = json.load(f)

                # Extract description
                description = None
                if "annotations" in data and "description" in data["annotations"]:
                    description = data["annotations"]["description"]

                if not description:
                    print(f"  Skipping {entry.name}: no description in data.json")
                    continue

                uid = entry.name
                self.assets.append(uid)

                # Transform to objathor-compatible format for annotations.json
                ann = {
                    "uid": uid,
                    "description": description,
                    "category": data.get("annotations", {}).get("category", "unknown"),
                    "onFloor": data.get("annotations", {}).get("onFloor", True),
                    "onWall": data.get("annotations", {}).get("onWall", False),
                    "onCeiling": data.get("annotations", {}).get("onCeiling", False),
                    "onObject": data.get("annotations", {}).get("onObject", False),
                }

                # Add bounding box in thor_metadata format
                if "assetMetadata" in data and "boundingBox" in data["assetMetadata"]:
                    bbox = data["assetMetadata"]["boundingBox"]
                    ann["thor_metadata"] = {
                        "assetMetadata": {
                            "boundingBox": bbox
                        }
                    }
                    # Also add as top-level boundingBox for ObjaverseRetriever compatibility
                    ann["boundingBox"] = bbox

                self.annotations[uid] = ann

            except Exception as e:
                print(f"  Skipping {entry.name}: error loading data.json: {e}")
                continue

        print(f"Found {len(self.assets)} valid assets")

    def __len__(self) -> int:
        return len(self.assets)

    def __getitem__(self, idx: int) -> Dict[str, Union[int, str, torch.Tensor]]:
        uid = self.assets[idx]
        ann = self.annotations[uid]
        asset_path = self.asset_dir / uid

        item = {
            "idx": idx,
            "uid": uid,
            "text": ann["description"],
        }

        # Load images for each angle
        renders_dir = asset_path / "blender_renders"
        for angle in self.img_angles:
            render_path = renders_dir / f"render_{angle}.png"
            if not render_path.exists():
                render_path = renders_dir / f"render_{angle}.jpg"

            img = Image.open(render_path).convert("RGB")
            item[f"img_{angle:.1f}"] = self.image_preprocessor(img)

        return item


def generate_features(
    asset_dir: str,
    device: str,
    batch_size: int,
    num_workers: int,
):
    """Generate CLIP and SBERT features for all curated assets."""

    asset_dir = Path(asset_dir)
    features_dir = asset_dir / "features"
    features_dir.mkdir(exist_ok=True)

    # Output paths
    clip_save_path = features_dir / "clip_features.pkl"
    sbert_save_path = features_dir / "sbert_features.pkl"
    annotations_save_path = asset_dir / "annotations.json"

    # Load CLIP model
    print("Loading CLIP model (ViT-L-14, laion2b_s32b_b82k)...")
    device = torch.device(device)
    clip_model, _, clip_img_preprocessor = open_clip.create_model_and_transforms(
        model_name="ViT-L-14",
        pretrained="laion2b_s32b_b82k",
        device=device
    )
    clip_model.eval()

    # Load SBERT model
    print("Loading SBERT model (all-mpnet-base-v2)...")
    sbert_model = SentenceTransformer("all-mpnet-base-v2").to(device)

    # Create dataset
    dataset = CuratedAssetDataset(
        asset_dir=str(asset_dir),
        image_preprocessor=clip_img_preprocessor,
        img_angles=(0.0, 90.0, 270.0),  # Using 3 views like objathor
    )

    if len(dataset) == 0:
        print("No valid assets found!")
        return

    # Save annotations
    print(f"Saving annotations to {annotations_save_path}...")
    with open(annotations_save_path, "w") as f:
        json.dump(dataset.annotations, f, indent=2)
    print(f"  Saved {len(dataset.annotations)} annotations")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Generate features
    print("Generating features...")
    uids = []
    clip_img_features = []
    sbert_text_features = []

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Processing assets"):
            uids.extend(batch["uid"])

            # CLIP image features for each angle
            clip_img_features_per_angle = []
            for angle in dataset.img_angles:
                img_tensor = batch[f"img_{angle:.1f}"].to(device)
                features = clip_model.encode_image(img_tensor)
                clip_img_features_per_angle.append(features)

            # Stack angles: (batch, num_angles, dim)
            stacked = torch.stack(clip_img_features_per_angle, dim=1).cpu()
            clip_img_features.append(stacked)

            # SBERT text features
            text_features = sbert_model.encode(
                batch["text"],
                convert_to_tensor=True,
                show_progress_bar=False
            ).cpu()
            sbert_text_features.append(text_features)

    # Concatenate all batches
    clip_img_features = torch.cat(clip_img_features, dim=0).numpy().astype("float16")
    sbert_text_features = torch.cat(sbert_text_features, dim=0).numpy().astype("float16")

    # Sort by uid for consistency
    sort_indices = np.argsort(uids)
    uids = [uids[i] for i in sort_indices]
    clip_img_features = clip_img_features[sort_indices]
    sbert_text_features = sbert_text_features[sort_indices]

    # Save features
    print(f"Saving CLIP features to {clip_save_path}...")
    compress_pickle.dump(
        {
            "uids": uids,
            "img_features": clip_img_features,
        },
        str(clip_save_path),
    )

    print(f"Saving SBERT features to {sbert_save_path}...")
    compress_pickle.dump(
        {
            "uids": uids,
            "text_features": sbert_text_features,
        },
        str(sbert_save_path),
    )

    print(f"\nDone! Generated features for {len(uids)} assets.")
    print(f"  CLIP features shape: {clip_img_features.shape}")
    print(f"  SBERT features shape: {sbert_text_features.shape}")


def main():
    parser = ArgumentParser(
        description="Generate CLIP and SBERT features for curated assets"
    )
    parser.add_argument(
        "--asset_dir",
        type=str,
        default="test_asset_dir",
        help="Directory containing curated assets (default: test_asset_dir)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help=f"Torch device (default: {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for processing (default: 8)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers (default: 4)",
    )
    args = parser.parse_args()

    generate_features(
        asset_dir=args.asset_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
