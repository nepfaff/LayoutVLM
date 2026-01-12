"""
Scene generator for open-set 3D layout generation.
Implements the pipeline from LayoutVLM Appendix B.1.
"""

import ast
import base64
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
from openai import OpenAI
from PIL import Image
from tqdm import tqdm

from .objaverse_retriever import ObjaverseRetriever
from .prompts import (
    ASSET_LIST_PROMPT,
    ASSET_VERIFICATION_PROMPT,
    LAYOUT_CRITERIA_PROMPT,
    ROOM_DIMENSIONS_PROMPT,
)


class SceneGenerator:
    """Generates scene configurations from natural language descriptions."""

    def __init__(
        self,
        objathor_dir: str,
        asset_dir: str,
        objathor_assets_dir: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        retrieval_threshold: float = 28.0,
        top_k: int = 5,
        model: str = "gpt-4o",
    ):
        """
        Initialize the scene generator.

        Args:
            objathor_dir: Path to objathor data with pre-computed features
            asset_dir: Directory to download/cache processed assets
            objathor_assets_dir: Path to pre-downloaded objathor assets (optional)
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            retrieval_threshold: CLIP similarity threshold for retrieval
            top_k: Number of candidates to consider per object
            model: OpenAI model to use for generation
        """
        self.objathor_dir = Path(objathor_dir)
        self.asset_dir = Path(asset_dir)
        self.asset_dir.mkdir(parents=True, exist_ok=True)
        self.objathor_assets_dir = Path(objathor_assets_dir) if objathor_assets_dir else None
        self.retrieval_threshold = retrieval_threshold
        self.top_k = top_k
        self.model = model

        # Initialize OpenAI client
        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var.")
        self.client = OpenAI(api_key=api_key)

        # Initialize retriever (models loaded lazily)
        self._retriever = None
        self._clip_model = None
        self._clip_tokenizer = None
        self._sbert_model = None

    def _init_retriever(self):
        """Lazily initialize the retriever with models."""
        if self._retriever is not None:
            return

        print("Loading CLIP and SBERT models...")
        import open_clip
        from sentence_transformers import SentenceTransformer

        # Load CLIP model (ViT-L-14 with laion2b weights, same as Holodeck/objathor)
        self._clip_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="laion2b_s32b_b82k"
        )
        self._clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")
        self._clip_model.eval()

        # Load SBERT model (768-dim model matching pre-computed features)
        self._sbert_model = SentenceTransformer("all-mpnet-base-v2")

        # Initialize retriever
        self._retriever = ObjaverseRetriever(
            objathor_dir=str(self.objathor_dir),
            clip_model=self._clip_model,
            clip_tokenizer=self._clip_tokenizer,
            sbert_model=self._sbert_model,
            retrieval_threshold=self.retrieval_threshold,
        )
        print("Retriever initialized")

    def generate_layout_criteria(self, task_description: str) -> str:
        """
        Generate layout criteria from task description using GPT-4.

        Args:
            task_description: Natural language description of the room

        Returns:
            Layout criteria string
        """
        prompt = LAYOUT_CRITERIA_PROMPT.format(task_description=task_description)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.7,
        )

        return response.choices[0].message.content.strip()

    def estimate_room_dimensions(self, task_description: str) -> tuple[float, float]:
        """
        Estimate room dimensions from task description using GPT-4o-mini.

        Inspired by Holodeck's guidelines:
        - Room dimensions typically 3-8m per side
        - Max area 48 m²

        Args:
            task_description: Natural language description of the room

        Returns:
            Tuple of (width, depth) in meters
        """
        # First check if dimensions are explicitly mentioned in the description
        import re
        # Match patterns like "5m x 5m", "5m by 5m", "5 x 5", etc.
        pattern = r'(\d+(?:\.\d+)?)\s*m?\s*[xX×by]+\s*(\d+(?:\.\d+)?)\s*m?'
        match = re.search(pattern, task_description)
        if match:
            width = float(match.group(1))
            depth = float(match.group(2))
            print(f"  Found explicit dimensions in description: {width}m x {depth}m")
            return width, depth

        # Use GPT-4o-mini for fast/cheap estimation
        prompt = ROOM_DIMENSIONS_PROMPT.format(task_description=task_description)

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",  # Use mini model for speed and cost
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.3,  # Lower temperature for more consistent results
        )

        content = response.choices[0].message.content.strip()

        # Parse JSON response
        try:
            # Find JSON in response
            json_match = re.search(r'\{[^{}]+\}', content)
            if json_match:
                import json
                dims = json.loads(json_match.group())
                width = float(dims.get("width", 4.0))
                depth = float(dims.get("depth", 5.0))
            else:
                print(f"Warning: Could not parse dimensions from: {content}")
                width, depth = 4.0, 5.0
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Failed to parse dimensions: {e}")
            width, depth = 4.0, 5.0

        # Clamp to reasonable bounds (Holodeck guidelines)
        width = max(2.0, min(8.0, width))
        depth = max(2.0, min(8.0, depth))

        # Ensure area doesn't exceed 48 m²
        if width * depth > 48:
            scale = (48 / (width * depth)) ** 0.5
            width *= scale
            depth *= scale

        return width, depth

    def generate_asset_list(
        self,
        task_description: str,
        layout_criteria: str,
        room_width: float,
        room_depth: float,
    ) -> dict[str, list[int]]:
        """
        Generate list of assets needed for the scene using GPT-4.

        Args:
            task_description: Natural language description of the room
            layout_criteria: Generated layout criteria
            room_width: Room width in meters
            room_depth: Room depth in meters

        Returns:
            Dictionary mapping object names to [count, num_types]
        """
        room_size = f"{room_width}m x {room_depth}m"
        prompt = ASSET_LIST_PROMPT.format(
            task_description=task_description,
            layout_criteria=layout_criteria,
            room_size=room_size,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.7,
        )

        content = response.choices[0].message.content.strip()

        # Parse the dictionary from the response
        try:
            # Try to find a dict in the response
            match = re.search(r'\{[^{}]+\}', content, re.DOTALL)
            if match:
                dict_str = match.group()
                asset_dict = ast.literal_eval(dict_str)
            else:
                asset_dict = ast.literal_eval(content)
        except (SyntaxError, ValueError) as e:
            print(f"Warning: Failed to parse asset list: {e}")
            print(f"Raw response: {content}")
            asset_dict = {}

        return asset_dict

    def _render_asset_thumbnail(self, asset_path: Path) -> Optional[str]:
        """
        Render a thumbnail of the asset for verification.

        Args:
            asset_path: Path to the GLB file

        Returns:
            Base64-encoded PNG image or None if failed
        """
        try:
            # Load the mesh
            mesh = trimesh.load(str(asset_path), force="mesh")
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)

            # Create a simple render using trimesh
            scene = trimesh.Scene(mesh)

            # Get PNG bytes
            png_bytes = scene.save_image(resolution=[512, 512])
            if png_bytes is None:
                return None

            # Encode as base64
            return base64.b64encode(png_bytes).decode("utf-8")
        except Exception as e:
            print(f"Warning: Failed to render thumbnail: {e}")
            return None

    def verify_asset(
        self,
        task_description: str,
        layout_criteria: str,
        object_description: str,
        object_looking_for: str,
        asset_path: Path,
    ) -> bool:
        """
        Verify an asset belongs in the room using GPT-4 Vision.

        Args:
            task_description: Room description
            layout_criteria: Layout criteria
            object_description: Description from asset metadata
            object_looking_for: Original object query
            asset_path: Path to the asset GLB file

        Returns:
            True if asset should be included, False otherwise
        """
        # Render thumbnail
        thumbnail_b64 = self._render_asset_thumbnail(asset_path)

        if thumbnail_b64 is None:
            # If we can't render, assume it's valid
            print(f"Warning: Could not render {asset_path}, assuming valid")
            return True

        prompt = ASSET_VERIFICATION_PROMPT.format(
            task_description=task_description,
            layout_criteria=layout_criteria,
            object_description=object_description,
            object_looking_for=object_looking_for,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{thumbnail_b64}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=10,
                temperature=0.0,
            )

            result = response.choices[0].message.content.strip().lower()
            return result == "true"
        except Exception as e:
            print(f"Warning: Verification failed: {e}, assuming valid")
            return True

    def _download_asset(self, asset_id: str) -> Optional[Path]:
        """
        Download and process an asset from Objaverse.

        Args:
            asset_id: Objaverse asset ID

        Returns:
            Path to the processed asset directory or None if failed
        """
        asset_dir = self.asset_dir / asset_id

        # Check if already downloaded/processed
        glb_path = asset_dir / f"{asset_id}.glb"
        data_path = asset_dir / "data.json"
        if glb_path.exists() and data_path.exists():
            return asset_dir

        asset_dir.mkdir(parents=True, exist_ok=True)

        try:
            import shutil
            source_path = None

            # First check if asset exists in local objathor assets directory
            if self.objathor_assets_dir is not None:
                local_asset_dir = self.objathor_assets_dir / asset_id
                local_glb = local_asset_dir / f"{asset_id}.glb"
                if local_glb.exists():
                    source_path = str(local_glb)
                    print(f"  Using local asset: {asset_id}")

            # If not found locally, download from Objaverse
            if source_path is None:
                import objaverse
                print(f"  Downloading from Objaverse: {asset_id}")
                objects = objaverse.load_objects([asset_id])
                if asset_id not in objects:
                    print(f"Warning: Could not download asset {asset_id}")
                    return None
                source_path = objects[asset_id]

            # Copy to our asset directory
            shutil.copy(source_path, glb_path)

            # Get metadata from retriever
            self._init_retriever()
            metadata = self._retriever.get_asset_metadata(asset_id)

            # Create data.json with required format
            if metadata:
                # Extract bounding box from thor_metadata if available
                thor_metadata = metadata.get("thor_metadata", {})
                asset_metadata = thor_metadata.get("assetMetadata", {})
                thor_bbox = asset_metadata.get("boundingBox", {})

                # Compute dimensions from min/max if available
                if thor_bbox.get("min") and thor_bbox.get("max"):
                    bbox_x = thor_bbox["max"]["x"] - thor_bbox["min"]["x"]
                    bbox_y = thor_bbox["max"]["y"] - thor_bbox["min"]["y"]
                    bbox_z = thor_bbox["max"]["z"] - thor_bbox["min"]["z"]
                elif thor_bbox.get("x") is not None:
                    # Direct x, y, z values (curated format)
                    bbox_x = thor_bbox["x"]
                    bbox_y = thor_bbox["y"]
                    bbox_z = thor_bbox["z"]
                else:
                    # Fallback to size if available (in cm, convert to meters)
                    size = metadata.get("size", [100, 100, 100])
                    bbox_x = size[0] / 100.0
                    bbox_y = size[1] / 100.0
                    bbox_z = size[2] / 100.0

                data = {
                    "annotations": {
                        "category": metadata.get("category", "object"),
                        "description": metadata.get("description", ""),
                        "onCeiling": metadata.get("onCeiling", False),
                        "onFloor": metadata.get("onFloor", True),
                        "onWall": metadata.get("onWall", False),
                        "onObject": metadata.get("onObject", False),
                        "frontView": metadata.get("frontView", "+x"),
                    },
                    "assetMetadata": {
                        "boundingBox": {
                            "x": bbox_x,
                            "y": bbox_y,
                            "z": bbox_z,
                        }
                    },
                }
            else:
                # Try to compute bounding box from mesh
                try:
                    mesh = trimesh.load(str(glb_path), force="mesh")
                    if isinstance(mesh, trimesh.Scene):
                        mesh = mesh.dump(concatenate=True)
                    bounds = mesh.bounds
                    size = bounds[1] - bounds[0]
                    data = {
                        "annotations": {
                            "category": "object",
                            "description": "",
                            "onCeiling": False,
                            "onFloor": True,
                            "onWall": False,
                            "onObject": False,
                            "frontView": "+x",
                        },
                        "assetMetadata": {
                            "boundingBox": {
                                "x": float(size[0]),
                                "y": float(size[1]),
                                "z": float(size[2]),
                            }
                        },
                    }
                except Exception:
                    data = {
                        "annotations": {
                            "category": "object",
                            "description": "",
                            "onCeiling": False,
                            "onFloor": True,
                            "onWall": False,
                            "onObject": False,
                            "frontView": "+x",
                        },
                        "assetMetadata": {
                            "boundingBox": {"x": 1.0, "y": 1.0, "z": 1.0}
                        },
                    }

            with open(data_path, "w") as f:
                json.dump(data, f, indent=2)

            return asset_dir

        except Exception as e:
            print(f"Warning: Failed to download asset {asset_id}: {e}")
            return None

    def generate_scene(
        self,
        task_description: str,
        room_width: Optional[float] = None,
        room_depth: Optional[float] = None,
        wall_height: float = 2.5,
        skip_verification: bool = False,
        auto_dimensions: bool = True,
    ) -> dict:
        """
        Generate a complete scene configuration.

        Args:
            task_description: Natural language description of the room
            room_width: Room width in meters (if None, will be estimated)
            room_depth: Room depth in meters (if None, will be estimated)
            wall_height: Wall height in meters
            skip_verification: Skip GPT-4 Vision verification step
            auto_dimensions: If True and dimensions not provided, estimate them

        Returns:
            Scene configuration dictionary compatible with LayoutVLM
        """
        print(f"Generating scene for: {task_description}")

        # Estimate room dimensions if not provided
        if room_width is None or room_depth is None:
            if auto_dimensions:
                print("Step 0: Estimating room dimensions...")
                est_width, est_depth = self.estimate_room_dimensions(task_description)
                room_width = room_width if room_width is not None else est_width
                room_depth = room_depth if room_depth is not None else est_depth
                print(f"  Estimated dimensions: {room_width:.1f}m x {room_depth:.1f}m")
            else:
                # Use defaults
                room_width = room_width if room_width is not None else 4.0
                room_depth = room_depth if room_depth is not None else 5.0

        # Step 1: Generate layout criteria
        print("Step 1: Generating layout criteria...")
        layout_criteria = self.generate_layout_criteria(task_description)
        print(f"  Layout criteria: {layout_criteria}")

        # Step 2: Generate asset list
        print("Step 2: Generating asset list...")
        asset_list = self.generate_asset_list(
            task_description, layout_criteria, room_width, room_depth
        )
        print(f"  Assets: {list(asset_list.keys())}")

        # Step 3: Initialize retriever
        print("Step 3: Initializing retriever...")
        self._init_retriever()

        # Step 4: Retrieve and verify assets
        print("Step 4: Retrieving assets from Objaverse...")
        scene_assets = {}
        asset_instance_idx = 0

        for object_name, (count, num_types) in tqdm(asset_list.items()):
            # Query the retriever
            candidates = self._retriever.retrieve([object_name])

            if not candidates:
                print(f"  Warning: No candidates found for '{object_name}'")
                continue

            # Take top candidates
            top_candidates = candidates[: self.top_k * num_types]

            # Verify and download assets
            verified_assets = []
            for asset_id, score in top_candidates:
                if len(verified_assets) >= num_types:
                    break

                # Download asset
                asset_path = self._download_asset(asset_id)
                if asset_path is None:
                    continue

                glb_path = asset_path / f"{asset_id}.glb"

                # Verify with GPT-4 Vision
                if not skip_verification:
                    metadata = self._retriever.get_asset_metadata(asset_id)
                    obj_desc = metadata.get("description", "") if metadata else ""

                    is_valid = self.verify_asset(
                        task_description=task_description,
                        layout_criteria=layout_criteria,
                        object_description=obj_desc,
                        object_looking_for=object_name,
                        asset_path=glb_path,
                    )

                    if not is_valid:
                        print(f"  Rejected: {asset_id} for '{object_name}'")
                        continue

                verified_assets.append(asset_id)
                print(f"  Accepted: {asset_id} for '{object_name}'")

            # Add verified assets to scene
            for i, asset_id in enumerate(verified_assets):
                # For each type, add 'count' instances
                instances_per_type = max(1, count // num_types)
                if i < count % num_types:
                    instances_per_type += 1

                for j in range(instances_per_type):
                    scene_assets[f"{asset_id}-{asset_instance_idx}"] = {}
                    asset_instance_idx += 1

        # Step 5: Generate scene configuration
        print("Step 5: Generating scene configuration...")
        scene_config = {
            "task_description": task_description,
            "layout_criteria": layout_criteria,
            "boundary": {
                "floor_vertices": [
                    [0, 0, 0],
                    [room_width, 0, 0],
                    [room_width, room_depth, 0],
                    [0, room_depth, 0],
                ],
                "wall_height": wall_height,
            },
            "assets": scene_assets,
        }

        print(f"Generated scene with {len(scene_assets)} asset instances")
        return scene_config
