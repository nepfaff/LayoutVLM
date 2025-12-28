#!/usr/bin/env python3
"""Run scene generation for all prompts in a CSV file."""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

# Hardcoded paths
CSV_FILE = str(Path.home() / "SceneEval/input/annotations.csv")
RESULTS_DIR = "/home/ubuntu/LayoutVLM/results"

# Asset source configurations
ASSET_CONFIGS = {
    "curated": {
        # Features generated with scripts/generate_curated_features.py
        "objathor_dir": "/home/ubuntu/LayoutVLM/test_asset_dir",
        "objathor_assets_dir": "/home/ubuntu/LayoutVLM/test_asset_dir",
        "description": "674 curated assets with custom CLIP/SBERT features",
    },
    "full": {
        "objathor_dir": "/home/ubuntu/SceneEval/_data/objathor-assets/2023_09_23",
        "objathor_assets_dir": "/home/ubuntu/SceneEval/_data/objathor-assets",
        "description": "Full 50K assets from objathor-assets",
    },
}

def main():
    parser = argparse.ArgumentParser(description="Run scene generation from CSV prompts")
    parser.add_argument(
        "--csv_file",
        type=str,
        default=CSV_FILE,
        help=f"Path to CSV file with prompts (default: {CSV_FILE})"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=RESULTS_DIR,
        help=f"Directory to save results (default: {RESULTS_DIR})"
    )
    parser.add_argument(
        "--asset_source",
        type=str,
        choices=["curated", "full"],
        default="curated",
        help="Asset source: 'curated' (674 assets) or 'full' (50K assets)"
    )
    parser.add_argument("--start_id", type=int, default=None, help="Start from this ID (inclusive)")
    parser.add_argument("--end_id", type=int, default=None, help="End at this ID (inclusive)")
    parser.add_argument(
        "--mode",
        type=str,
        default="finetuned",
        choices=["one_shot", "finetuned", "no_image", "no_visual_coordinate", "no_visual_assetname", "no_visual_mark"],
        help="LayoutVLM mode: 'finetuned' (iterative with visual feedback, default), 'one_shot' (single pass)",
    )
    parser.add_argument(
        "--render_final",
        action="store_true",
        help="Render final scene with actual 3D assets using Blender after each scene",
    )
    parser.add_argument(
        "--save_blend",
        action="store_true",
        help="Save .blend files when rendering (requires --render_final)",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip scenes that already have results (save_dir exists and contains files)",
    )
    args = parser.parse_args()

    # Get asset config
    config = ASSET_CONFIGS[args.asset_source]
    objathor_dir = config["objathor_dir"]
    objathor_assets_dir = config["objathor_assets_dir"]

    print(f"Using asset source: {args.asset_source}")
    print(f"  {config['description']}")
    print(f"  objathor_dir: {objathor_dir}")
    print(f"  objathor_assets_dir: {objathor_assets_dir}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(args.csv_file, "r") as f:
        reader = csv.DictReader(f)
        prompts = list(reader)

    total = len(prompts)
    for i, row in enumerate(prompts):
        prompt_id = int(row["ID"])

        # Filter by ID range if specified
        if args.start_id is not None and prompt_id < args.start_id:
            continue
        if args.end_id is not None and prompt_id > args.end_id:
            continue

        description = row["Description"]
        save_dir = results_dir / f"scene_{prompt_id:03d}"

        # Skip if results already exist
        if args.skip_existing and save_dir.exists() and any(save_dir.iterdir()):
            print(f"Skipping scene {prompt_id} (results already exist in {save_dir})")
            continue

        print(f"\n{'='*60}")
        print(f"Scene {prompt_id} ({i+1}/{total}): {description[:50]}...")
        print(f"{'='*60}\n")

        cmd = [
            "xvfb-run", "-a", "uv", "run", "python", "generate_scene.py",
            "--task_description", description,
            "--objathor_dir", objathor_dir,
            "--objathor_assets_dir", objathor_assets_dir,
            "--save_dir", str(save_dir),
            "--mode", args.mode,
        ]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Warning: Scene {prompt_id} failed with code {result.returncode}")
            continue

        # Render final scene with actual 3D assets
        if args.render_final:
            print(f"\nRendering final scene with 3D assets...")
            render_cmd = [
                "xvfb-run", "-a", "uv", "run", "python", "render_scene.py",
                "--scene_dir", str(save_dir),
            ]
            if args.save_blend:
                render_cmd.append("--save_blend")
            render_result = subprocess.run(render_cmd)
            if render_result.returncode != 0:
                print(f"Warning: Render for scene {prompt_id} failed with code {render_result.returncode}")

    print(f"\n{'='*60}")
    print(f"All scenes completed! Results in: {results_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
