#!/usr/bin/env python3
"""Run scene generation for all prompts in a CSV file."""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

# Hardcoded paths
CSV_FILE = "/home/ubuntu/LayoutVLM/prompts.csv"
RESULTS_DIR = "/home/ubuntu/LayoutVLM/results_curated"

# Asset source configurations
ASSET_CONFIGS = {
    "curated": {
        # Note: Only 126 of 675 curated assets have matching features in 2023_09_23
        "objathor_dir": "/home/ubuntu/LayoutVLM/test_asset_dir",
        "objathor_assets_dir": "/home/ubuntu/LayoutVLM/test_asset_dir",
        "description": "675 curated assets (126 with features from 2023_09_23)",
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
        "--asset_source",
        type=str,
        required=True,
        choices=["curated", "full"],
        help="Asset source: 'curated' (675 assets) or 'full' (50K assets)"
    )
    parser.add_argument("--start_id", type=int, default=None, help="Start from this ID (inclusive)")
    parser.add_argument("--end_id", type=int, default=None, help="End at this ID (inclusive)")
    args = parser.parse_args()

    # Get asset config
    config = ASSET_CONFIGS[args.asset_source]
    objathor_dir = config["objathor_dir"]
    objathor_assets_dir = config["objathor_assets_dir"]

    print(f"Using asset source: {args.asset_source}")
    print(f"  {config['description']}")
    print(f"  objathor_dir: {objathor_dir}")
    print(f"  objathor_assets_dir: {objathor_assets_dir}")

    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(CSV_FILE, "r") as f:
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

        print(f"\n{'='*60}")
        print(f"Scene {prompt_id} ({i+1}/{total}): {description[:50]}...")
        print(f"{'='*60}\n")

        cmd = [
            "xvfb-run", "-a", "uv", "run", "python", "generate_scene.py",
            "--task_description", description,
            "--objathor_dir", objathor_dir,
            "--objathor_assets_dir", objathor_assets_dir,
            "--save_dir", str(save_dir),
        ]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Warning: Scene {prompt_id} failed with code {result.returncode}")

    print(f"\n{'='*60}")
    print(f"All scenes completed! Results in: {results_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
