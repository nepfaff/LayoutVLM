#!/usr/bin/env python3
"""Run scene generation for all prompts in a CSV file."""

import argparse
import csv
import subprocess
from pathlib import Path

# Hardcoded paths
CSV_FILE = "/home/ubuntu/LayoutVLM/prompts.csv"
OBJATHOR_DIR = "/home/ubuntu/SceneEval/_data/2023_09_23"
OBJATHOR_ASSETS_DIR = "/home/ubuntu/SceneEval/_data/objathor-assets"
RESULTS_DIR = "/home/ubuntu/LayoutVLM/results"

def main():
    parser = argparse.ArgumentParser(description="Run scene generation from CSV prompts")
    parser.add_argument("--start_id", type=int, default=None, help="Start from this ID (inclusive)")
    parser.add_argument("--end_id", type=int, default=None, help="End at this ID (inclusive)")
    args = parser.parse_args()

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
            "--objathor_dir", OBJATHOR_DIR,
            "--objathor_assets_dir", OBJATHOR_ASSETS_DIR,
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
