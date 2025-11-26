#!/usr/bin/env python3
"""
Open-set scene generation pipeline for LayoutVLM.
Generates scene configurations from natural language descriptions,
retrieves assets from Objaverse, and runs the full layout optimization.

Based on Appendix B.1 of the LayoutVLM paper.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate 3D scenes from natural language descriptions"
    )
    parser.add_argument(
        "--task_description",
        type=str,
        required=True,
        help="Natural language description of the room (e.g., 'a cozy bedroom with a queen bed, 4m x 5m')",
    )
    parser.add_argument(
        "--room_width",
        type=float,
        default=4.0,
        help="Room width in meters (default: 4.0)",
    )
    parser.add_argument(
        "--room_depth",
        type=float,
        default=5.0,
        help="Room depth in meters (default: 5.0)",
    )
    parser.add_argument(
        "--wall_height",
        type=float,
        default=2.5,
        help="Wall height in meters (default: 2.5)",
    )
    parser.add_argument(
        "--objathor_dir",
        type=str,
        required=True,
        help="Path to objathor data directory with pre-computed CLIP/SBERT features",
    )
    parser.add_argument(
        "--asset_dir",
        type=str,
        default="./objaverse_processed",
        help="Directory to download/cache processed assets (default: ./objaverse_processed)",
    )
    parser.add_argument(
        "--objathor_assets_dir",
        type=str,
        default=None,
        help="Path to pre-downloaded objathor assets directory (optional, will download from Objaverse if not provided)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results/generated_scene",
        help="Output directory for results (default: ./results/generated_scene)",
    )
    parser.add_argument(
        "--retrieval_threshold",
        type=float,
        default=28.0,
        help="CLIP similarity threshold for retrieval (default: 28.0)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of asset candidates per object type (default: 5)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model for generation (default: gpt-4o)",
    )
    parser.add_argument(
        "--skip_verification",
        action="store_true",
        help="Skip GPT-4 Vision verification of retrieved assets",
    )
    parser.add_argument(
        "--scene_only",
        action="store_true",
        help="Only generate scene JSON, do not run LayoutVLM optimization",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Import here to avoid slow import on --help
    from src.layoutvlm.scene_generator import SceneGenerator

    # Initialize scene generator
    print("=" * 60)
    print("LayoutVLM Open-Set Scene Generation Pipeline")
    print("=" * 60)
    print(f"Task: {args.task_description}")
    print(f"Room size: {args.room_width}m x {args.room_depth}m x {args.wall_height}m")
    print(f"Output: {save_dir}")
    print("=" * 60)

    generator = SceneGenerator(
        objathor_dir=args.objathor_dir,
        asset_dir=args.asset_dir,
        objathor_assets_dir=args.objathor_assets_dir,
        retrieval_threshold=args.retrieval_threshold,
        top_k=args.top_k,
        model=args.model,
    )

    # Generate scene configuration
    scene_config = generator.generate_scene(
        task_description=args.task_description,
        room_width=args.room_width,
        room_depth=args.room_depth,
        wall_height=args.wall_height,
        skip_verification=args.skip_verification,
    )

    # Save scene configuration
    scene_json_path = save_dir / "scene.json"
    with open(scene_json_path, "w") as f:
        json.dump(scene_config, f, indent=2)
    print(f"\nScene configuration saved to: {scene_json_path}")

    if args.scene_only:
        print("\n--scene_only specified, skipping LayoutVLM optimization")
        print("To run optimization manually:")
        print(f"  python main.py --scene_json_file {scene_json_path} --asset_dir {args.asset_dir} --save_dir {save_dir}")
        return

    # Run LayoutVLM optimization
    print("\n" + "=" * 60)
    print("Running LayoutVLM Layout Optimization")
    print("=" * 60)

    from src.layoutvlm.layoutvlm import LayoutVLM
    from main import prepare_task_assets

    # Prepare assets for LayoutVLM
    scene_config = prepare_task_assets(scene_config, args.asset_dir)

    # Check if we have any assets
    if not scene_config.get("assets"):
        print("Warning: No assets in scene configuration. Cannot run optimization.")
        return

    # Initialize and run LayoutVLM
    layout_solver = LayoutVLM(
        mode="one_shot",
        save_dir=str(save_dir),
        asset_source="objaverse",
    )

    layout = layout_solver.solve(scene_config)

    # Save layout
    layout_json_path = save_dir / "layout.json"
    with open(layout_json_path, "w") as f:
        json.dump(layout, f, indent=2)

    print(f"\nLayout saved to: {layout_json_path}")
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"Scene config: {scene_json_path}")
    print(f"Final layout: {layout_json_path}")


if __name__ == "__main__":
    main()
