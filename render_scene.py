#!/usr/bin/env python3
"""Render a final scene from layout.json and scene.json with actual 3D assets."""

import argparse
import json
from pathlib import Path

from utils.blender_render import render_existing_scene
from utils.blender_utils import reset_blender


def main():
    parser = argparse.ArgumentParser(description="Render final scene with 3D assets")
    parser.add_argument("--scene_dir", type=str, required=True,
                        help="Path to scene directory (containing layout.json and scene.json)")
    parser.add_argument("--save_blend", action="store_true",
                        help="Save .blend file for opening in Blender")
    parser.add_argument("--high_res", action="store_true",
                        help="Render at high resolution")
    parser.add_argument("--output_prefix", type=str, default="final_render",
                        help="Prefix for output files")
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)

    # Load scene.json (contains task info and asset paths)
    scene_json = scene_dir / "scene.json"
    if not scene_json.exists():
        print(f"Error: {scene_json} not found")
        return

    with open(scene_json) as f:
        task = json.load(f)

    # Load layout.json (contains final positions/rotations)
    layout_json = scene_dir / "layout.json"
    if not layout_json.exists():
        print(f"Error: {layout_json} not found")
        return

    with open(layout_json) as f:
        layout = json.load(f)

    # Convert layout to placed_assets format
    placed_assets = {}
    for asset_key, pose in layout.items():
        # asset_key is like "desk-0", need to match with scene.json keys
        placed_assets[asset_key] = {
            "position": pose["position"],
            "rotation": pose["rotation"],
        }
        # Add asset metadata from task
        if asset_key in task.get("assets", {}):
            asset_info = task["assets"][asset_key]
            if "assetMetadata" in asset_info:
                placed_assets[asset_key]["assetMetadata"] = asset_info["assetMetadata"]
            if "scale" in asset_info:
                placed_assets[asset_key]["scale"] = asset_info.get("scale", 1.0)

    print(f"Rendering scene with {len(placed_assets)} assets...")
    print(f"Assets: {list(placed_assets.keys())}")

    output_dir = scene_dir / "renders"
    output_dir.mkdir(exist_ok=True)

    # Render the scene
    # Note: save files must be full paths since render_existing_scene uses them directly
    output_images, visual_marks = render_existing_scene(
        placed_assets=placed_assets,
        task=task,
        save_dir=str(output_dir),
        add_hdri=True,
        topdown_save_file=str(output_dir / f"{args.output_prefix}_topdown.png"),
        sideview_save_file=str(output_dir / f"{args.output_prefix}_side.png"),
        add_coordinate_mark=True,
        annotate_object=True,
        annotate_wall=True,
        render_top_down=True,
        high_res=args.high_res,
        save_blend=args.save_blend,
        side_view_phi=45,
        side_view_indices=[0, 1, 2, 3],  # Render from multiple angles
    )

    reset_blender()

    print(f"\nRenderings saved to: {output_dir}")
    if args.save_blend:
        print(f"Blender file saved to: {output_dir}/scene.blend")
        print("\nTo open in Blender GUI:")
        print(f"  blender {output_dir}/scene.blend")


if __name__ == "__main__":
    main()
