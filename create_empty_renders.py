#!/usr/bin/env python3
"""
Create empty room placeholder renders for scenes that failed asset retrieval.
These allow SceneEval to process them without crashing.
"""

import argparse
import json
from pathlib import Path

def create_empty_room_render(scene_dir: Path, force: bool = False):
    """Create a minimal empty room render for a failed scene.

    Args:
        scene_dir: Path to scene directory
        force: If True, create placeholder even if assets exist
    """
    scene_json = scene_dir / "scene.json"
    if not scene_json.exists():
        print(f"  No scene.json found, skipping")
        return False

    with open(scene_json) as f:
        scene = json.load(f)

    # Check if assets are empty (unless force)
    if not force and scene.get("assets"):
        print(f"  Scene has assets, use --force to create placeholder anyway")
        return False

    # Create minimal group_0 directory with placeholder render
    group_dir = scene_dir / "group_0"
    group_dir.mkdir(exist_ok=True)

    # Create a minimal top_down_rendering.png
    from PIL import Image, ImageDraw, ImageFont

    # Get room dimensions from boundary
    boundary = scene.get("boundary", {})
    floor_vertices = boundary.get("floor_vertices", [[0,0,0], [4,0,0], [4,4,0], [0,4,0]])

    # Create image with room outline
    img = Image.new('RGB', (512, 512), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)

    # Draw room boundary
    if floor_vertices:
        # Scale vertices to fit image
        xs = [v[0] for v in floor_vertices]
        ys = [v[1] for v in floor_vertices]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        scale = 400 / max(max_x - min_x + 0.1, max_y - min_y + 0.1)
        offset_x = 56 + (400 - (max_x - min_x) * scale) / 2
        offset_y = 56 + (400 - (max_y - min_y) * scale) / 2

        points = [(offset_x + (v[0] - min_x) * scale,
                   offset_y + (v[1] - min_y) * scale) for v in floor_vertices]
        draw.polygon(points, outline=(100, 100, 100), fill=(220, 220, 220))

    # Add text
    draw.text((256, 480), "Empty Room (No Assets)", fill=(150, 150, 150), anchor="mm")

    render_path = group_dir / "top_down_rendering.png"
    img.save(render_path)

    # Create minimal layout.json (empty - no objects)
    layout = {}
    layout_path = scene_dir / "layout.json"
    with open(layout_path, 'w') as f:
        json.dump(layout, f, indent=2)

    # Create complete_sandbox_program.py (minimal)
    program_path = scene_dir / "complete_sandbox_program.py"
    if not program_path.exists():
        with open(program_path, 'w') as f:
            f.write("# Empty scene - no assets retrieved\n")
            f.write("# This is a placeholder for SceneEval compatibility\n")

    # Create grouping files
    grouping = {"group_0": []}
    with open(scene_dir / "grouping.json", 'w') as f:
        json.dump(grouping, f, indent=2)

    with open(scene_dir / "grouping_0.txt", 'w') as f:
        f.write("# Empty scene - no assets retrieved\n")

    print(f"  Created placeholder files:")
    print(f"    - {render_path}")
    print(f"    - {layout_path}")
    print(f"    - {scene_dir / 'grouping.json'}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Create empty room placeholders for failed scenes")
    parser.add_argument("--results_dir", type=str, required=True, help="Results directory")
    parser.add_argument("--scenes", type=str, nargs="+", help="Scene IDs to process (e.g., 168 177 184)")
    parser.add_argument("--dry_run", action="store_true", help="Just list what would be done")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    for scene_id in args.scenes:
        scene_dir = results_dir / f"scene_{int(scene_id):03d}"
        print(f"\nProcessing scene_{scene_id}...")

        if not scene_dir.exists():
            print(f"  Directory not found: {scene_dir}")
            continue

        # Check if already has render
        existing_renders = list(scene_dir.glob("group_*/top_down_rendering.png"))
        if existing_renders:
            print(f"  Already has render, skipping")
            continue

        if args.dry_run:
            print(f"  Would create placeholder render")
        else:
            create_empty_room_render(scene_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
