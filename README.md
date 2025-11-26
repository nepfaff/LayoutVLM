# LayoutVLM

<div align="left">
    <a href="https://ai.stanford.edu/~sunfanyun/layoutvlm"><img src="https://img.shields.io/badge/🌐 Website-Visit-orange"></a>
    <a href=""><img src="https://img.shields.io/badge/arXiv-PDF-blue"></a>
</div>

<br>

## Installation

1. Clone this repository

2. Install [uv](https://docs.astral.sh/uv/) (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Install system dependencies (for headless rendering):
```bash
sudo apt-get install -y xvfb libgl1-mesa-glx libegl1-mesa
```

4. Sync dependencies (Python 3.10):
```bash
uv sync
```

5. (Optional, requires CUDA) Install Rotated IOU Loss for more accurate bounding box overlap:
```bash
uv add setuptools
cd third_party/Rotated_IoU/cuda_op
uv run python setup.py install
cd ../../..
```
Without this, the optimizer uses a simplified fallback that still works but may be less accurate.

## Data preprocessing
1. Download the dataset https://drive.google.com/file/d/1WGbj8gWn-f-BRwqPKfoY06budBzgM0pu/view?usp=sharing
2. Unzip it.

Refer to https://github.com/allenai/Holodeck and https://github.com/allenai/objathor for how we preprocess Objaverse assets.

## Usage

1. Prepare a scene configuration JSON file of Objaverse assets with the following structure:
```json
{
    "task_description": ...,
    "layout_criteria": ...,
    "boundary": {
        "floor_vertices": [[x1, y1, z1], [x2, y2, z2], ...],
        "wall_height": height
    },
    "assets": {
        "asset_id": {
            "path": "path/to/asset.glb",
            "assetMetadata": {
                "boundingBox": {
                    "x": width,
                    "y": depth,
                    "z": height
                }
            }
        }
    }
}
```

2. Run LayoutVLM:
```bash
uv run python main.py --scene_json_file path/to/scene.json --openai_api_key your_api_key
```

## Open-Set Scene Generation

Generate scenes from natural language descriptions using the pipeline from Appendix B.1:

```bash
uv run python generate_scene.py \
    --task_description "a cozy bedroom with a queen bed and nightstands, 4m x 5m" \
    --objathor_dir /path/to/objathor-assets \
    --room_width 4.0 \
    --room_depth 5.0 \
    --save_dir ./results/my_bedroom
```

### Headless Server Usage

On headless servers (no display), use `xvfb-run` for thumbnail rendering during GPT-4 Vision verification:

```bash
xvfb-run -a uv run python generate_scene.py \
    --task_description "a cozy bedroom with a queen bed" \
    --objathor_dir /path/to/objathor-data \
    --objathor_assets_dir /path/to/objathor-data/assets \
    --save_dir ./results/my_bedroom
```

Example:
```bash
xvfb-run -a uv run python generate_scene.py \
    --task_description "a cozy bedroom with a queen bed, 4m x 5m" \
    --objathor_dir /home/ubuntu/SceneEval/_data/2023_09_23 \
    --objathor_assets_dir /home/ubuntu/SceneEval/_data/2023_09_23/assets \
    --room_width 4.0 \
    --room_depth 5.0 \
    --save_dir ./results/my_bedroom
```

Install xvfb if needed: `sudo apt-get install xvfb`

Alternatively, use `--skip_verification` to skip the GPT-4 Vision verification step (assets are accepted based on CLIP+SBERT scores alone).

This will:
1. Generate layout criteria from the task description using GPT-4o
2. Generate a list of required objects using GPT-4o
3. Retrieve matching 3D assets from Objaverse using CLIP+SBERT embeddings
4. Verify each asset fits the room using GPT-4o Vision
5. Run LayoutVLM to optimize the final layout

### Prerequisites for Scene Generation

You need pre-computed CLIP/SBERT embeddings for Objaverse assets. These can be obtained from:
- [objathor](https://github.com/allenai/objathor) - download the asset annotations and features

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--task_description` | Natural language room description | Required |
| `--objathor_dir` | Path to objathor data with features | Required |
| `--room_width` | Room width in meters | 4.0 |
| `--room_depth` | Room depth in meters | 5.0 |
| `--wall_height` | Wall height in meters | 2.5 |
| `--asset_dir` | Directory to cache downloaded assets | ./objaverse_processed |
| `--save_dir` | Output directory | ./results/generated_scene |
| `--retrieval_threshold` | CLIP similarity threshold | 28.0 |
| `--top_k` | Candidates per object type | 5 |
| `--skip_verification` | Skip GPT-4 Vision verification | False |
| `--scene_only` | Only generate scene JSON, skip layout optimization | False |

## Output
The script will generate a layout.json file in the specified save directory containing the optimized positions and orientations of all assets in the scene.

## BibTeX
```bibtex
@inproceedings{sun2025layoutvlm,
  title={Layoutvlm: Differentiable optimization of 3d layout via vision-language models},
  author={Sun, Fan-Yun and Liu, Weiyu and Gu, Siyi and Lim, Dylan and Bhat, Goutam and Tombari, Federico and Li, Manling and Haber, Nick and Wu, Jiajun},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={29469--29478},
  year={2025}
}
```
