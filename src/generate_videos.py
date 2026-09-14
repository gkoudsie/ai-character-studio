"""
Video generation stage.

Animates each generated still image into a short clip using Stable Video
Diffusion (SVD) via ComfyUI. Clip length and fps come from config; the number
of frames is derived as clip_length_seconds * fps (clamped to SVD's practical
range).

SVD is trained around ~14-25 frames, so we cap frames to keep results stable.
For longer clips, the reel assembly stage can slow/loop, but here we respect the
model's sweet spot.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .comfy_client import ComfyClient, save_bytes

WORKFLOW_PATH = Path(__file__).resolve().parent.parent / "workflows" / "image_to_video.json"

# SVD practical frame bounds.
_MIN_FRAMES = 14
_MAX_FRAMES = 25

# SVD conditioning likes dimensions that are multiples of 64.
_SVD_WIDTH = 576
_SVD_HEIGHT = 1024


def _load_template() -> dict[str, Any]:
    with WORKFLOW_PATH.open("r", encoding="utf-8") as fh:
        wf = json.load(fh)
    wf.pop("_comment", None)
    return wf


def _frames_for(clip_seconds: float, fps: int) -> int:
    raw = int(round(clip_seconds * fps))
    return max(_MIN_FRAMES, min(_MAX_FRAMES, raw))


def _motion_bucket(motion: float) -> int:
    """
    Map a friendly 0..2 'motion' scalar to SVD's motion_bucket_id (roughly 1..255).
    1.0 -> ~127 (default amount of movement).
    """
    bucket = int(round(127 * max(0.0, min(2.0, motion))))
    return max(1, min(255, bucket))


def _build_workflow(
    *,
    source_name: str,
    video_cfg: dict[str, Any],
    seed: int,
    filename_prefix: str,
) -> dict[str, Any]:
    wf = copy.deepcopy(_load_template())
    fps = int(video_cfg.get("fps", 24))
    frames = _frames_for(float(video_cfg.get("clip_length_seconds", 4)), fps)

    wf["15"]["inputs"]["ckpt_name"] = video_cfg.get("model", "svd_xt.safetensors")
    wf["23"]["inputs"]["image"] = source_name

    wf["12"]["inputs"]["width"] = _SVD_WIDTH
    wf["12"]["inputs"]["height"] = _SVD_HEIGHT
    wf["12"]["inputs"]["video_frames"] = frames
    wf["12"]["inputs"]["fps"] = fps
    wf["12"]["inputs"]["motion_bucket_id"] = _motion_bucket(float(video_cfg.get("motion", 1.0)))

    wf["14"]["inputs"]["seed"] = int(seed)

    wf["30"]["inputs"]["frame_rate"] = fps
    wf["30"]["inputs"]["filename_prefix"] = filename_prefix

    return wf


def generate_videos(
    client: ComfyClient,
    pipeline_cfg: dict[str, Any],
    image_paths: list[Path],
) -> list[Path]:
    """
    Animate each image in image_paths into a clip.

    Returns the list of saved clip paths, in the same order as the images.
    """
    video_cfg = pipeline_cfg.get("video", {})
    clips_dir = Path(pipeline_cfg.get("output", {}).get("clips_dir", "output/clips"))
    base_seed = int(pipeline_cfg.get("seed", 0))
    if base_seed < 0:
        base_seed = 0

    saved: list[Path] = []
    for idx, image_path in enumerate(image_paths):
        if not image_path.is_file():
            print(f"  Skipping missing image: {image_path}")
            continue

        print(f"[clip {idx + 1}/{len(image_paths)}] animating {image_path.name}")
        source_name = client.upload_image(image_path)
        prefix = f"clip_{idx:03d}"

        workflow = _build_workflow(
            source_name=source_name,
            video_cfg=video_cfg,
            seed=base_seed + idx,
            filename_prefix=prefix,
        )

        outputs = client.run(workflow)
        # Prefer an mp4 output; fall back to whatever came back first.
        filename, data = next(
            ((n, d) for n, d in outputs if n.lower().endswith(".mp4")),
            outputs[0],
        )
        ext = Path(filename).suffix or ".mp4"
        dest = clips_dir / f"{prefix}{ext}"
        save_bytes(data, dest)
        saved.append(dest)
        print(f"    saved -> {dest}")

    return saved
