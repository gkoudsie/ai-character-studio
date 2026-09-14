"""
Image generation stage.

Takes the character definition + a list of scene prompts and produces one image
per scene via ComfyUI. Handles the optional pieces gracefully:
  - No reference image  -> IPAdapter nodes are removed, KSampler reads the model
                           straight from the LoRA (or checkpoint).
  - No LoRA             -> LoraLoader node is removed, everything downstream reads
                           from the checkpoint instead.

Seeds are derived as base_seed + scene_index so the run is reproducible while
each scene still differs. This is a big part of keeping the character consistent.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

from .comfy_client import ComfyClient, save_bytes

WORKFLOW_PATH = Path(__file__).resolve().parent.parent / "workflows" / "image_generation.json"


def _load_template() -> dict[str, Any]:
    with WORKFLOW_PATH.open("r", encoding="utf-8") as fh:
        wf = json.load(fh)
    wf.pop("_comment", None)
    return wf


def _resolve_scene_prompts(pipeline_cfg: dict[str, Any]) -> list[str]:
    """Pick exactly num_images prompts, cycling the list if we need more."""
    prompts = list(pipeline_cfg.get("scene_prompts", []))
    num = int(pipeline_cfg.get("num_images", len(prompts)))
    if not prompts:
        raise ValueError("No scene_prompts defined in pipeline config.")
    if num <= 0:
        raise ValueError("num_images must be a positive integer.")
    return [prompts[i % len(prompts)] for i in range(num)]


def _base_seed(pipeline_cfg: dict[str, Any]) -> int:
    seed = int(pipeline_cfg.get("seed", -1))
    if seed < 0:
        seed = random.randint(0, 2**31 - 1)
    return seed


def _build_workflow(
    *,
    positive: str,
    negative: str,
    seed: int,
    character_cfg: dict[str, Any],
    image_cfg: dict[str, Any],
    reference_name: str | None,
    filename_prefix: str,
) -> dict[str, Any]:
    """Fill the template, wiring nodes based on which optional parts are active."""
    wf = copy.deepcopy(_load_template())
    models = character_cfg.get("models", {})

    # ---- checkpoint ----
    wf["4"]["inputs"]["ckpt_name"] = models.get("checkpoint")

    # ---- LoRA (optional) ----
    lora_name = models.get("lora")
    if lora_name:
        wf["10"]["inputs"]["lora_name"] = lora_name
        strength = float(models.get("lora_strength", 0.8))
        wf["10"]["inputs"]["strength_model"] = strength
        wf["10"]["inputs"]["strength_clip"] = strength
        model_source = ["10", 0]
        clip_source = ["10", 1]
    else:
        # Remove the LoRA node and read directly from the checkpoint.
        wf.pop("10", None)
        model_source = ["4", 0]
        clip_source = ["4", 1]

    # ---- prompts (CLIP text encode reads from the resolved clip source) ----
    wf["6"]["inputs"]["text"] = positive
    wf["6"]["inputs"]["clip"] = clip_source
    wf["7"]["inputs"]["text"] = negative
    wf["7"]["inputs"]["clip"] = clip_source

    # ---- IPAdapter character reference (optional) ----
    if reference_name:
        wf["20"]["inputs"]["image"] = reference_name
        wf["21"]["inputs"]["clip_name"] = models.get("clip_vision")
        wf["22"]["inputs"]["ipadapter_file"] = models.get("ipadapter")
        wf["23"]["inputs"]["weight"] = float(character_cfg.get("reference_strength", 0.7))
        wf["23"]["inputs"]["model"] = model_source
        sampler_model_source = ["23", 0]
    else:
        # No reference: drop all IPAdapter-related nodes and feed KSampler direct.
        for node_id in ("20", "21", "22", "23"):
            wf.pop(node_id, None)
        sampler_model_source = model_source

    # ---- latent size ----
    wf["5"]["inputs"]["width"] = int(image_cfg.get("width", 832))
    wf["5"]["inputs"]["height"] = int(image_cfg.get("height", 1216))

    # ---- sampler ----
    wf["3"]["inputs"]["model"] = sampler_model_source
    wf["3"]["inputs"]["seed"] = int(seed)
    wf["3"]["inputs"]["steps"] = int(image_cfg.get("steps", 30))
    wf["3"]["inputs"]["cfg"] = float(image_cfg.get("cfg", 7.0))
    wf["3"]["inputs"]["sampler_name"] = image_cfg.get("sampler", "euler")
    wf["3"]["inputs"]["scheduler"] = image_cfg.get("scheduler", "normal")

    # ---- output ----
    wf["9"]["inputs"]["filename_prefix"] = filename_prefix

    return wf


def generate_images(
    client: ComfyClient,
    character_cfg: dict[str, Any],
    pipeline_cfg: dict[str, Any],
) -> list[Path]:
    """
    Generate one image per resolved scene prompt.

    Returns the list of saved image paths, in scene order.
    """
    scene_prompts = _resolve_scene_prompts(pipeline_cfg)
    base_seed = _base_seed(pipeline_cfg)
    character_prompt = character_cfg.get("character_prompt", "").strip()
    negative = character_cfg.get("negative_prompt", "").strip()
    image_cfg = pipeline_cfg.get("image", {})

    images_dir = Path(pipeline_cfg.get("output", {}).get("images_dir", "output/images"))

    # Upload the first reference image once (IPAdapter uses a single reference here).
    reference_name: str | None = None
    refs = character_cfg.get("reference_images") or []
    project_root = Path(__file__).resolve().parent.parent
    for ref in refs:
        ref_path = (project_root / ref) if not Path(ref).is_absolute() else Path(ref)
        if ref_path.is_file():
            reference_name = client.upload_image(ref_path)
            print(f"  Using reference image: {ref_path.name}")
            break
        else:
            print(f"  Reference image not found, skipping: {ref_path}")
    if refs and reference_name is None:
        print("  No valid reference images found - generating from prompt only.")

    saved: list[Path] = []
    for idx, scene in enumerate(scene_prompts):
        positive = f"{character_prompt}, {scene}" if character_prompt else scene
        seed = base_seed + idx
        prefix = f"image_{idx:03d}"
        print(f"[image {idx + 1}/{len(scene_prompts)}] seed={seed} :: {scene}")

        workflow = _build_workflow(
            positive=positive,
            negative=negative,
            seed=seed,
            character_cfg=character_cfg,
            image_cfg=image_cfg,
            reference_name=reference_name,
            filename_prefix=prefix,
        )

        outputs = client.run(workflow)
        # Take the first image output for this scene.
        filename, data = outputs[0]
        ext = Path(filename).suffix or ".png"
        dest = images_dir / f"{prefix}{ext}"
        save_bytes(data, dest)
        saved.append(dest)
        print(f"    saved -> {dest}")

    return saved
