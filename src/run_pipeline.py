"""
Pipeline orchestrator + command-line interface.

Runs the full flow: images -> videos -> reel. Each stage can be toggled, and
every important setting can be adjusted either in the YAML config files or with
a command-line flag (flags win over the files).

Run from the project root:

    python -m src.run_pipeline --help
    python -m src.run_pipeline                       # use config as-is
    python -m src.run_pipeline --num-images 10       # override count
    python -m src.run_pipeline --clip-length 3 --fps 30
    python -m src.run_pipeline --images-only         # skip video + reel
    python -m src.run_pipeline --no-reel             # images + clips only
    python -m src.run_pipeline --seed -1             # randomize this run
    python -m src.run_pipeline --scene "on a rooftop at night" --scene "in a library"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

# Support running as a module (python -m src.run_pipeline) or as a script.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.comfy_client import ComfyClient, ComfyUIError
    from src.generate_images import generate_images
    from src.generate_videos import generate_videos
    from src.assemble_reel import assemble_reel, FFmpegError
    from src.creative import generate_scenes
else:
    from .comfy_client import ComfyClient, ComfyUIError
    from .generate_images import generate_images
    from .generate_videos import generate_videos
    from .assemble_reel import assemble_reel, FFmpegError
    from .creative import generate_scenes

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHARACTER_CFG = PROJECT_ROOT / "config" / "character.yaml"
DEFAULT_PIPELINE_CFG = PROJECT_ROOT / "config" / "pipeline.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _apply_overrides(pipeline_cfg: dict[str, Any], args: argparse.Namespace) -> None:
    """Mutate pipeline_cfg in place based on any CLI flags that were provided."""
    pipeline_cfg.setdefault("image", {})
    pipeline_cfg.setdefault("video", {})
    pipeline_cfg.setdefault("reel", {})
    pipeline_cfg.setdefault("comfyui", {})
    pipeline_cfg.setdefault("creative", {})

    if args.num_images is not None:
        pipeline_cfg["num_images"] = args.num_images
    if args.seed is not None:
        pipeline_cfg["seed"] = args.seed
    if args.scene:
        # Explicit manual scenes on the CLI turn off creative mode for this run.
        pipeline_cfg["scene_prompts"] = list(args.scene)
        pipeline_cfg["creative"]["enabled"] = False

    # Creative-mode overrides.
    if args.theme is not None:
        pipeline_cfg["creative"]["enabled"] = True
        pipeline_cfg["creative"]["theme"] = args.theme
    if args.use_llm:
        pipeline_cfg["creative"]["use_llm"] = True
    if args.no_creative:
        pipeline_cfg["creative"]["enabled"] = False

    if args.clip_length is not None:
        pipeline_cfg["video"]["clip_length_seconds"] = args.clip_length
    if args.fps is not None:
        pipeline_cfg["video"]["fps"] = args.fps

    if args.host is not None:
        pipeline_cfg["comfyui"]["host"] = args.host
    if args.port is not None:
        pipeline_cfg["comfyui"]["port"] = args.port

    # Stage toggles.
    if args.images_only:
        pipeline_cfg["video"]["enabled"] = False
        pipeline_cfg["reel"]["enabled"] = False
    if args.no_video:
        pipeline_cfg["video"]["enabled"] = False
        pipeline_cfg["reel"]["enabled"] = False
    if args.no_reel:
        pipeline_cfg["reel"]["enabled"] = False
    if args.no_captions:
        pipeline_cfg["reel"].setdefault("captions", {})["enabled"] = False


def _existing_files(dir_path: Path, patterns: tuple[str, ...]) -> list[Path]:
    found: list[Path] = []
    for pattern in patterns:
        found.extend(sorted(dir_path.glob(pattern)))
    return found


def _resolve_creative(pipeline_cfg: dict[str, Any]) -> None:
    """
    If creative mode is on, invent scene_prompts (and captions) from the theme
    and write them into pipeline_cfg so the normal stages consume them.
    No-op when creative mode is off.
    """
    creative_cfg = pipeline_cfg.get("creative", {}) or {}
    if not creative_cfg.get("enabled", False):
        return

    theme = str(creative_cfg.get("theme", "lifestyle"))
    count = int(pipeline_cfg.get("num_images", 5))
    seed = int(pipeline_cfg.get("seed", 12345))
    if seed < 0:
        # Creative engine needs a concrete seed; derive one so runs still vary.
        import random as _r
        seed = _r.randint(0, 2**31 - 1)

    print(f"Creative mode: theme={theme!r}, generating {count} scene idea(s)"
          + (" via local LLM" if creative_cfg.get("use_llm") else " (offline engine)"))

    scenes = generate_scenes(
        theme,
        count,
        seed=seed,
        use_llm=bool(creative_cfg.get("use_llm", False)),
        llm_model=str(creative_cfg.get("llm_model", "llama3")),
    )

    pipeline_cfg["scene_prompts"] = [s.prompt for s in scenes]
    for i, s in enumerate(scenes):
        print(f"  scene {i + 1}: {s.caption} :: {s.prompt}")

    # Feed the invented captions into the reel unless the user turned captions off.
    if creative_cfg.get("auto_captions", True):
        reel_cfg = pipeline_cfg.setdefault("reel", {})
        captions_cfg = reel_cfg.setdefault("captions", {})
        if captions_cfg.get("enabled", True):
            captions_cfg["texts"] = [s.caption for s in scenes]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_pipeline",
        description="Generate consistent AI-character images/videos and assemble a Reel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--character-config", type=Path, default=DEFAULT_CHARACTER_CFG,
                   help="Path to character.yaml")
    p.add_argument("--pipeline-config", type=Path, default=DEFAULT_PIPELINE_CFG,
                   help="Path to pipeline.yaml")

    # Common per-run overrides.
    p.add_argument("--num-images", type=int, help="How many images/scenes to generate")
    p.add_argument("--seed", type=int, help="Base seed (-1 to randomize)")
    p.add_argument("--scene", action="append", metavar="PROMPT",
                   help="Scene prompt (repeat for multiple; replaces config list)")
    p.add_argument("--clip-length", type=float, help="Clip length in seconds")
    p.add_argument("--fps", type=int, help="Video fps")
    p.add_argument("--host", help="ComfyUI host")
    p.add_argument("--port", type=int, help="ComfyUI port")

    # Creative mode.
    p.add_argument("--theme", metavar="THEME",
                   help="Creative theme; auto-invents scenes + captions "
                        "(e.g. \"travel in Japan\"). Enables creative mode.")
    p.add_argument("--use-llm", action="store_true",
                   help="Use a local Ollama LLM for scene ideas (falls back to "
                        "the offline engine if unavailable)")
    p.add_argument("--no-creative", action="store_true",
                   help="Force manual scene_prompts (disable creative mode)")

    # Stage toggles.
    p.add_argument("--images-only", action="store_true",
                   help="Only generate images (skip video and reel)")
    p.add_argument("--no-video", action="store_true",
                   help="Skip video generation (and reel)")
    p.add_argument("--no-reel", action="store_true",
                   help="Generate images and clips but do not assemble a reel")
    p.add_argument("--no-captions", action="store_true",
                   help="Disable burned-in captions on the reel")

    # Reuse existing outputs to skip regenerating.
    p.add_argument("--reel-from-existing", action="store_true",
                   help="Skip image/video generation; build reel from clips already "
                        "in the clips output dir")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    try:
        character_cfg = _load_yaml(args.character_config)
        pipeline_cfg = _load_yaml(args.pipeline_config)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    _apply_overrides(pipeline_cfg, args)

    output_cfg = pipeline_cfg.get("output", {})
    clips_dir = Path(output_cfg.get("clips_dir", "output/clips"))

    video_enabled = bool(pipeline_cfg.get("video", {}).get("enabled", True))
    reel_enabled = bool(pipeline_cfg.get("reel", {}).get("enabled", True))

    # --- Shortcut: assemble a reel from clips already on disk. ---
    if args.reel_from_existing:
        clips = _existing_files(clips_dir, ("*.mp4", "*.webm", "*.gif"))
        if not clips:
            print(f"ERROR: no clips found in {clips_dir}", file=sys.stderr)
            return 1
        try:
            print(f"Assembling reel from {len(clips)} existing clip(s)...")
            reel = assemble_reel(pipeline_cfg, clips)
        except FFmpegError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"\nDone. Reel: {reel}")
        return 0

    # --- Normal flow: connect to ComfyUI. ---
    comfy_cfg = pipeline_cfg.get("comfyui", {})
    client = ComfyClient(
        host=comfy_cfg.get("host", "127.0.0.1"),
        port=int(comfy_cfg.get("port", 8188)),
    )
    try:
        print("Checking ComfyUI connection...")
        client.check_connection()
    except ComfyUIError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # --- Creative mode: invent scenes/captions before generating. ---
    try:
        _resolve_creative(pipeline_cfg)
    except (ValueError, Exception) as exc:  # noqa: BLE001 - never block on ideation
        print(f"WARNING: creative scene generation failed ({exc}); "
              f"falling back to configured scene_prompts.", file=sys.stderr)

    # --- Stage 1: images. ---
    try:
        print("\n=== Generating images ===")
        images = generate_images(client, character_cfg, pipeline_cfg)
    except (ComfyUIError, ValueError) as exc:
        print(f"ERROR during image generation: {exc}", file=sys.stderr)
        return 1
    print(f"Generated {len(images)} image(s).")

    if not video_enabled:
        print("\nVideo disabled. Done.")
        print("Images:")
        for img in images:
            print(f"  {img}")
        return 0

    # --- Stage 2: videos. ---
    try:
        print("\n=== Generating videos ===")
        clips = generate_videos(client, pipeline_cfg, images)
    except ComfyUIError as exc:
        print(f"ERROR during video generation: {exc}", file=sys.stderr)
        return 1
    print(f"Generated {len(clips)} clip(s).")

    if not reel_enabled:
        print("\nReel disabled. Done.")
        print("Clips:")
        for clip in clips:
            print(f"  {clip}")
        return 0

    # --- Stage 3: reel. ---
    try:
        print("\n=== Assembling reel ===")
        reel = assemble_reel(pipeline_cfg, clips)
    except FFmpegError as exc:
        print(f"ERROR during reel assembly: {exc}", file=sys.stderr)
        return 1

    print(f"\nDone. Reel: {reel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
