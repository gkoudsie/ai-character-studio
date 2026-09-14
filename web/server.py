"""
Mobile/cloud web backend (FastAPI).

Serves a phone-friendly PWA and drives the generation pipeline. Designed to run
in two ways:

  * Local:  dashboard on your PC, ComfyUI on your PC.
  * Cloud (Option A): dashboard AND ComfyUI both run on a free Colab GPU. You
    open one public link; nothing is stored on your own computer beyond what you
    tap "download" on.

EPHEMERAL MODE (default in cloud): generated files go to a temporary folder,
are served to the browser, and are auto-deleted after a TTL (or on demand).
Nothing persists after the session ends.

Environment variables (all optional):
  AISTUDIO_EPHEMERAL=1            -> use a temp output dir + auto-cleanup
  AISTUDIO_TTL_SECONDS=1800       -> how long results live before auto-delete
  COMFYUI_URL=https://host        -> full base URL of ComfyUI (cloud case)
  COMFYUI_HOST / COMFYUI_PORT     -> host/port instead of a URL (local case)

Run:
    py -m uvicorn web.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.creative import generate_scenes, available_themes  # noqa: E402
from src.comfy_client import ComfyClient, ComfyUIError  # noqa: E402
from src.generate_images import generate_images  # noqa: E402
from src.generate_videos import generate_videos  # noqa: E402
from src.assemble_reel import assemble_reel, FFmpegError  # noqa: E402

CONFIG_DIR = PROJECT_ROOT / "config"
STATIC_DIR = Path(__file__).resolve().parent / "static"


# --------------------------------------------------------------------------- #
# Storage: ephemeral (temp, auto-wiped) vs persistent (project output/)
# --------------------------------------------------------------------------- #
EPHEMERAL = os.environ.get("AISTUDIO_EPHEMERAL", "0") in ("1", "true", "True")
TTL_SECONDS = int(os.environ.get("AISTUDIO_TTL_SECONDS", "1800"))

if EPHEMERAL:
    OUTPUT_DIR = Path(tempfile.mkdtemp(prefix="aistudio_out_"))
else:
    OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Uploaded reference images always live in a temp dir (never the user's disk
# in the cloud case; harmless locally).
REF_DIR = Path(tempfile.mkdtemp(prefix="aistudio_ref_"))


# --------------------------------------------------------------------------- #
# Job tracking (in-memory)
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    id: str
    status: str = "queued"          # queued | running | done | error
    stage: str = ""
    progress: float = 0.0
    scenes: list[dict[str, str]] = field(default_factory=list)
    images: list[str] = field(default_factory=list)   # URL paths
    clips: list[str] = field(default_factory=list)
    reel: str | None = None
    error: str | None = None
    out_dir: Path | None = None     # this job's own subfolder (for cleanup)
    expires_at: float | None = None


_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()


def _load_configs() -> tuple[dict[str, Any], dict[str, Any]]:
    with (CONFIG_DIR / "character.yaml").open("r", encoding="utf-8") as fh:
        character_cfg = yaml.safe_load(fh) or {}
    with (CONFIG_DIR / "pipeline.yaml").open("r", encoding="utf-8") as fh:
        pipeline_cfg = yaml.safe_load(fh) or {}
    return character_cfg, pipeline_cfg


def _job_dir(job_id: str) -> Path:
    d = OUTPUT_DIR / job_id
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "clips").mkdir(parents=True, exist_ok=True)
    (d / "reels").mkdir(parents=True, exist_ok=True)
    return d


def _to_url(path: Path) -> str:
    rel = path.resolve().relative_to(OUTPUT_DIR.resolve())
    return "/outputs/" + str(rel).replace("\\", "/")


def _comfy_client() -> ComfyClient:
    """Build a client from env: full URL preferred, else host/port."""
    url = os.environ.get("COMFYUI_URL")
    if url:
        return ComfyClient(base_url=url)
    host = os.environ.get("COMFYUI_HOST", "127.0.0.1")
    port = int(os.environ.get("COMFYUI_PORT", "8188"))
    return ComfyClient(host=host, port=port)


# --------------------------------------------------------------------------- #
# Background cleanup: delete expired jobs' files (ephemeral mode)
# --------------------------------------------------------------------------- #
def _reap_expired() -> None:
    while True:
        time.sleep(30)
        now = time.time()
        expired: list[str] = []
        with _jobs_lock:
            for jid, job in list(_jobs.items()):
                if job.expires_at and now >= job.expires_at:
                    expired.append(jid)
        for jid in expired:
            _delete_job(jid)


def _delete_job(job_id: str) -> bool:
    with _jobs_lock:
        job = _jobs.pop(job_id, None)
    if not job:
        return False
    if job.out_dir and job.out_dir.exists():
        shutil.rmtree(job.out_dir, ignore_errors=True)
    return True


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class GenerateRequest(BaseModel):
    theme: str = "lifestyle"
    num_images: int = 4
    clip_length: float | None = None
    fps: int | None = None
    make_video: bool = True
    make_reel: bool = True
    captions: bool = True
    use_llm: bool = False
    seed: int | None = None
    reference: str | None = None    # filename returned by /api/reference


class PreviewRequest(BaseModel):
    theme: str = "lifestyle"
    num_images: int = 4
    use_llm: bool = False
    seed: int | None = None


# --------------------------------------------------------------------------- #
# The worker
# --------------------------------------------------------------------------- #
def _run_job(job_id: str, req: GenerateRequest) -> None:
    job = _jobs[job_id]

    def update(**kw: Any) -> None:
        with _jobs_lock:
            for k, v in kw.items():
                setattr(job, k, v)

    try:
        character_cfg, pipeline_cfg = _load_configs()

        # Point output dirs at THIS job's private folder.
        out_dir = _job_dir(job_id)
        update(out_dir=out_dir)
        pipeline_cfg["output"] = {
            "images_dir": str(out_dir / "images"),
            "clips_dir": str(out_dir / "clips"),
            "reels_dir": str(out_dir / "reels"),
            "reel_filename": "reel.mp4",
        }

        # If the user uploaded a reference image, use only that one.
        if req.reference:
            ref_path = REF_DIR / req.reference
            if ref_path.is_file():
                character_cfg["reference_images"] = [str(ref_path)]

        pipeline_cfg["num_images"] = max(1, int(req.num_images))
        if req.seed is not None:
            pipeline_cfg["seed"] = req.seed
        seed = int(pipeline_cfg.get("seed", 12345))
        if seed < 0:
            import random
            seed = random.randint(0, 2**31 - 1)
            pipeline_cfg["seed"] = seed

        pipeline_cfg.setdefault("video", {})["enabled"] = bool(req.make_video)
        pipeline_cfg.setdefault("reel", {})["enabled"] = bool(req.make_reel and req.make_video)
        if req.clip_length is not None:
            pipeline_cfg["video"]["clip_length_seconds"] = req.clip_length
        if req.fps is not None:
            pipeline_cfg["video"]["fps"] = req.fps
        pipeline_cfg["reel"].setdefault("captions", {})["enabled"] = bool(req.captions)

        # --- Creative ideation (works even without ComfyUI). ---
        update(status="running", stage="Dreaming up scenes", progress=0.05)
        scenes = generate_scenes(
            req.theme, pipeline_cfg["num_images"],
            seed=seed, use_llm=bool(req.use_llm),
            llm_model=str(pipeline_cfg.get("creative", {}).get("llm_model", "llama3")),
        )
        pipeline_cfg["scene_prompts"] = [s.prompt for s in scenes]
        if req.captions:
            pipeline_cfg["reel"]["captions"]["texts"] = [s.caption for s in scenes]
        update(scenes=[{"prompt": s.prompt, "caption": s.caption} for s in scenes])

        # --- Connect to ComfyUI (local or cloud). ---
        update(stage="Connecting to ComfyUI", progress=0.1)
        client = _comfy_client()
        client.check_connection()

        # --- Images. ---
        update(stage="Generating images", progress=0.2)
        images = generate_images(client, character_cfg, pipeline_cfg)
        update(images=[_to_url(p) for p in images], progress=0.5)

        if not pipeline_cfg["video"]["enabled"]:
            _finish(job_id, "Done (images only)")
            return

        # --- Videos. ---
        update(stage="Animating clips", progress=0.6)
        clips = generate_videos(client, pipeline_cfg, images)
        update(clips=[_to_url(p) for p in clips], progress=0.85)

        if not pipeline_cfg["reel"]["enabled"]:
            _finish(job_id, "Done (clips ready)")
            return

        # --- Reel. ---
        update(stage="Assembling reel", progress=0.9)
        reel = assemble_reel(pipeline_cfg, clips)
        update(reel=_to_url(reel))
        _finish(job_id, "Done")

    except ComfyUIError as exc:
        update(status="error", error=str(exc), stage="Failed (ComfyUI)")
    except FFmpegError as exc:
        update(status="error", error=str(exc), stage="Failed (FFmpeg)")
    except Exception as exc:  # noqa: BLE001
        update(status="error", error=f"{type(exc).__name__}: {exc}", stage="Failed")


def _finish(job_id: str, stage: str) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return
        job.status = "done"
        job.stage = stage
        job.progress = 1.0
        if EPHEMERAL:
            job.expires_at = time.time() + TTL_SECONDS


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
app = FastAPI(title="AI Character Studio")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


@app.on_event("startup")
def _start_reaper() -> None:
    if EPHEMERAL:
        threading.Thread(target=_reap_expired, daemon=True).start()


@app.get("/api/config")
def api_config() -> dict[str, Any]:
    return {"ephemeral": EPHEMERAL, "ttl_seconds": TTL_SECONDS if EPHEMERAL else None}


@app.get("/api/themes")
def api_themes() -> dict[str, list[str]]:
    return {"themes": available_themes()}


@app.post("/api/reference")
async def api_reference(file: UploadFile = File(...)) -> dict[str, str]:
    """Upload a character reference image; returns a name to pass to /api/generate."""
    suffix = Path(file.filename or "ref.png").suffix.lower() or ".png"
    if suffix not in (".png", ".jpg", ".jpeg", ".webp"):
        raise HTTPException(status_code=400, detail="Use a PNG, JPG, or WEBP image.")
    name = f"{uuid.uuid4().hex[:12]}{suffix}"
    dest = REF_DIR / name
    data = await file.read()
    dest.write_bytes(data)
    return {"reference": name}


@app.post("/api/preview")
def api_preview(req: PreviewRequest) -> dict[str, Any]:
    seed = req.seed if req.seed is not None else 12345
    if seed < 0:
        import random
        seed = random.randint(0, 2**31 - 1)
    scenes = generate_scenes(req.theme, max(1, req.num_images),
                             seed=seed, use_llm=req.use_llm)
    return {"scenes": [{"prompt": s.prompt, "caption": s.caption} for s in scenes]}


@app.post("/api/generate")
def api_generate(req: GenerateRequest) -> dict[str, str]:
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = Job(id=job_id)
    threading.Thread(target=_run_job, args=(job_id, req), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
def api_status(job_id: str) -> dict[str, Any]:
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown or expired job id")
    return {
        "id": job.id,
        "status": job.status,
        "stage": job.stage,
        "progress": job.progress,
        "scenes": job.scenes,
        "images": job.images,
        "clips": job.clips,
        "reel": job.reel,
        "error": job.error,
        "expires_at": job.expires_at,
    }


@app.get("/api/download/{job_id}")
def api_download(job_id: str) -> StreamingResponse:
    """Stream a zip of all of this job's outputs, for one-tap download."""
    job = _jobs.get(job_id)
    if not job or not job.out_dir or not job.out_dir.exists():
        raise HTTPException(status_code=404, detail="Unknown or expired job id")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in job.out_dir.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(job.out_dir)))
    buf.seek(0)
    headers = {"Content-Disposition": f'attachment; filename="ai_studio_{job_id}.zip"'}
    return StreamingResponse(buf, media_type="application/zip", headers=headers)


@app.delete("/api/results/{job_id}")
def api_delete(job_id: str) -> dict[str, bool]:
    """Wipe a job's files immediately (used after the user downloads)."""
    return {"deleted": _delete_job(job_id)}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
