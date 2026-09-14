"""
Reel assembly stage (FFmpeg).

Takes the generated clips and stitches them into one vertical 9:16 reel:
  1. Each clip is scaled + center-cropped to the target resolution (default
     1080x1920) so mixed source sizes line up perfectly.
  2. Optional caption text is burned onto each clip.
  3. Clips are joined with either hard cuts or crossfade transitions.
  4. Optional background music is mixed in and trimmed to the video length.

FFmpeg is free, cross-platform, and does all the heavy lifting, so this stage
has no Python video dependencies. FFmpeg must be installed and on PATH.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


class FFmpegError(RuntimeError):
    pass


def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FFmpegError(
            "FFmpeg not found on PATH. Install it (see README) and make sure "
            "`ffmpeg` runs from a terminal, then try again."
        )
    return ffmpeg


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise FFmpegError(
            "FFmpeg command failed:\n"
            f"  {' '.join(cmd)}\n\n{result.stderr[-2000:]}"
        )


def _escape_drawtext(text: str) -> str:
    """Escape characters that are special inside FFmpeg's drawtext filter."""
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace("%", "\\%")
    )


def _caption_y(position: str, font_size: int, height: int) -> str:
    position = (position or "bottom").lower()
    if position == "top":
        return f"{int(height * 0.08)}"
    if position == "center":
        return "(h-text_h)/2"
    # bottom (default) - sit above the very edge
    return f"h-text_h-{int(height * 0.10)}"


def _normalize_clip(
    ffmpeg: str,
    src: Path,
    dest: Path,
    *,
    width: int,
    height: int,
    fps: int,
    caption: str | None,
    caption_cfg: dict[str, Any],
) -> None:
    """Scale+crop one clip to target size, set fps, and optionally add a caption."""
    # Scale to cover the target box, then center-crop to exact dimensions.
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},"
        f"fps={fps},"
        f"setsar=1"
    )

    if caption:
        font_size = int(caption_cfg.get("font_size", 64))
        color = caption_cfg.get("font_color", "white")
        y = _caption_y(caption_cfg.get("position", "bottom"), font_size, height)
        drawtext = (
            f"drawtext=text='{_escape_drawtext(caption)}':"
            f"fontcolor={color}:fontsize={font_size}:"
            f"x=(w-text_w)/2:y={y}:"
            f"box=1:boxcolor=black@0.4:boxborderw=20"
        )
        vf = f"{vf},{drawtext}"

    cmd = [
        ffmpeg, "-y",
        "-i", str(src),
        "-vf", vf,
        "-r", str(fps),
        "-an",  # drop any source audio; music is added at the end
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-crf", "20",
        str(dest),
    ]
    _run(cmd)


def _concat_hard(ffmpeg: str, clips: list[Path], dest: Path) -> None:
    """Join clips with hard cuts using the concat demuxer (fast, lossless-ish)."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
        list_file = Path(fh.name)
        for clip in clips:
            # concat demuxer needs forward slashes / escaped paths
            safe = str(clip.resolve()).replace("\\", "/")
            fh.write(f"file '{safe}'\n")
    try:
        cmd = [
            ffmpeg, "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            str(dest),
        ]
        _run(cmd)
    finally:
        list_file.unlink(missing_ok=True)


def _concat_crossfade(
    ffmpeg: str, clips: list[Path], dest: Path, fps: int, transition: float
) -> None:
    """Join clips with xfade crossfades. Requires knowing each clip's duration."""
    durations = [_probe_duration(clip) for clip in clips]

    inputs: list[str] = []
    for clip in clips:
        inputs += ["-i", str(clip)]

    # Build a chain of xfade filters. Each xfade offset is the running total of
    # previous clip durations minus the accumulated transition overlaps.
    fil = []
    prev_label = "0:v"
    offset = durations[0] - transition
    for i in range(1, len(clips)):
        out_label = f"x{i}"
        fil.append(
            f"[{prev_label}][{i}:v]"
            f"xfade=transition=fade:duration={transition}:offset={offset:.3f}"
            f"[{out_label}]"
        )
        prev_label = out_label
        # Next offset adds this clip's contribution minus one more overlap.
        offset += durations[i] - transition

    filter_complex = ";".join(fil)
    cmd = [
        ffmpeg, "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", f"[{prev_label}]",
        "-r", str(fps),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-crf", "20",
        str(dest),
    ]
    _run(cmd)


def _probe_duration(path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise FFmpegError("ffprobe not found on PATH (installed alongside FFmpeg).")
    result = subprocess.run(
        [
            ffprobe, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise FFmpegError(f"ffprobe failed on {path}:\n{result.stderr}")
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise FFmpegError(f"Could not read duration of {path}") from exc


def _add_music(ffmpeg: str, video: Path, music: Path, dest: Path) -> None:
    """Mix background music, trimming to the shorter of video/audio."""
    cmd = [
        ffmpeg, "-y",
        "-i", str(video),
        "-i", str(music),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(dest),
    ]
    _run(cmd)


def assemble_reel(
    pipeline_cfg: dict[str, Any],
    clip_paths: list[Path],
) -> Path:
    """
    Stitch clips into a single vertical reel. Returns the final reel path.
    """
    if not clip_paths:
        raise FFmpegError("No clips to assemble into a reel.")

    ffmpeg = _require_ffmpeg()
    reel_cfg = pipeline_cfg.get("reel", {})
    width = int(reel_cfg.get("width", 1080))
    height = int(reel_cfg.get("height", 1920))
    fps = int(reel_cfg.get("fps", 30))
    transition = float(reel_cfg.get("transition_seconds", 0.0))

    caption_cfg = reel_cfg.get("captions", {}) or {}
    captions_on = bool(caption_cfg.get("enabled", False))
    caption_texts = list(caption_cfg.get("texts", [])) if captions_on else []

    out_cfg = pipeline_cfg.get("output", {})
    reels_dir = Path(out_cfg.get("reels_dir", "output/reels"))
    reels_dir.mkdir(parents=True, exist_ok=True)
    reel_name = out_cfg.get("reel_filename", "reel.mp4")
    final_path = reels_dir / reel_name

    project_root = Path(__file__).resolve().parent.parent

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # 1. Normalize every clip to the same size/fps + optional caption.
        normalized: list[Path] = []
        for idx, clip in enumerate(clip_paths):
            caption = None
            if caption_texts:
                caption = caption_texts[idx % len(caption_texts)]
            norm = tmp_dir / f"norm_{idx:03d}.mp4"
            print(f"  normalizing clip {idx + 1}/{len(clip_paths)}"
                  + (f" (caption: {caption!r})" if caption else ""))
            _normalize_clip(
                ffmpeg, clip, norm,
                width=width, height=height, fps=fps,
                caption=caption, caption_cfg=caption_cfg,
            )
            normalized.append(norm)

        # 2. Concatenate (crossfade if requested and we have >1 clip).
        stitched = tmp_dir / "stitched.mp4"
        if transition > 0 and len(normalized) > 1:
            print(f"  joining with {transition}s crossfades")
            _concat_crossfade(ffmpeg, normalized, stitched, fps, transition)
        else:
            print("  joining with hard cuts")
            _concat_hard(ffmpeg, normalized, stitched)

        # 3. Optional music.
        music_file = reel_cfg.get("music_file")
        if music_file:
            music_path = (project_root / music_file) if not Path(music_file).is_absolute() else Path(music_file)
            if music_path.is_file():
                print(f"  adding music: {music_path.name}")
                _add_music(ffmpeg, stitched, music_path, final_path)
            else:
                print(f"  music file not found, skipping: {music_path}")
                shutil.copyfile(stitched, final_path)
        else:
            shutil.copyfile(stitched, final_path)

    print(f"  reel -> {final_path}")
    return final_path
