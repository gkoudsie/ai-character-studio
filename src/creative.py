"""
Creative theme engine.

Turns a high-level THEME (e.g. "coffee shop lifestyle", "travel in Japan",
"fitness motivation") into a set of varied, SFW scene prompts and matching
captions, so you don't have to hand-write every scene.

Two backends:
  1. Offline idea engine (default): remixes curated pools of locations,
     activities, moods, times of day, and camera framing into coherent scenes.
     Zero setup, fully offline, deterministic per seed so runs are reproducible.
  2. Optional local LLM via Ollama (http://localhost:11434): asks a local model
     to invent scenes/captions. More varied; used only if reachable, otherwise
     we fall back to the offline engine automatically.

Everything here is safe-for-work by design.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class Scene:
    """One generated idea: the image prompt fragment plus a short caption."""
    prompt: str    # describes the SCENE only (character is added by the pipeline)
    caption: str   # short overlay text for the reel


# ---------------------------------------------------------------------------
# Offline idea engine
# ---------------------------------------------------------------------------
# Each theme maps to pools we remix. A theme not listed here falls back to the
# generic "lifestyle" pools blended with the theme words, so any theme works.

_THEMES: dict[str, dict[str, list[str]]] = {
    "lifestyle": {
        "locations": [
            "in a sunny city park", "on a rooftop terrace at golden hour",
            "in a cozy bookshop", "at a farmers market", "by a big window at home",
            "walking a quiet tree-lined street", "in a bright modern kitchen",
        ],
        "activities": [
            "sipping coffee and smiling", "reading a book", "laughing candidly",
            "arranging fresh flowers", "looking thoughtfully into the distance",
            "taking a relaxed stroll", "enjoying a quiet morning",
        ],
        "captions": [
            "Slow mornings", "Little joys", "Just being", "Good vibes only",
            "Everyday magic", "Take it easy", "Present moment",
        ],
    },
    "travel": {
        "locations": [
            "in front of a historic temple", "on a scenic mountain overlook",
            "wandering a lantern-lit old town", "at a seaside promenade",
            "in a bustling street market", "beside a famous landmark at dusk",
            "on a train with countryside rolling by",
        ],
        "activities": [
            "checking a paper map", "snapping a photo", "trying local street food",
            "gazing at the view", "waving hello", "carrying a small backpack, exploring",
            "sipping tea at a small cafe",
        ],
        "captions": [
            "Wander often", "New horizons", "Lost in the moment", "Adventure awaits",
            "Collect memories", "Somewhere new", "Keep exploring",
        ],
    },
    "fitness": {
        "locations": [
            "in a bright modern gym", "on a running track at sunrise",
            "in a yoga studio", "at an outdoor workout park", "on a forest trail",
            "stretching by the beach", "in a home workout space",
        ],
        "activities": [
            "mid-stretch, focused", "jogging with determination",
            "holding a strong yoga pose", "taking a water break, smiling",
            "lifting light dumbbells", "catching breath after a run",
            "doing a warm-up",
        ],
        "captions": [
            "Stronger daily", "Show up", "One more rep", "Move your body",
            "Progress not perfection", "Earn it", "Keep going",
        ],
    },
    "fashion": {
        "locations": [
            "against a colorful painted wall", "in a chic boutique",
            "on a stylish city street", "in a minimalist studio",
            "by industrial-style windows", "at a rooftop with skyline behind",
            "in an autumn park",
        ],
        "activities": [
            "showing off a casual chic outfit", "adjusting a stylish jacket",
            "posing confidently", "mid-stride, effortless", "leaning against a wall",
            "twirling lightly", "glancing over the shoulder",
        ],
        "captions": [
            "Outfit of the day", "Keep it simple", "Style notes", "Effortless",
            "Details matter", "Feeling it", "Look of the day",
        ],
    },
    "food": {
        "locations": [
            "in a warm rustic kitchen", "at a cozy brunch table",
            "in a sunlit cafe", "at a colorful food market", "on a picnic blanket",
            "by a dessert counter", "at a home dining table",
        ],
        "activities": [
            "plating a fresh dish", "sipping a latte", "tasting something delicious",
            "holding a colorful smoothie", "cutting fresh fruit",
            "presenting a homemade meal", "enjoying brunch",
        ],
        "captions": [
            "Homemade goodness", "Treat yourself", "Fresh and simple", "Yum",
            "Made with love", "Brunch mood", "Tasty things",
        ],
    },
}

# Shared modifiers layered onto any theme for variety.
_TIMES = [
    "soft morning light", "warm golden-hour light", "bright midday sun",
    "cozy indoor lighting", "gentle overcast light", "blue-hour evening glow",
]
_FRAMING = [
    "full-body shot", "waist-up portrait", "close-up portrait",
    "candid wide shot", "over-the-shoulder shot", "medium shot",
]
_MOODS = [
    "cheerful and relaxed", "confident", "serene and calm",
    "playful", "warm and friendly", "focused",
]


def _theme_pools(theme: str) -> dict[str, list[str]]:
    """Pick the closest theme pool, or blend generic lifestyle with theme words."""
    key = theme.strip().lower()
    for name, pools in _THEMES.items():
        if name in key:
            return pools
    # Unknown theme: base on lifestyle but inject the theme itself into locations
    # so the theme still clearly influences the scenes.
    base = {k: list(v) for k, v in _THEMES["lifestyle"].items()}
    base["locations"] = [f"in a setting themed around {theme}"] + base["locations"]
    return base


def _offline_scenes(theme: str, count: int, seed: int) -> list[Scene]:
    rng = random.Random(seed)
    pools = _theme_pools(theme)
    scenes: list[Scene] = []
    used: set[str] = set()

    attempts = 0
    while len(scenes) < count and attempts < count * 20:
        attempts += 1
        loc = rng.choice(pools["locations"])
        act = rng.choice(pools["activities"])
        mood = rng.choice(_MOODS)
        time = rng.choice(_TIMES)
        frame = rng.choice(_FRAMING)
        prompt = f"{loc}, {act}, {mood}, {time}, {frame}"
        if prompt in used:
            continue
        used.add(prompt)
        caption = rng.choice(pools["captions"])
        scenes.append(Scene(prompt=prompt, caption=caption))

    # If the theme's combinations are exhausted, top up allowing repeats.
    while len(scenes) < count:
        loc = rng.choice(pools["locations"])
        act = rng.choice(pools["activities"])
        scenes.append(
            Scene(
                prompt=f"{loc}, {act}, {rng.choice(_MOODS)}, {rng.choice(_TIMES)}, {rng.choice(_FRAMING)}",
                caption=rng.choice(pools["captions"]),
            )
        )
    return scenes[:count]


# ---------------------------------------------------------------------------
# Optional Ollama (local LLM) backend
# ---------------------------------------------------------------------------
_OLLAMA_URL = "http://localhost:11434"

_LLM_SYSTEM = (
    "You are a creative director for a social media character. "
    "Generate short, vivid, SAFE-FOR-WORK scene ideas for photos/reels. "
    "No nudity, no sexual content, no violence. Keep it wholesome and brand-safe."
)


def _ollama_available(model: str, timeout: float = 2.0) -> bool:
    try:
        resp = requests.get(f"{_OLLAMA_URL}/api/tags", timeout=timeout)
        resp.raise_for_status()
        tags = resp.json().get("models", [])
        names = {m.get("name", "").split(":")[0] for m in tags}
        return model.split(":")[0] in names or bool(names)
    except requests.RequestException:
        return False


def _ollama_scenes(theme: str, count: int, seed: int, model: str) -> list[Scene] | None:
    """Ask a local LLM for scenes. Returns None on any failure (caller falls back)."""
    prompt = (
        f"Theme: {theme}\n"
        f"Produce exactly {count} distinct scene ideas for an AI character.\n"
        "Return ONLY a JSON array. Each item: "
        '{"prompt": "<scene description, no character appearance, include setting, '
        'action, mood, lighting, camera framing>", "caption": "<max 4 words>"}. '
        "All ideas must be safe-for-work and wholesome."
    )
    body = {
        "model": model,
        "prompt": prompt,
        "system": _LLM_SYSTEM,
        "stream": False,
        "options": {"seed": seed, "temperature": 0.9},
    }
    try:
        resp = requests.post(f"{_OLLAMA_URL}/api/generate", json=body, timeout=120)
        resp.raise_for_status()
        text = resp.json().get("response", "")
    except requests.RequestException:
        return None

    data = _extract_json_array(text)
    if not data:
        return None

    scenes: list[Scene] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        p = str(item.get("prompt", "")).strip()
        c = str(item.get("caption", "")).strip() or "New day"
        if p:
            scenes.append(Scene(prompt=p, caption=c))
    return scenes[:count] if scenes else None


def _extract_json_array(text: str) -> list[Any] | None:
    """Pull the first JSON array out of an LLM response, tolerating extra prose."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, list) else None
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_scenes(
    theme: str,
    count: int,
    *,
    seed: int = 12345,
    use_llm: bool = False,
    llm_model: str = "llama3",
) -> list[Scene]:
    """
    Generate `count` creative SFW scenes for a theme.

    If use_llm is True and Ollama is reachable, uses the local LLM; on any
    problem it silently falls back to the offline idea engine so the pipeline
    never gets stuck.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    theme = (theme or "lifestyle").strip()

    if use_llm and _ollama_available(llm_model):
        llm = _ollama_scenes(theme, count, seed, llm_model)
        if llm:
            # If the LLM returned fewer than requested, top up from the offline engine.
            if len(llm) < count:
                extra = _offline_scenes(theme, count - len(llm), seed + 1)
                llm.extend(extra)
            return llm[:count]

    return _offline_scenes(theme, count, seed)


def available_themes() -> list[str]:
    """Theme presets the offline engine knows well (any string still works)."""
    return sorted(_THEMES.keys())
