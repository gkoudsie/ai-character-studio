# AI Character Content Pipeline

Generate **consistent images and short videos of your AI character**, then
automatically assemble them into a vertical **9:16 Instagram Reel**.

Everything runs **locally and free** (no per-image cost, no cloud account, no
vendor lock-in). It's plain Python driving [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
for AI generation and [FFmpeg](https://ffmpeg.org/) for video assembly, so it
works on any computer: Windows, macOS, or Linux.

> **You provide:** a few reference pictures of your character and a **theme**
> (e.g. "coffee shop lifestyle"). **The pipeline invents the scenes and captions
> for you**, then produces matching images → animated clips → one finished Reel.

Two ways to use it:
- **Command line** — run the whole pipeline with one command.
- **Phone / web app** — open a link on your phone (same Wi-Fi), pick a theme,
  tap Create, and watch the results. Your PC does the heavy lifting.

All generated content is **safe-for-work**.

---

## How it works

```
reference pics + scene prompts
        │
        ▼
 [1] generate_images   ──►  output/images/*.png     (SDXL + your character)
        │
        ▼
 [2] generate_videos   ──►  output/clips/*.mp4       (each image animated, SVD)
        │
        ▼
 [3] assemble_reel     ──►  output/reels/reel.mp4    (9:16, captions, music)
```

Character **consistency** comes from three things working together:
1. A shared **character prompt** applied to every scene.
2. Your **reference images** fed in via IPAdapter (character reference).
3. **Locked seeds** so runs are reproducible.
For the tightest consistency you can later train a LoRA (see below).

---

## Hardware note (the honest tradeoff)

"Local + free" means **no cost per image**, but the AI models run on your
machine, so speed depends on your hardware:

- **NVIDIA GPU (8GB+ VRAM):** fast, recommended.
- **Apple Silicon (M1/M2/M3):** works via ComfyUI's MPS support, slower.
- **CPU only:** works but very slow (minutes per image). Fine for testing.

The Python code is portable and runs anywhere; only the generation speed
changes between machines.

---

## Setup (one time)

### 1. Install Python 3.9+
Check what you have:

```powershell
py --version        # Windows (this machine)
python3 --version   # macOS / Linux
```

> On this machine, use **`py`** to run Python — plain `python` is not on PATH.
> On macOS/Linux use `python3`. Substitute accordingly in the commands below.

### 2. Install the Python dependencies

```powershell
py -m pip install --user -r requirements.txt
```

### 3. Install FFmpeg (for reel assembly)
FFmpeg must be on your PATH so `ffmpeg` and `ffprobe` run from a terminal.

- **Windows:** `winget install Gyan.FFmpeg` (or download from ffmpeg.org), then
  reopen the terminal and verify with `ffmpeg -version`.
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt install ffmpeg` (or your distro's package manager)

> Only needed if you assemble reels. Image/clip generation works without it.

### 4. Install ComfyUI (the free local AI backend)
1. Download/clone ComfyUI: https://github.com/comfyanonymous/ComfyUI
   (or use the one-click **ComfyUI Desktop** installer for your OS).
2. Install these **custom nodes** via the ComfyUI Manager (search by name):
   - **ComfyUI IPAdapter Plus** (character reference)
   - **ComfyUI-VideoHelperSuite** (video output node used by the clip workflow)
3. Start ComfyUI. By default it serves at `http://127.0.0.1:8188`, which is
   what this pipeline expects.

### 5. Download the models
Place each file in the matching ComfyUI folder. Filenames must match what's in
`config/character.yaml` and `config/pipeline.yaml` (edit either side to match).

| Purpose | Default filename | Put it in | Where to get it |
|---|---|---|---|
| Base image model (SDXL) | `sd_xl_base_1.0.safetensors` | `ComfyUI/models/checkpoints/` | Stability AI SDXL release |
| IPAdapter (SDXL) | `ip-adapter-plus_sdxl_vit-h.safetensors` | `ComfyUI/models/ipadapter/` | h94/IP-Adapter |
| CLIP vision | `CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors` | `ComfyUI/models/clip_vision/` | laion / h94 IP-Adapter repo |
| Image-to-video (SVD) | `svd_xt.safetensors` | `ComfyUI/models/checkpoints/` | Stability AI SVD-XT release |

> These are standard open models. Any equivalent SDXL checkpoint works; just
> update the filename in the config.

### 6. Add your character's reference images
Put your pictures in the `reference_images/` folder and list them in
`config/character.yaml` under `reference_images:`. One good, clear, front-facing
photo is enough to start.

```
reference_images/
  char_01.jpg
```

---

## Configure your character

Edit **`config/character.yaml`** once to describe who the character is:

- `character_prompt` — detailed appearance (hair, eyes, style, etc.)
- `negative_prompt` — things to avoid
- `reference_images` — paths to your pictures
- `reference_strength` — how strongly to match them (0.6–0.8 is a good start)
- `models` — model filenames, and an optional `lora`

## Adjust each run

Edit **`config/pipeline.yaml`** for the knobs you'll change often, or override
any of them on the command line (CLI flags win over the file):

| What | Config key | CLI flag |
|---|---|---|
| Number of images/scenes | `num_images` | `--num-images N` |
| Scene descriptions | `scene_prompts` | `--scene "..."` (repeatable) |
| Reproducibility seed | `seed` | `--seed N` (`-1` = random) |
| Clip length (seconds) | `video.clip_length_seconds` | `--clip-length S` |
| Video frame rate | `video.fps` | `--fps N` |
| Reel size / fps | `reel.width/height/fps` | (config only) |
| Crossfade between clips | `reel.transition_seconds` | (config only) |
| Captions on/off + text | `reel.captions.*` | `--no-captions` to disable |
| Background music | `reel.music_file` | (config only) |
| ComfyUI location | `comfyui.host/port` | `--host` / `--port` |

---

## Run it

From the project root, with ComfyUI running:

```powershell
# Full pipeline: images -> clips -> reel, using the config as-is
py -m src.run_pipeline

# Generate 10 images instead of the configured count
py -m src.run_pipeline --num-images 10

# Only images (skip video + reel) - great for dialing in the character
py -m src.run_pipeline --images-only

# Images + clips, but don't stitch a reel yet
py -m src.run_pipeline --no-reel

# Shorter, snappier clips at 30fps
py -m src.run_pipeline --clip-length 3 --fps 30

# Provide scenes on the fly (replaces the config list)
py -m src.run_pipeline --scene "on a rooftop at night" --scene "in a bookshop"

# Randomize this run instead of the fixed seed
py -m src.run_pipeline --seed -1

# Re-stitch a reel from clips already in output/clips (no regeneration)
py -m src.run_pipeline --reel-from-existing
```

See every option:

```powershell
py -m src.run_pipeline --help
```

Outputs land in:
- `output/images/` — the stills
- `output/clips/` — the animated clips
- `output/reels/reel.mp4` — the finished Reel

---

## Creative mode (let it invent the scenes)

Instead of hand-writing every scene, give it a **theme** and it auto-generates
varied scene ideas **and** captions. Your reference picture is used only for the
character's *look*; the scenes and content are invented fresh each run.

Turn it on in `config/pipeline.yaml`:

```yaml
creative:
  enabled: true
  theme: "coffee shop lifestyle"
  use_llm: false        # true = try a local Ollama LLM, else the offline engine
  auto_captions: true
```

Or straight from the command line (the flag enables creative mode for that run):

```powershell
py -m src.run_pipeline --theme "travel in Japan" --num-images 6
py -m src.run_pipeline --theme "fitness motivation" --images-only
py -m src.run_pipeline --no-creative        # force the manual scene_prompts
```

**Two idea sources:**
- **Offline idea engine (default):** free, instant, no extra setup. It knows the
  themes `lifestyle`, `travel`, `fitness`, `fashion`, `food`, and still works
  with *any* theme text you type.
- **Local LLM (optional):** for more variety, install [Ollama](https://ollama.com)
  (free), run `ollama pull llama3`, then set `use_llm: true` (or pass `--use-llm`).
  If Ollama isn't running, it silently falls back to the offline engine.

---

## Use it from your phone (web app)

Run a small web app on your PC and control everything from your phone's browser
on the **same Wi-Fi**. The phone is the remote; the PC does the generation.

### 1. Start the web app (on your PC)

```powershell
py -m uvicorn web.server:app --host 0.0.0.0 --port 8000
```

### 2. Find your PC's local IP address

```powershell
ipconfig        # Windows: look for "IPv4 Address", e.g. 192.168.1.42
```
(macOS/Linux: `ifconfig` or `ip addr`.)

### 3. Open it on your phone

On a phone connected to the **same Wi-Fi**, open:

```
http://<your-pc-ip>:8000
```
e.g. `http://192.168.1.42:8000`

Pick a theme, tap **Preview ideas** to see the invented scenes, then **Create**.
You'll see live progress, then the images, clips, and final reel with a download
button.

> **Install it like an app:** in your phone browser's menu choose *Add to Home
> Screen*. It then opens fullscreen with its own icon (this is a PWA). ComfyUI
> must be running on the PC for actual generation; the idea **Preview** works
> even without it.

### Firewall note
The first time, Windows may ask to allow Python through the firewall on
**private networks** — allow it so your phone can reach the app. Keep this to
your home Wi-Fi; this app has no login and isn't meant to be exposed to the
public internet.

### Why not a native app (APK / App Store)?
A phone can't run the AI models itself (they need a desktop GPU and several GB of
models), so a native app would just be a remote control talking to your PC —
exactly what this web app already is, minus the app-store overhead and signing.
The "Add to Home Screen" PWA gives you the app feel for free. If you ever want a
packaged APK later, this same web UI can be wrapped with a tool like Capacitor.

---

## Run it in the cloud on a free GPU (nothing stored on your device)

If your computer has no dedicated GPU, run everything on a **free Google Colab
GPU** instead. Both ComfyUI *and* the dashboard run in the cloud; you just open
a link. **Nothing is saved on your own device** except what you tap *download*
on, and results on the server **auto-delete** after ~30 minutes.

### One-time prep: put this project on GitHub
The Colab notebook pulls the app from a Git repo, so push this folder to a
**public GitHub repo** once:

```powershell
git init
git add .
git commit -m "AI Character Studio"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```
(Your `.gitignore` already keeps `output/` and personal `reference_images/` out.)

### Each time you want to create
1. Open **`colab/AI_Character_Studio.ipynb`** in Google Colab
   (colab.research.google.com → File → Upload notebook, or open it from your
   GitHub repo).
2. **Runtime → Change runtime type → GPU → Save.**
3. In **cell 4**, set `PROJECT_GIT_URL` to your repo URL.
4. Run the cells top to bottom (Shift+Enter). The first run downloads ComfyUI
   and the models (a few GB), so give it a few minutes.
5. The **last cell prints a public link** (`https://something.trycloudflare.com`).
   Open it on your phone or laptop.
6. In the dashboard: upload your character photo, pick a theme, tap **Create**,
   then **Download all**. Tap **Delete from server** when done (or just close
   the Colab runtime — everything is discarded).

### How "nothing stored" works
- **Your device:** the dashboard is a web page; it keeps nothing except your
  downloads.
- **The server (Colab):** runs in *ephemeral mode* — each job writes to a temp
  folder, is served to your browser, and is auto-deleted after a TTL
  (`AISTUDIO_TTL_SECONDS`, default 30 min). When the Colab session ends,
  everything is gone.

### Free-tier limits (so there are no surprises)
- Colab free GPUs have **daily usage limits** and sessions **time out when idle
  or closed** — that's exactly what makes this ephemeral, but it means it isn't
  always-on. Start it when you want to create.
- **Images are quick.** **Video (SVD) is heavier** and slower on a free GPU; if
  it's slow or the GPU is busy, generate images only and add video later.

### Connecting a local dashboard to a cloud ComfyUI (alternative)
Prefer to keep the dashboard on your PC but borrow a cloud GPU only for ComfyUI?
Point the app at a remote ComfyUI with an environment variable:

```powershell
$env:COMFYUI_URL = "https://your-comfyui.trycloudflare.com"
py -m uvicorn web.server:app --host 0.0.0.0 --port 8000
```
The client auto-derives the secure websocket (`wss://`) from an `https://` URL.

---

## Tips for better character consistency

1. **Lock the seed** (`seed:` a fixed number) so you can reproduce good runs.
2. **Use 1–3 clean reference images** of the same person, similar lighting.
3. **Keep the character prompt identical** across runs; only vary the scene.
4. **Train a LoRA** for the strongest consistency: collect ~15–30 images of your
   character, train an SDXL LoRA (e.g. with kohya_ss or a ComfyUI trainer), drop
   the `.safetensors` into `ComfyUI/models/loras/`, and set `models.lora` in
   `config/character.yaml`. The pipeline picks it up automatically.

## Troubleshooting

- **"Could not reach ComfyUI"** — Start ComfyUI first; confirm host/port match
  `config/pipeline.yaml` (default `127.0.0.1:8188`).
- **"ComfyUI rejected the workflow"** — Usually a missing model file or custom
  node. Check the filenames in the config match the files in `ComfyUI/models/…`,
  and that IPAdapter Plus + VideoHelperSuite are installed.
- **"FFmpeg not found"** — Install FFmpeg and reopen your terminal so it's on
  PATH (`ffmpeg -version` should work).
- **Faces look inconsistent** — Raise `reference_strength` toward 0.8, use a
  better reference image, or train a LoRA.
- **Clips too short/long** — SVD works best around 14–25 frames; the pipeline
  clamps to that range. For longer Reels, generate more scenes rather than
  longer single clips.

## Responsible use

You're responsible for complying with Instagram's terms, disclosing AI-generated
content where required, and making sure what you create is appropriate and
respects others' likeness and rights.
