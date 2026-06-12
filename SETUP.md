# SETUP — one-time environment install

Run this once on each new machine before using the pipeline. It is safe to re-run; every step is
idempotent. Target: **macOS (Apple Silicon)** or **Linux**. Everything lives inside the kit, so
nothing pollutes the system Python.

> **Why a dedicated Python 3.11:** `torch`, `onnxruntime`, and `whisperx` do not yet publish wheels
> for very new Python (e.g. 3.13/3.14). Pin **3.11**. If you skip this you will hit
> `No matching distribution` errors.

## 1. System packages

### macOS (Homebrew)
```bash
brew install python@3.11 ffmpeg exiftool
```

### Linux (Debian/Ubuntu)
```bash
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv ffmpeg libimage-exiftool-perl
```

Verify:
```bash
python3.11 --version      # → Python 3.11.x
ffmpeg  -version | head -1
ffprobe -version | head -1
exiftool -ver
```

## 2. Python virtualenv + packages

From the repo root (the cloned `exquisite-kit/`):
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
pip install -r requirements.txt

# SAM 2 (interactive mask refinement) — not reliably on PyPI, install from source:
pip install "git+https://github.com/facebookresearch/sam2.git"
```

> **Apple Silicon note:** the default `torch` wheel already supports the **MPS** GPU backend. No
> CUDA needed. On a Linux+NVIDIA box, instead `pip install onnxruntime-gpu` and the CUDA torch
> build for big speedups.

## 3. Model checkpoints

```bash
# in the repo root, venv active
python scripts/process.py --download-models
```
This downloads/caches:
- **BiRefNet** background-removal model (via `rembg`, cached under `~/.u2net` / HF cache),
- **Whisper** transcription model (via `whisperx`, cached under `~/.cache`),
- **SAM 2** checkpoint into `checkpoints/` (`sam2.1_hiera_base_plus.pt`, ~80 MB).

If `--download-models` can't reach a source, it prints the exact URLs to fetch manually into
`checkpoints/`.

## 4. Chapter-text reading (how the audio split learns where the chapters are)

The audio splitter needs the text of the 3 chapters to know where narration moves from chapter 1→2→3.
Two ways to provide it — pick one:

- **A. Claude orchestrator (default, no API key).** You — the Claude instance running this kit —
  read the 3 chapter boxes off each image yourself and write them to a sidecar file. `process.py`
  prompts for this and tells you exactly which file to write (`WORK/<pair>/chapters.json`). Kid
  handwriting is easy for you and needs no extra credentials.
- **B. Anthropic API (fully automated).** Set a key and `process.py` will read chapters itself:
  ```bash
  export ANTHROPIC_API_KEY=sk-ant-...
  ```
  (If you build against the SDK, confirm the current vision-capable model id via the `claude-api`
  skill before coding — model ids change.)

## 5. Verify the install

```bash
python scripts/process.py --selfcheck
```
Expected output (abridged):
```
[ok] ffmpeg / ffprobe / exiftool found
[ok] torch 2.x  device=mps            # or cuda / cpu
[ok] rembg BiRefNet session created
[ok] whisperx model loaded
[ok] SAM2 checkpoint present
[ok] chapter reading: claude-orchestrator (or: anthropic-api)
SELFCHECK PASSED
```

If every line says `[ok]`, you're ready for [MATCH.md](MATCH.md).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No matching distribution found for torch` | You're not on Python 3.11. Recreate the venv with `python3.11`. |
| `onnxruntime` import crash on Linux | `pip install onnxruntime` (CPU) or the matching `onnxruntime-gpu`. |
| SAM 2 import error | Re-run the `git+` install; needs a C++ toolchain (`xcode-select --install` / `build-essential`). |
| First `process.py` run is slow | Models download on first use; subsequent runs are cached. |
| `exiftool: command not found` (match stage) | Install it (step 1); match falls back to Pillow EXIF but exiftool is more reliable. |
