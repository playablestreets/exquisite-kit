# SETUP — one-time environment install

Run this once on each new machine before using the pipeline. It is safe to re-run; every step is
idempotent. Target: **macOS (Apple Silicon)** or **Linux**. Everything lives inside the kit, so
nothing pollutes the system Python.

> **Why a dedicated Python 3.11:** `faster-whisper`/`ctranslate2` and `whisperx` do not yet publish
> wheels for very new Python (e.g. 3.13/3.14). Pin **3.11**. If you skip this you will hit
> `No matching distribution` errors. (The image side is pure OpenCV + the `claude` CLI — no ML deps.)

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
```

## 2b. The `claude` CLI (hard-case image judgement)

The image pipeline crops most pages with plain OpenCV geometry, but **escalates hard pages**
(poor quality, non-workbook drawings, panel not found) to the local **`claude` CLI**. Make sure
`claude` is on `PATH` and authenticated. It's optional — run `process.py --no-claude` to skip
escalation entirely and stay fully offline/deterministic.

## 3. Model checkpoints

```bash
# in the repo root, venv active
python scripts/process.py --download-models
```
This caches the **Whisper** transcription model (via `faster-whisper`, under `~/.cache`). The image
side needs **no** model checkpoints.

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
[ok] opencv 4.x importable
[ok] claude CLI found (hard-case image judgement)   # or [--] not found → deterministic-only
[ok] faster-whisper importable
[ok] chapter reading: claude-orchestrator (or: anthropic-api)
SELFCHECK PASSED
```

If every line says `[ok]`, you're ready for [MATCH.md](MATCH.md).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No matching distribution found for faster-whisper`/`ctranslate2` | You're not on Python 3.11. Recreate the venv with `python3.11`. |
| `claude: command not found` | Install/authenticate the Claude Code CLI, or run `process.py --no-claude` (deterministic-only). |
| First `process.py` run is slow | The Whisper model downloads on first use; subsequent runs are cached. |
| `exiftool: command not found` (match stage) | Install it (step 1); match falls back to Pillow EXIF but exiftool is more reliable. |
