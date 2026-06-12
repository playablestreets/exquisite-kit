# UPDATE — migrate an existing machine to the new image pipeline

**If you are a Claude instance pointed at this folder to update it: run the steps below in order.**
This migrates a machine that ran the *old* image pipeline to the new one. The audio pipeline is
unchanged.

## What changed

The old image stack (ML background-removal: `rembg`/BiRefNet matte, flood-fill silhouettes, SAM2
mask refinement, a "cuttability gate", solid-colour crop fallback) is **gone**. It produced bad
crops — characters jammed to one side, mismatched coloured borders.

The new image pipeline is a **simple flat crop**: locate the right-page drawing panel → bound the
character's ink → fit a **1:3 box** (three stacked squares) → slice into **three equal 1024×1024
opaque PNGs** (raw paper kept, no transparency, no background fill). It's pure OpenCV and runs on
the whole batch for **zero tokens**; only **hard pages** (poor quality, non-workbook drawings,
panel not found) escalate to the local **`claude` CLI** for a human-eye box. The reviewer now edits
a **crop box** over the page photo instead of mask dividers/brushes.

## Migration steps

```bash
cd <repo root>           # the cloned exquisite-kit
git pull

source .venv/bin/activate

# 1. Drop the old heavy deps (no longer used). Ignore "not installed" warnings.
pip uninstall -y rembg onnxruntime torch torchvision sam2 2>/dev/null || true
rm -f checkpoints/*.pt   # old SAM2 checkpoint, if any

# 2. Sync the slimmed dependency set.
pip install -r requirements.txt

# 3. Confirm host prerequisites: ffmpeg/ffprobe, exiftool, and the claude CLI on PATH
#    (authenticated) for hard-case escalation — or plan to run with --no-claude.
python scripts/process.py --selfcheck
```

`--selfcheck` should show `opencv … importable`, the `claude` CLI line, and the Whisper backend —
and **no** torch / rembg / SAM2 lines. If `claude` isn't installed, the pipeline still runs
deterministically (equivalent to `--no-claude`).

## Re-process note

Any `WORK/` proposals from the old run are **stale** (old `image_state.json` schema and tile
format). Regenerate them:

```bash
# rebuild one id to eyeball it, then the batch
python scripts/process.py --in PROCESSING --work WORK --only pair-001
# (use --no-claude to stay fully offline, or --claude-threshold to tune escalation)
```

Open the reviewer and confirm the three tiles are 1024×1024, opaque, centred, with no coloured
border; adjust the crop box if needed.

## Rollback

```bash
git checkout <previous-commit>
pip install -r requirements.txt        # the old requirements.txt (with torch/rembg/sam2)
```
