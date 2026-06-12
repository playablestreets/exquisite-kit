# MATCH — stage 1 runbook

**Goal:** figure out which audio recording belongs with which photo, tag each pair with a shared
`__pair-NNN` suffix, and move the tagged files into `PROCESSING/`. Leftovers get `__nopair`.

**Script:** `scripts/match.py`. **Prereq:** [SETUP.md](SETUP.md) done; venv active.

## Why not just use the file dates?

The filesystem modified/created timestamps are **useless** for this: once a batch has been
unzipped or copied, every audio file shares one identical timestamp. The real link lives **inside**
each file:

- **Audio** carries a `creation_time` tag in its container metadata → read with `ffprobe`.
- **Images** carry an EXIF `DateTimeOriginal` → read with `exiftool` (Pillow fallback).

In practice each **photo is taken a few seconds before** the child starts narrating it. So match
links each audio to the image captured **2–25 seconds before** it. (On the first real batch this
gave 49 clean pairs with gaps of 2–15 s.)

## Run it

```bash
source .venv/bin/activate

# 1. Drop the raw batch (audio + images, mixed) into INBOX/
#    INBOX/ may have subfolders; match scans recursively.

# 2. Dry run first — see the proposed pairs, move nothing:
python scripts/match.py --inbox INBOX --out PROCESSING --dry-run

# 3. If the pairing looks right, run for real:
python scripts/match.py --inbox INBOX --out PROCESSING
```

Result:
```
PROCESSING/
  audio/   New Recording 2__pair-002.m4a , ... , <unpaired>__nopair.m4a
  images/  E0E95298…__pair-002.jpeg      , ... , <unpaired>__nopair.jpeg
  match_manifest.json     # pair → {audio, image, gap_seconds} + the nopair lists
```

## Knobs

| Flag | Meaning | Default |
|---|---|---|
| `--window N` | max seconds the image may precede the audio | `25` |
| `--back N` | smallest allowed gap (negative tolerates tiny clock jitter) | `-3` |
| `--dry-run` | report the plan, touch nothing | off |

If too many files land in `__nopair`, widen `--window`. If images get matched to the *wrong*
(later) audio, narrow it.

## Properties you can rely on

- **Idempotent:** files already carrying a `__pair-`/`__nopair` token are skipped, so re-running is
  a no-op. Safe to run twice.
- **Safe:** it refuses to start if any destination filename already exists (no overwrites).
- **Portable:** only `ffprobe` + `exiftool`/Pillow — no macOS-only tools.

## Verify

```bash
ls PROCESSING/audio  PROCESSING/images
python - <<'PY'
import json; m=json.load(open("PROCESSING/match_manifest.json"))
print(len(m["pairs"]),"pairs",len(m["audio_unpaired"]),"audio-nopair",len(m["image_unpaired"]),"image-nopair")
assert all(0 <= p["gap_seconds"] <= 25 for p in m["pairs"]), "a pair has an out-of-window gap"
print("gaps ok")
PY
```

Spot-check a couple of pairs: the audio `creation_time` should be a few seconds **after** its
image's `DateTimeOriginal`. Then proceed to [PROCESS.md](PROCESS.md).

> **Note on the existing batch in this repo:** `TO_PROCESS/NEW AUDIO MPRG` + `NEW IMAGES MPRG` were
> already matched and tagged by an earlier run of this method. You can feed them straight into
> `process` (point `--in` at them) to validate stage 2 without re-running match.
