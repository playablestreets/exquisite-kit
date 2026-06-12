# Exquisite Stories — asset pipeline kit

**If you are a Claude instance pointed at this folder: start here, then follow the runbooks in
order — [SETUP.md](SETUP.md) → [MATCH.md](MATCH.md) → [PROCESS.md](PROCESS.md).**

This kit turns raw photos + audio recordings of children's "exquisite corpse" stories into
website-ready assets. It is self-contained and portable: drop this folder onto a machine that has
Claude installed, run setup once, then run the two stages on each new batch.

## What the inputs are

Each story is **one photo + one audio recording**:

- **Left page** of the photo holds **3 handwritten chapters** (CHAPTER 1 / 2 / 3). A child reads
  them aloud — that reading is the **audio recording**.
- **Right page** holds **one drawn figure** inside a bordered panel, divided by **2 printed dotted
  horizontal lines** into three cells: **head / body / legs**.

The destination is an interactive website that **remixes** stories — it mixes the *top* of one
drawing with the *middle* of another and the *bottom* of a third, played with the matching audio
*beginning / middle / end*. So every story must be cut into **6 assets**.

## What the kit produces

For each story (identified by a shared `pair-NNN` token):

| Asset | Folder | Format |
|-------|--------|--------|
| Head crop | `IMAGES_TOP/` | square transparent PNG, background masked |
| Body crop | `IMAGES_MIDDLE/` | square transparent PNG, background masked |
| Legs crop | `IMAGES_BOTTOM/` | square transparent PNG, background masked |
| Chapter 1 narration | `AUDIO_BEGINNING/` | trimmed WAV |
| Chapter 2 narration | `AUDIO_MIDDLE/` | trimmed WAV |
| Chapter 3 narration | `AUDIO_END/` | trimmed WAV |

…plus a `manifest.json` that links all six assets of a story together for the website to ingest.

## The two stages

```
INBOX/  ──[ match ]──▶  PROCESSING/  ──[ process + review ]──▶  READY_FOR_DRIVE/
raw audio+images       paired & suffixed                       6 asset folders + manifest.json
```

1. **match** ([MATCH.md](MATCH.md)) — figures out which audio file goes with which image (by the
   capture time *embedded inside* each file), gives the pair a shared `__pair-NNN` suffix, and
   moves the tagged files into `PROCESSING/`. Unmatched files get `__nopair`.
2. **process** ([PROCESS.md](PROCESS.md)) — for each pair: crops + masks the drawing into
   top/middle/bottom tiles, splits + trims the audio into beginning/middle/end, opens a **review
   tool** so a human confirms/adjusts every crop, mask, and cut, then exports the sorted assets +
   manifest into `READY_FOR_DRIVE/` (ready to upload to Google Drive by hand).

## The one rule that ties it all together

**The `pair-NNN` token is the association key.** Every one of a story's six output files carries it
(`pair-007_top.png`, `pair-007_beginning.wav`, …). Never rename a file in a way that drops or
changes its token — that is how the website knows the head, body, legs and the three audio clips
all belong to the same story.

## Folder map

This repo's root **is** the kit — run everything from here.

```
./                     ← repo root (the cloned exquisite-kit)
  README.md            ← you are here
  SETUP.md             ← one-time environment install (run first, on each new machine)
  MATCH.md             ← stage 1 runbook
  PROCESS.md           ← stage 2 runbook
  requirements.txt
  scripts/             ← the deterministic Python (you run these; you don't rewrite them)
    match.py
    process.py
    image_ops.py
    audio_ops.py
    export.py
  reviewer/            ← the human review/correct web app
  checkpoints/         ← model files (populated by SETUP; not committed)
  .venv/               ← Python 3.11 virtualenv (created by SETUP)

# data folders — scaffolded empty in the repo; their CONTENTS are git-ignored (never committed,
# so children's photos/audio stay local). Drop your files in and point the scripts here:
INBOX/                 ← drop a new batch of raw audio + image files here
PROCESSING/            ← match's output; process's input
WORK/                  ← process's intermediate proposals + review state
READY_FOR_DRIVE/       ← final assets, mirrors the table above
```

## Quick start (after SETUP)

```bash
source .venv/bin/activate

# Stage 1: pair + tag + move
python scripts/match.py --inbox INBOX --out PROCESSING

# Stage 2: build proposals, then open the reviewer, then export
python scripts/process.py --in PROCESSING --work WORK
python -m uvicorn reviewer.app:app --app-dir . --port 8765
#   → open http://localhost:8765 , review every pair, click Save
python scripts/export.py --work WORK --out READY_FOR_DRIVE
```

Each script is **idempotent and resumable** — safe to re-run; it skips work already done.
