# PROCESS — stage 2 runbook

**Goal:** turn every matched pair in `PROCESSING/` into the 6 website assets (3 body-part image
tiles + 3 trimmed audio clips), with a human confirming each one, then sort them into
`READY_FOR_DRIVE/` with a `manifest.json`.

**Scripts:** `scripts/process.py` (build proposals), `reviewer/` (human review), `scripts/export.py`
(final sort). **Prereq:** [SETUP.md](SETUP.md) done + `--selfcheck` passes; [MATCH.md](MATCH.md) run.

## The flow

```
PROCESSING/ ──process.py──▶ WORK/<id>/ (proposals) ──reviewer──▶ (confirmed) ──export.py──▶ READY_FOR_DRIVE/
```

Each `WORK/<id>/` holds editable state, not finals:
```
WORK/pair-007/
  status.json            # reviewed? which assets exist
  image/  source.png     # downscaled page photo (the reviewer draws the box over this)
          image_state.json   # box, panel_rect, char_bbox, dividers, engine, confidence
          top.png middle.png bottom.png      # 1024x1024 opaque tile proposals
  audio/  audio_state.json    # boundaries, method, chapters, transcript
          beginning.wav middle.wav end.wav  # clip proposals
          chapters.json        # the 3 chapter texts (see step 2)
```

## Step 1 — build proposals

```bash
source .venv/bin/activate

# validate on a few first:
python scripts/process.py --in PROCESSING --work WORK --only pair-001,pair-002,pair-003
# then the whole batch:
python scripts/process.py --in PROCESSING --work WORK
```
Per pair this:
- **Image (simple flat crop — no masking/transparency):** locates the right-page drawing panel,
  finds the drawn character's ink bounding box inside it, and grows that to a **1:3 box** (three
  stacked squares) with a little padding — anchored by the 2 printed dotted dividers when they look
  sensible. The box is sliced into **three EQUAL squares** (head / torso / legs) and each is resized
  to a **1024×1024 opaque PNG** of the raw paper. No background removal, no solid-colour fill — so no
  mismatched borders. Think "cut three squares out with scissors so they re-stack into the figure."
  Everything is axis-aligned in original-image pixels; the **box** is the one editable thing.

  **Hard cases escalate to Claude (only when needed).** Every image first gets the cheap
  deterministic crop above and a **confidence score**. Clean workbook pages pass and cost **zero
  tokens**. When the geometry is unsure — poor-quality photo, panel not found, **non-workbook / plain
  drawing**, character barely detected — the page escalates to the local **`claude` CLI**
  (`scripts/vision.py`), which looks at the photo like a person and returns a box placed so head /
  torso / legs land in the thirds. Controls:
  - `--no-claude` — never escalate (fully offline/deterministic).
  - `--always-claude` — escalate every image (evaluation only; spends tokens).
  - `--claude-threshold 0.55` — escalate when confidence is below this.
  The run prints `Images: N built, K escalated to claude`. Per image, `image_state.json` records
  `box`, `panel_rect`, `char_bbox`, `dividers`, `source_engine` (`opencv`/`claude`), `confidence`
  and `gate_reasons`; the reviewer shows these.
- **Audio:** Whisper (faster-whisper backend) transcribes with word timestamps; the 3 chapter texts
  (step 2) are aligned to the transcript to place **4 edges** — story start, the two chapter cuts,
  and story end — so adult preamble before and chatter after the story are trimmed off, not just the
  interior cuts. The recording is split + silence-trimmed into 3 clips.

`--in` can point at any folder of already-tagged files (it discovers them by `pair-NNN` token
regardless of folder layout). Already-built ids are skipped; use `--only` to force a rebuild.

> **Performance:** the deterministic crop is pure OpenCV (~0.1 s/image), so a whole batch flies and
> costs no tokens. Only low-confidence images call the `claude` CLI (a few seconds each) — keep that
> count down on large batches with a sensible `--claude-threshold`, or run `--no-claude` entirely.
> Whisper transcription is a few seconds per clip. `process.py` is resumable — leave it running,
> stop/restart freely, it skips finished ids. The reviewer's **Recompute** is on-demand, so review
> stays snappy.

## Step 2 — chapter texts (so the audio splits at the right places)

The aligner needs the 3 chapter texts. Two ways (set in SETUP):
- **Claude orchestrator (default):** for each id, read the 3 chapter boxes off
  `WORK/<id>/image/panel.png` (or the original photo) and write them to
  `WORK/<id>/audio/chapters.json`:
  ```json
  { "chapters": ["Once Frank ate a bean", "the bean made body parts sprout all over him", "he went home"] }
  ```
  Then re-run `process.py --only <id>` (or just adjust cuts in the reviewer). You can also type them
  directly in the reviewer's chapter boxes and click **Save chapter texts**.
- **Anthropic API:** with `ANTHROPIC_API_KEY` set, `process.py` reads them automatically — no action.

If chapters are missing, the splitter falls back to silence detection, then even thirds, so there's
always something to correct in the reviewer.

## Step 3 — review & correct (the human-in-the-loop)

```bash
WORK=WORK python -m uvicorn reviewer.app:app --app-dir . --port 8765
# open http://localhost:8765
```
Per story:
- **Image:** the badge shows `kind · engine · confidence` (and any gate reasons). Usually nothing to
  do. To adjust, **drag the blue crop box** over the page photo to move it, or drag a **corner** to
  resize; the dashed lines show the equal-thirds head/torso/legs cuts. Click **Recompute tiles** to
  regenerate the three squares.
- **Audio:** check/fix the 3 **chapter texts**; drag the 4 **edge handles** on the timeline — the
  two orange handles are the **story start / end** (they trim adult preamble before and chatter
  after the story), the inner two are the **chapter 1→2 / 2→3** cuts; **Re-split**; play each clip to
  confirm it starts/ends on the right chapter.
- Click **✓ Save & mark reviewed**, then **Next unreviewed →**.

The tool is resumable — reviewed state persists in `status.json`; close and reopen anytime.

## Step 4 — export to Drive-ready folders

```bash
python scripts/export.py --work WORK --out READY_FOR_DRIVE
```
Produces:
```
READY_FOR_DRIVE/
  IMAGES_TOP/pair-007_top.png   IMAGES_MIDDLE/pair-007_middle.png   IMAGES_BOTTOM/pair-007_bottom.png
  AUDIO_BEGINNING/pair-007_beginning.wav  AUDIO_MIDDLE/pair-007_middle.wav  AUDIO_END/pair-007_end.wav
  manifest.json
```
Only ids marked reviewed are exported (override with `--include-unreviewed`). Every file keeps its
`pair-NNN` token so the website links the 6 assets of a story. `manifest.json` lists, per story, the
6 asset paths + chapter texts + split method + `unpaired` flag.

## Step 5 — publish to Google Drive

`export.py` only *stages* into `READY_FOR_DRIVE/`; it never touches the network or stores any
credentials (so nothing secret lives in this repo). Uploading is a separate, final step:

- **By hand:** drag `READY_FOR_DRIVE/`'s six asset folders + `manifest.json` into the Drive folder.
- **As a routine (recommended):** have the agent (e.g. cowork) upload `READY_FOR_DRIVE/**` to a
  target Drive folder using **its own connected Google Drive** — preserve the folder names
  (`IMAGES_TOP/ … AUDIO_END/`) and `manifest.json`, and don't rename files (the `pair-NNN` token is
  the association key). Credentials stay in the agent's environment, never in the repo. If you prefer
  a CLI instead, configure `rclone` locally and `rclone copy READY_FOR_DRIVE <remote>:<folder>` —
  the `rclone` config is machine-local and git-ignored.

## Leftovers (`__nopair`)
`__nopair` images still produce 3 tiles; `__nopair` audio still splits. They appear in `WORK/` as
`nopair-image-NNN` / `nopair-audio-NNN`, are reviewable like any other, and are flagged
`"unpaired": true` in the manifest so the website can decide whether to use them.

## Verify (see also the plan's verification section)
- `READY_FOR_DRIVE/` has the 6 asset folders + `manifest.json`; counts match reviewed ids × assets.
- Each PNG: 1024×1024, opaque, character centred and frame-filling, **no mismatched border**
  (`python -c "from PIL import Image; im=Image.open(p); print(im.size, im.mode)"`).
- Each WAV: starts/ends on its chapter, silence trimmed; durations present in `manifest.json`.
- Every `pair-NNN` appears once per relevant asset folder; manifest links all 6.
