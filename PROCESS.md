# PROCESS — stage 2 runbook

**Goal:** turn every matched pair in `PROCESSING/` into the 6 website assets (3 masked body-part
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
  image/  panel.png      # rectified right-hand drawing panel
          image_state.json   # divider positions + rectify info
          top.png middle.png bottom.png      # tile proposals (RGBA)
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
- **Image:** locates the subject (rectifies the right drawing panel — perspective quad, or a
  rotated-rect fit when the page is skewed — or, for a **non-workbook** single drawing with no
  bordered panel, crops to the drawn content), **trims any residual printed border** rectify left
  along an edge, finds the 2 dotted dividers → head/body/legs (or equal thirds for a plain drawing),
  and masks the **whole panel once** (so the silhouette stays closed head-to-toe) before slicing.

  **The masking method is chosen automatically per image (`--method auto`, default) by a
  *cuttability gate*** — see below. You can still force a single method:
  - **`fill`** (best for faint line drawings): a **scissor cut-out** — thicken the drawn lines to
    close gaps, flood the background in from the panel border, and everything the flood can't reach
    (the interior enclosed by the outline) becomes a **solid filled silhouette**. Pure OpenCV, fast.
    - **Leak repair (`gap_seal`)**: if the child left a gap in an outline (e.g. an "implicit" belly
      wall), the background leaks in and that interior comes out hollow. `auto` (default) safely fills
      only interiors enclosed by a *thin* neck — it can never fill the border-connected exterior, so
      it never over-fills. The reviewer's **"fill leaks" slider** bridges a *wide* gap by hand, but
      `auto` mode avoids needing it: if no clean cut-out is possible it **crops** instead (below).
  - **`matte`**: BiRefNet learned matte unioned with an ink pass (best for **bold/shaded/coloured**
    art — gives cleaner edges than fill, but goes hollow on faint outline-only drawings).
  - **`luma`**: luminance key (fastest, lowest quality).
  - **`crop`**: **no cut-out** — slice the region into thirds on a **solid page-colour background**.
    The hands-off fallback for drawings that can't be cleanly cut out.
  Output: 3 square PNG tiles (transparent when cut, opaque solid-bg when cropped), plus
  `image/debug/01..05` intermediates for inspection.

  **Cuttability gate (`auto`)** — fully hands-off, no manual slider. For each image it scores the
  `fill` silhouette and (if BiRefNet is installed) the `matte` silhouette on shape-quality metrics
  (`solidity` = area ÷ convex hull, `enclosure` = filled area ÷ raw ink, dominant-mass fraction,
  fragment count, ink coverage), then:
  - prefers **matte** when it is clean *and* actually filled the figure (bold/shaded art);
  - else uses **fill** when it is clean (faint line art);
  - if **neither** produces a clean silhouette (broken/scribbly/messy outline that would leak or come
    out stringy), falls back to **crop** — a solid-background crop of the top/middle/bottom is
    acceptable here, and avoids exporting a broken cut-out.
  The decision is recorded in `image_state.json` (`kind`, `cut_method`, `cuttable`, `scores`) and
  in `status.json`, and shown as a badge in the reviewer. Thresholds live in `_CLEAN` in
  `scripts/image_ops.py` (calibrated on the sample panels). To re-tune, drop a known-bad drawing in,
  inspect its `scores`, and nudge the thresholds.

  > Rectification is the fragile step on messy phone photos (skew, the whole spread, or a partial
  > strip). When auto-detection can't find a clean panel it falls back to a right-half crop (still
  > border-trimmed) and flags `rectify: fallback-right-half` in `status.json`; non-workbook drawings
  > are tagged `kind: plain`. Fix any mis-located subject in the reviewer (drag dividers; switch
  > method; panel-corner adjustment is the planned next addition).
- **Audio:** Whisper (faster-whisper backend) transcribes with word timestamps; the 3 chapter texts
  (step 2) are aligned to the transcript to place **4 edges** — story start, the two chapter cuts,
  and story end — so adult preamble before and chatter after the story are trimmed off, not just the
  interior cuts. The recording is split + silence-trimmed into 3 clips.

`--in` can point at any folder of already-tagged files (it discovers them by `pair-NNN` token
regardless of folder layout). Already-built ids are skipped; use `--only` to force a rebuild.

> **Performance:** the `auto` gate runs the fast OpenCV `fill` on every image (~0.2 s) and only
> spends a BiRefNet `matte` pass (~1–2 min on a Mac CPU via onnxruntime) where it can help — when
> fill failed or the art is bold/shaded. So a batch of mostly faint line drawings flies; only the
> shaded ones pay the model cost. Whisper transcription is a few seconds per clip. `process.py` is
> resumable — leave it running, stop/restart freely, it skips finished ids. On a Linux+NVIDIA box
> install `onnxruntime-gpu` + CUDA torch (SETUP) to speed up the matte/Whisper passes. The reviewer's
> per-tile **Recompute** is on-demand, so review stays snappy.

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
- **Image:** the badge by the **mask** dropdown shows the hands-off decision (`workbook/plain`, the
  chosen `cut: fill|matte` or `CROP (not cuttable)`, and `border-trimmed`). Usually nothing to do —
  the gate already picked. To override, drag the 2 dotted **divider lines**, pick a **mask** method
  (`auto` re-runs the gate; `crop` forces solid-bg thirds; `fill`/`matte`/`luma` force a cut-out),
  optionally set **fill leaks** → **Recompute tiles**. Click a tile to **brush** its mask (Erase
  stray background / Restore missing strokes). For tricky cutouts toggle **SAM click** and click on
  the figure (shift-click = background) to get a clean segmentation, then apply.
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
- Each PNG: square, has alpha, background gone, pencil retained (`sips -g hasAlpha -g pixelWidth`).
- Each WAV: starts/ends on its chapter, silence trimmed; durations present in `manifest.json`.
- Every `pair-NNN` appears once per relevant asset folder; manifest links all 6.
