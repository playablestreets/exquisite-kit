#!/usr/bin/env python3
"""
image_ops.py — image side of Stage 2.

Turns one photo of a story page into three square tiles (head / torso / legs) that the website
re-stacks into a character. The approach is deliberately simple and "human-eye":

  1. locate_panel()    — find the right-page drawing panel (axis-aligned rect in the ORIGINAL image).
  2. detect_dividers() — read the 2 printed dotted rules (a sanity/anchor signal, not a hard cut).
  3. character_bbox()  — bounding box of the drawn character's ink inside the panel.
  4. fit_target_box()  — grow that to a 1:3 box (three stacked squares) with a little padding,
                         anchored by the dividers when they're sensible.
  5. box_confidence()  — score the deterministic box; only LOW-confidence ("hard") pages escalate
                         to the local `claude` CLI (see vision.py) for a human-judgement box.
  6. slice_and_resize()— cut the box into three EQUAL squares and write 1024x1024 opaque PNGs of
                         the raw paper (no transparency, no background fill — so no mismatched borders).

`propose()` runs the whole thing and writes editable state for the reviewer, which can later call
`retile()` with a human-adjusted box. Everything is axis-aligned in original-image pixels, so the
box is the single thing both the OpenCV path, Claude, and the reviewer all agree on.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

import cv2
import numpy as np

PARTS = ["top", "middle", "bottom"]


# --------------------------------------------------------------------------- small shared helpers
def _ink_map(gray: np.ndarray) -> np.ndarray:
    """Binary map of the drawn lines: dark adaptive-threshold ink OR Canny edges (catches faint pencil)."""
    at = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 25, 9)
    edges = cv2.Canny(gray, 30, 90)
    return cv2.bitwise_or(at, edges)


def _keep_components(bw: np.ndarray, min_frac: float = 0.004) -> np.ndarray:
    """Keep connected masses above a fraction of the frame area (a figure may be a few masses)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    out = np.zeros_like(bw)
    thr = int(min_frac * bw.size)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= thr:
            out[lab == i] = 255
    return out


def _clampi(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(v))))


# ----------------------------------------------------------------------------- 1. locate the panel
def _panel_valid(cx: float, w: float, short: float, long_: float):
    """Score a candidate rectangle as 'the right-hand drawing panel', or None if it can't be one.

    The drawing panel is a tall portrait box on the RIGHT page: its centre sits right of mid-page,
    its short side spans roughly a third-to-half of the image width (NOT the whole spread / left
    page), and it is clearly taller than wide. Returns area as the score (prefer the larger valid)."""
    ar = long_ / max(short, 1.0)
    if cx < 0.45 * w:               # must be on the right page, not the spread centre / left page
        return None
    if not (0.27 * w <= short <= 0.58 * w):   # a real panel is ~a third of the page wide:
        return None                           # reject narrow partial strips AND whole-spread grabs
    if not (2.0 <= ar <= 4.3):      # the panel is tall and narrow (≈2.7); reject square spreads
        return None
    return long_ * short


def locate_panel(img_bgr: np.ndarray) -> tuple[list[int], dict]:
    """Find the right-page drawing panel as an axis-aligned rect [x0,y0,x1,y1] in ORIGINAL image
    coords. We look for the printed frame's contour on the right page (scored by _panel_valid),
    take its bounding box, and inset a few px so the printed border isn't included. If nothing
    qualifies we fall back to the right half (flagged so the confidence gate escalates the page)."""
    h, w = img_bgr.shape[:2]
    gray = cv2.bilateralFilter(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), 7, 60, 60)
    edges = cv2.dilate(cv2.Canny(gray, 40, 130), np.ones((3, 3), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best = None  # (score, method, contour)
    for c in cnts:
        if cv2.contourArea(c) < 0.05 * h * w:
            continue
        (cx, _cy), (rw, rh), _ang = cv2.minAreaRect(c)
        short, long_ = min(rw, rh), max(rw, rh)
        score = _panel_valid(cx, w, short, long_)
        if score is None:
            continue
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        method = "quad" if (len(approx) == 4 and cv2.isContourConvex(approx)) else "rrect"
        if best is None or score > best[0]:
            best = (score, method, c)

    if best is None:
        x0 = int(0.50 * w)
        return [x0, 0, w, h], {"method": "fallback-right-half"}

    x, y, rw, rh = cv2.boundingRect(best[2])
    inset = max(2, int(round(0.01 * min(rw, rh))))      # drop the printed border line itself
    rect = [x + inset, y + inset, x + rw - inset, y + rh - inset]
    return rect, {"method": best[1]}


# --------------------------------------------------------------------------- 2. divider detection
def detect_dividers(panel_bgr: np.ndarray) -> list[float]:
    """Return 2 divider y-positions as fractions of panel height. The printed dotted rules appear as
    rows with high dark-pixel density; we take the 2 strongest interior peaks, biased toward the
    1/3 and 2/3 lines (the template's nominal positions)."""
    h, w = panel_bgr.shape[:2]
    if h < 6 or w < 6:
        return [1 / 3.0, 2 / 3.0]
    gray = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 31, 12)
    m = int(0.06 * h)
    row_density = bw[:, int(0.08 * w):int(0.92 * w)].sum(axis=1).astype("float32")
    row_density[:m] = 0
    row_density[-m:] = 0
    row_density = cv2.GaussianBlur(row_density.reshape(-1, 1), (1, 9), 0).ravel()

    picks = []
    for ny in (h / 3.0, 2.0 * h / 3.0):
        lo, hi = int(ny - 0.12 * h), int(ny + 0.12 * h)
        band = row_density[lo:hi]
        picks.append((lo + int(np.argmax(band))) / h if band.size else ny / h)
    return sorted(min(max(p, 0.05), 0.95) for p in picks)


# ----------------------------------------------------------------------- 3. character bounding box
def character_bbox(img_bgr: np.ndarray, panel_rect: list[int],
                   dividers: list[float]) -> tuple[list[int], dict]:
    """Bounding box of the drawn character's ink, in ORIGINAL image coords. Works inside the panel,
    drops the printed frame (outer inset) and suppresses the printed dotted divider rows so they
    don't inflate the box. Falls back to the inset panel if too little ink is found."""
    px0, py0, px1, py1 = panel_rect
    panel = img_bgr[py0:py1, px0:px1]
    ph, pw = panel.shape[:2]
    info: dict = {}
    if ph < 8 or pw < 8:
        return list(panel_rect), {"reason": "panel-too-small"}

    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    ink = _keep_components(_ink_map(gray), 0.0008)

    inset = max(3, int(round(0.045 * min(ph, pw))))     # drop residual printed frame
    ink[:inset, :] = 0; ink[-inset:, :] = 0; ink[:, :inset] = 0; ink[:, -inset:] = 0
    band = max(2, int(round(0.015 * ph)))               # suppress the printed dotted divider rows
    for d in dividers:
        yc = int(round(d * ph))
        ink[max(0, yc - band):min(ph, yc + band), :] = 0

    ys, xs = np.where(ink > 0)
    if len(xs) < 50:
        info["reason"] = "too-little-ink"
        return [px0 + inset, py0 + inset, px1 - inset, py1 - inset], info
    bx0, bx1 = int(xs.min()), int(xs.max())
    by0, by1 = int(ys.min()), int(ys.max())
    info["ink_px"] = int(len(xs))
    return [px0 + bx0, py0 + by0, px0 + bx1, py0 + by1], info


# ------------------------------------------------------------------------- 4. fit the 1:3 crop box
def fit_target_box(img_shape: tuple[int, int], content_bbox: list[int],
                   anchors: list[float] | None = None, pad_frac: float = 0.06,
                   bounds: list[int] | None = None) -> list[int]:
    """Grow content_bbox (orig coords) into a 1:3 box (height = 3*width — three stacked squares),
    centred horizontally on the content, padded a little, kept INSIDE `bounds`. `bounds` is the
    rectangle the box may not exceed — the inset workbook panel border (so we never spill into the
    printed frame / page margin), or the whole image when there's no panel. `anchors` are two
    absolute y's (the dotted dividers, or Claude's neck/waist) used to place the box vertically so
    the equal-thirds cut lines land on them; otherwise the box is centred on the content."""
    H_img, W_img = img_shape[:2]
    bx0, by0, bx1, by1 = bounds if bounds is not None else [0, 0, W_img, H_img]
    bw, bh = max(1, bx1 - bx0), max(1, by1 - by0)

    x0, y0, x1, y1 = content_bbox
    cw, ch = max(1, x1 - x0), max(1, y1 - y0)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    cw *= (1 + 2 * pad_frac); ch *= (1 + 2 * pad_frac)

    # largest 1:3 box that fits the bounds (so a panel-filling figure crops to the panel itself,
    # tight inside its border, instead of spilling past it and getting clamped to the page edge)
    W = max(cw, ch / 3.0)
    W = min(W, float(bw), bh / 3.0)
    H = 3.0 * W

    # vertical placement: anchor on the two divider/cut lines if they look like real thirds
    top = cy - H / 2.0
    if anchors and len(anchors) == 2:
        a1, a2 = sorted(anchors)
        if a2 - a1 > 0.15 * H:                          # plausible spacing for two third-lines
            top = ((a1 - H / 3.0) + (a2 - 2.0 * H / 3.0)) / 2.0

    iW, iH = int(round(W)), int(round(H))
    left = _clampi(cx - W / 2.0, bx0, bx1 - iW)
    top = _clampi(top, by0, by1 - iH)
    return [left, top, left + iW, top + iH]


# ----------------------------------------------------------------- 5. confidence gate (escalation)
def box_confidence(panel_info: dict, char_bbox: list[int], box: list[int],
                   dividers: list[float], panel_rect: list[int]) -> tuple[float, list[str]]:
    """Score the deterministic crop 0..1; low scores escalate to Claude. Cheap signals only: did we
    find a real panel (vs the right-half fallback), is there a sensible amount of character ink, do
    the dotted dividers sit near 1/3 & 2/3, and does the box stay inside the panel."""
    reasons: list[str] = []
    score = 1.0
    px0, py0, px1, py1 = panel_rect
    pw, ph = max(1, px1 - px0), max(1, py1 - py0)

    if panel_info.get("method") == "fallback-right-half":
        score -= 0.5; reasons.append("panel-not-detected")

    cw = char_bbox[2] - char_bbox[0]
    chh = char_bbox[3] - char_bbox[1]
    if cw / pw < 0.12:
        score -= 0.3; reasons.append("character-very-narrow")
    if chh / ph < 0.30:
        score -= 0.3; reasons.append("character-short")
    if cw / pw > 0.98 and chh / ph > 0.98:
        score -= 0.2; reasons.append("ink-fills-panel")

    if len(dividers) == 2 and (0.20 < dividers[0] < 0.45) and (0.55 < dividers[1] < 0.80):
        pass
    else:
        score -= 0.15; reasons.append("dividers-off-nominal")

    if box[0] < px0 - 0.02 * pw or box[2] > px1 + 0.02 * pw:
        score -= 0.1; reasons.append("box-exceeds-panel")

    return max(0.0, min(1.0, score)), reasons


# ------------------------------------------------------------------------- 6. slice + write tiles
def _write_tile(path: str, bgr: np.ndarray, out_size: int) -> None:
    """Resize a section to out_size square and write an opaque RGBA PNG (alpha=255)."""
    sq = cv2.resize(bgr, (out_size, out_size), interpolation=cv2.INTER_AREA)
    bgra = cv2.cvtColor(sq, cv2.COLOR_BGR2BGRA)
    cv2.imwrite(path, bgra)


def slice_and_resize(img_bgr: np.ndarray, box: list[int], out_dir: str,
                     out_size: int = 1024) -> dict[str, str]:
    """Crop `box` from the full image, split into three EQUAL-height squares, resize each to
    out_size, and write top/middle/bottom.png (opaque, raw paper kept)."""
    os.makedirs(out_dir, exist_ok=True)
    H_img, W_img = img_bgr.shape[:2]
    x0, y0, x1, y1 = (max(0, box[0]), max(0, box[1]), min(W_img, box[2]), min(H_img, box[3]))
    crop = img_bgr[y0:y1, x0:x1]
    h = crop.shape[0]
    cuts = [0, round(h / 3.0), round(2.0 * h / 3.0), h]
    written: dict[str, str] = {}
    for i, part in enumerate(PARTS):
        sec = crop[cuts[i]:cuts[i + 1]]
        if sec.size == 0:
            sec = np.full((out_size, out_size, 3), 255, np.uint8)
        p = os.path.join(out_dir, f"{part}.png")
        _write_tile(p, sec, out_size)
        written[part] = p
    return written


# ------------------------------------------------------------------------------------ orchestrate
@dataclass
class ImageState:
    source: str
    source_size: list           # [w, h] of the original image (box is in original pixels)
    kind: str                   # "workbook" | "plain" | "unknown"
    box: list                   # [x0,y0,x1,y1] final 1:3 crop box (the editable artifact)
    panel_rect: list            # [x0,y0,x1,y1] located drawing panel
    char_bbox: list             # [x0,y0,x1,y1] detected character ink box
    dividers: list              # 2 detected divider fractions of panel height (informational)
    source_engine: str          # "opencv" | "claude"
    confidence: float           # gate score for the opencv box
    gate_reasons: list          # why it (would have) escalated
    claude: dict | None = None  # raw Claude response when escalated
    out_size: int = 1024


def _norm_box_to_px(nb: dict, W: int, H: int) -> list[int]:
    return [int(round(nb["x0"] * W)), int(round(nb["y0"] * H)),
            int(round(nb["x1"] * W)), int(round(nb["y1"] * H))]


def _write_source_preview(img_bgr: np.ndarray, out_dir: str, max_w: int = 1000) -> None:
    """A downscaled copy of the page for the reviewer to draw the box over (box coords stay in
    original pixels; the UI scales using source_size)."""
    h, w = img_bgr.shape[:2]
    if w > max_w:
        img_bgr = cv2.resize(img_bgr, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(out_dir, "source.png"), img_bgr)


def retile(img_bgr: np.ndarray, box: list[int], out_dir: str, out_size: int = 1024) -> dict:
    """(Re)generate the 3 square tiles from an (adjusted) box. Returns {parts, box}."""
    parts = slice_and_resize(img_bgr, box, out_dir, out_size=out_size)
    return {"parts": parts, "box": box}


def propose(image_path: str, out_dir: str, use_claude: bool = True,
            claude_threshold: float = 0.55, force_claude: bool = False) -> ImageState:
    """Full pass for one image. Runs the cheap deterministic crop; escalates to the local `claude`
    CLI only when the confidence gate fails (or force_claude). Writes the 3 tiles, a source preview
    for the reviewer, and image_state.json."""
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    H, W = img.shape[:2]

    panel_rect, pinfo = locate_panel(img)
    px0, py0, px1, py1 = panel_rect
    dividers = detect_dividers(img[py0:py1, px0:px1])
    char_bbox, _ink = character_bbox(img, panel_rect, dividers)

    anchors = [py0 + d * (py1 - py0) for d in dividers]
    # On a real workbook panel, keep the box inside the panel's (inset) inner border so a
    # panel-filling figure crops tight to the border instead of spilling into the printed frame.
    bounds = None
    if pinfo.get("method") != "fallback-right-half":
        ins = int(round(0.02 * min(px1 - px0, py1 - py0)))
        bounds = [px0 + ins, py0 + ins, px1 - ins, py1 - ins]
    box = fit_target_box((H, W), char_bbox, anchors=anchors, bounds=bounds)
    conf, reasons = box_confidence(pinfo, char_bbox, box, dividers, panel_rect)

    engine = "opencv"
    kind = "workbook" if pinfo.get("method") != "fallback-right-half" else "unknown"
    claude_raw = None
    if force_claude or (use_claude and conf < claude_threshold):
        import vision
        res = vision.box_from_claude(image_path)
        if res:
            cbox = _norm_box_to_px(res["box"], W, H)
            cuts = res.get("cuts")
            canch = [cuts["neck"] * H, cuts["waist"] * H] if isinstance(cuts, dict) else None
            box = fit_target_box((H, W), cbox, anchors=canch, pad_frac=0.0)
            engine = "claude"
            kind = res.get("kind", kind)
            claude_raw = res

    _write_source_preview(img, out_dir)
    retile(img, box, out_dir)

    state = ImageState(source=os.path.abspath(image_path), source_size=[W, H], kind=kind,
                       box=box, panel_rect=panel_rect, char_bbox=char_bbox, dividers=dividers,
                       source_engine=engine, confidence=round(conf, 3), gate_reasons=reasons,
                       claude=claude_raw)
    with open(os.path.join(out_dir, "image_state.json"), "w") as fh:
        json.dump(asdict(state), fh, indent=2)
    return state


if __name__ == "__main__":  # tiny manual test: python image_ops.py <image> [<out_dir>] [--no-claude]
    import sys
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    st = propose(args[0], args[1] if len(args) > 1 else "._imgtest",
                 use_claude="--no-claude" not in sys.argv,
                 force_claude="--claude" in sys.argv)
    print(json.dumps(asdict(st), indent=2))
