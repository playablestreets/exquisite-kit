#!/usr/bin/env python3
"""
image_ops.py — image side of Stage 2.

Pipeline for one photo of a story page:
  1. rectify_panel()  — find the bordered drawing panel on the RIGHT page and perspective-warp it
                        to an upright rectangle (kills camera angle so the dividers are horizontal).
  2. detect_dividers()— locate the 2 printed dotted horizontal lines -> 3 cells (head/body/legs).
  3. mask_region()    — lift the drawing off the paper with a learned matte (BiRefNet via rembg),
                        unioned with an ink-aware pass that recovers faint pencil strokes.
  4. tile_square()    — crop a cell to its content and pad to a transparent square PNG.

`propose()` runs all four and writes proposals + editable state for the reviewer. The reviewer can
later call `retile()` with human-adjusted divider positions and/or an edited mask.

Everything degrades gracefully: if rembg/torch aren't installed the masker falls back to a
luminance+edge key so the rest of the pipeline still runs (lower quality — install per SETUP.md).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

import cv2
import numpy as np

# rembg is optional at import time so `--selfcheck` can report on it without crashing.
try:
    from rembg import new_session, remove  # type: ignore
    _REMBG = True
except Exception:  # pragma: no cover
    _REMBG = False

_SESSION = None
_MODEL_NAME = os.environ.get("REMBG_MODEL", "birefnet-general")


def _session():
    global _SESSION
    if _SESSION is None and _REMBG:
        try:
            _SESSION = new_session(_MODEL_NAME)
        except Exception:
            _SESSION = new_session("u2net")  # fallback model
    return _SESSION


# ----------------------------------------------------------------------------- 1. panel rectify
def _order_corners(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype="float32")  # tl,tr,br,bl


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


def rectify_panel(img_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
    """Detect the right-hand drawing panel and warp it upright. Tries a clean 4-corner quad first
    (corrects perspective); else fits a rotated rectangle to the best contour (corrects skew/rotation
    when the border isn't a clean quad). Falls back to a right-half crop only if nothing qualifies —
    the reviewer can still fix the crop."""
    h, w = img_bgr.shape[:2]
    gray = cv2.bilateralFilter(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), 7, 60, 60)
    edges = cv2.dilate(cv2.Canny(gray, 40, 130), np.ones((3, 3), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best = None  # (score, method, corners[4x2])
    for c in cnts:
        if cv2.contourArea(c) < 0.05 * h * w:
            continue
        (cx, _cy), (rw, rh), _ang = cv2.minAreaRect(c)
        short, long_ = min(rw, rh), max(rw, rh)
        score = _panel_valid(cx, w, short, long_)
        if score is None:
            continue
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            method, corners = "quad", approx.reshape(4, 2).astype("float32")
        else:
            method, corners = "rrect", cv2.boxPoints(cv2.minAreaRect(c)).astype("float32")
        if best is None or score > best[0]:
            best = (score, method, corners)

    if best is None:
        x0 = int(0.50 * w)
        return img_bgr[:, x0:].copy(), {"method": "fallback-right-half", "corners": None}

    corners = _order_corners(best[2])
    (tl, tr, br, bl) = corners
    out_w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    out_h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(corners, dst)
    panel = cv2.warpPerspective(img_bgr, M, (out_w, out_h))
    return panel, {"method": best[1], "corners": corners.tolist()}


def trim_printed_border(panel_bgr: np.ndarray, band: float = 0.16, lenfrac: float = 0.55,
                        pad: float = 0.012, ang: float = 14.0) -> tuple[np.ndarray, dict]:
    """Crop any residual printed workbook border that rectify left inside the panel.

    Rectify on a skewed phone photo often keeps a sliver of the printed frame (and the paper
    *outside* it) along one or two edges. That ink seals the background flood out of the outer
    strip, so it survives as a solid foreground bar (the "border in the cut-out"). We find long,
    near-edge, edge-parallel straight lines with a probabilistic Hough transform — robust to the
    dotted/tilted frame and to the figure touching it — and crop just inside the innermost line on
    each side. Sides with no such line (a clean rectify, e.g. pair-002) are left untouched.
    """
    h, w = panel_bgr.shape[:2]
    gray = cv2.bilateralFilter(cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2GRAY), 7, 55, 55)
    edges = cv2.Canny(gray, 40, 130)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=int(lenfrac * min(h, w)),
                            maxLineGap=int(0.04 * max(h, w)))
    x0, y0, x1, y1 = 0, 0, w, h
    bx, by = band * w, band * h
    if lines is not None:
        for lx0, ly0, lx1, ly1 in lines[:, 0]:
            length = float(np.hypot(lx1 - lx0, ly1 - ly0))
            theta = abs(np.degrees(np.arctan2(ly1 - ly0, lx1 - lx0)))
            horiz, vert = (theta < ang or theta > 180 - ang), (abs(theta - 90) < ang)
            my, mx = (ly0 + ly1) / 2.0, (lx0 + lx1) / 2.0
            if horiz and length > lenfrac * w:
                if my < by:           y0 = max(y0, int(max(ly0, ly1) + pad * h))   # top frame
                elif my > h - by:     y1 = min(y1, int(min(ly0, ly1) - pad * h))   # bottom frame
            elif vert and length > lenfrac * h:
                if mx < bx:           x0 = max(x0, int(max(lx0, lx1) + pad * w))    # left frame
                elif mx > w - bx:     x1 = min(x1, int(min(lx0, lx1) - pad * w))    # right frame
    applied = (x0, y0, x1, y1) != (0, 0, w, h)
    if x1 - x0 < 0.4 * w or y1 - y0 < 0.4 * h:            # refuse an implausible (over-)crop
        return panel_bgr, {"applied": False, "reason": "implausible-crop"}
    return panel_bgr[y0:y1, x0:x1].copy(), {"applied": applied, "crop": [x0, y0, x1, y1],
                                            "orig": [w, h]}


# --------------------------------------------------------------------------- 2. divider detection
def detect_dividers(panel_bgr: np.ndarray) -> list[float]:
    """Return 2 divider y-positions as fractions of panel height. The printed dotted rules appear as
    rows with high dark-pixel density; we take the 2 strongest interior peaks, biased toward the
    1/3 and 2/3 lines (the template's nominal positions)."""
    h, w = panel_bgr.shape[:2]
    gray = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 31, 12)
    # ignore the panel border (outer 6%) so the frame doesn't dominate
    m = int(0.06 * h)
    row_density = bw[:, int(0.08 * w):int(0.92 * w)].sum(axis=1).astype("float32")
    row_density[:m] = 0
    row_density[-m:] = 0
    row_density = cv2.GaussianBlur(row_density.reshape(-1, 1), (1, 9), 0).ravel()

    nominal = [h / 3.0, 2.0 * h / 3.0]
    picks = []
    for ny in nominal:
        lo, hi = int(ny - 0.12 * h), int(ny + 0.12 * h)
        band = row_density[lo:hi]
        if band.size:
            picks.append((lo + int(np.argmax(band)) ) / h)
        else:
            picks.append(ny / h)
    picks = sorted(min(max(p, 0.05), 0.95) for p in picks)
    return picks


# -------------------------------------------------------------------------------- 3. masking
def _luminance_key(region_bgr: np.ndarray) -> np.ndarray:
    """Fallback matte: paper is bright/uniform, drawing is darker/edgier. Returns uint8 alpha."""
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    dark = 255 - cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    edges = cv2.Canny(gray, 30, 90)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), 1)
    alpha = np.maximum(dark, edges).astype("uint8")
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    alpha[alpha < 28] = 0
    return alpha


def _ink_pass(region_bgr: np.ndarray) -> np.ndarray:
    """Adaptive-threshold ink detector to recover thin faint pencil the matte may drop."""
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 25, 10)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return ink


def _ink_map(gray: np.ndarray) -> np.ndarray:
    """Binary map of the drawn lines: dark adaptive-threshold ink OR Canny edges (catches faint pencil)."""
    at = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 25, 9)
    edges = cv2.Canny(gray, 30, 90)
    return cv2.bitwise_or(at, edges)


def _fill_holes(bw: np.ndarray) -> np.ndarray:
    """Fill interior holes of a binary silhouette (flood the outside from a corner, invert, OR)."""
    h, w = bw.shape
    ff = bw.copy()
    m = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, m, (0, 0), 255)
    return cv2.bitwise_or(bw, cv2.bitwise_not(ff))


def _keep_components(bw: np.ndarray, min_frac: float = 0.004) -> np.ndarray:
    """Keep connected masses above a fraction of the frame area (a figure may be a few masses)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    out = np.zeros_like(bw)
    thr = int(min_frac * bw.size)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= thr:
            out[lab == i] = 255
    return out


def _border_labels(lab: np.ndarray) -> set:
    return set(np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])).tolist())


def _interior_pockets(fg: np.ndarray, r: int):
    """Background regions enclosed by the figure except a neck thinner than ~2r. Opening the
    background severs such necks, after which any background component NOT touching the panel border
    is an enclosed pocket (e.g. a belly that leaked out through a gap). Returns (pocket_mask, area).

    This is leak-safe by construction: the border-connected exterior can never be a pocket, so it is
    impossible to flood the whole panel — the worst case is filling a nearly-enclosed region."""
    bg = np.where(fg == 0, np.uint8(255), np.uint8(0))
    ker = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    bg_open = cv2.morphologyEx(bg, cv2.MORPH_OPEN, ker)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bg_open, 8)
    border = _border_labels(lab)
    exterior = np.zeros_like(fg)
    pocket = np.zeros_like(fg)
    for i in range(1, n):
        (exterior if i in border else pocket)[lab == i] = 255
    if not pocket.any():
        return pocket, 0
    # regrow the eroded pocket back toward the outline, but never claim the protected exterior
    grown = cv2.dilate(pocket, ker) & bg
    grown[cv2.dilate(exterior, ker) > 0] = 0
    grown = cv2.bitwise_or(grown, pocket)
    return grown, int((grown > 0).sum())


def _seal_pockets(fg: np.ndarray, r: int) -> np.ndarray:
    """SAFE auto seal: fill interior background pockets enclosed except a neck of radius < ~r. Cannot
    fill the border-connected exterior, so it can never run away. Good as an automatic default."""
    if r <= 0:
        return fg
    pocket, _area = _interior_pockets(fg, r)
    return cv2.bitwise_or(fg, pocket)


def _seal_bridge(fg: np.ndarray, r: int) -> np.ndarray:
    """AGGRESSIVE manual seal: bridge a gap up to ~2r wide and fill what it encloses (dilate the
    mouth -> fill_holes -> erode back). Fills WIDE openings the safe pocket method can't (e.g. a
    missing belly wall), but can over-fill if r is too large — so it is human-driven in the reviewer,
    not automatic. Openings wider than ~2r (between-legs) are still preserved."""
    if r <= 0:
        return fg
    ker = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    return cv2.erode(_fill_holes(cv2.dilate(fg, ker)), ker)


def _auto_seal_radius(fg: np.ndarray, w: int, cap_frac: float = 0.07, min_frac: float = 0.015) -> int:
    """Smallest radius at which a real enclosed pocket emerges (area >= min_frac of the panel), as
    the neck is severed by opening the background. Capped so we never sever a genuinely wide opening
    (between-legs). Well-closed figures never produce a pocket -> 0 (no change)."""
    cap = max(2, int(cap_frac * w))
    thresh = min_frac * fg.size
    for r in range(2, cap + 1):
        _pocket, area = _interior_pockets(fg, r)
        if area >= thresh:
            return r
    return 0


def apply_seal(fg: np.ndarray, gap_seal, w: int) -> tuple[np.ndarray, str]:
    """Dispatch the leak repair. gap_seal: "auto" = safe pocket fill (default, never over-fills);
    a positive int = aggressive bridge at that radius (human-chosen in the reviewer); 0/"off" = none.
    Returns (alpha_fg, info)."""
    if gap_seal in (0, "0", "off", None):
        return fg, "off"
    if gap_seal == "auto":
        r = _auto_seal_radius(fg, w)
        return (_keep_components(_seal_pockets(fg, r)) if r > 0 else fg), f"auto-pocket r={r}"
    r = int(gap_seal)
    return (_keep_components(_seal_bridge(fg, r)) if r > 0 else fg), f"bridge r={r}"


def _fill_foreground(panel_bgr: np.ndarray, gap_seal="auto",
                     debug_dir: str | None = None) -> tuple[np.ndarray, np.ndarray, str]:
    """Core of the scissor cut-out: returns (fg_binary 0/255, ink_map, seal_info).

    Flood from the edges inward: thicken the drawn lines enough to close small gaps in the outline,
    then flood the *background* in from the panel border over everything that isn't a line. Whatever
    the flood can't reach — the area enclosed by the figure's outline — becomes SOLID foreground.
    Factored out of fill_alpha so the binary silhouette can also be scored for cuttability."""
    h, w = panel_bgr.shape[:2]
    gray = cv2.bilateralFilter(cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2GRAY), 7, 55, 55)

    ink = _ink_map(gray)
    # drop the printed panel border so it neither seals the flood nor survives as foreground
    inset = max(3, int(round(0.045 * min(h, w))))
    ink[:inset, :] = 0; ink[-inset:, :] = 0; ink[:, :inset] = 0; ink[:, -inset:] = 0

    # thicken lines to bridge gaps in faint/broken outlines
    k = max(3, (int(round(0.012 * max(h, w))) | 1))
    ink_closed = cv2.morphologyEx(cv2.dilate(ink, np.ones((k, k), np.uint8), 1),
                                  cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))

    # flood the background inward from every free border pixel
    free = np.where(ink_closed == 0, np.uint8(255), np.uint8(0))
    flooded = free.copy()
    ffmask = np.zeros((h + 2, w + 2), np.uint8)
    border = [(x, 0) for x in range(0, w, 4)] + [(x, h - 1) for x in range(0, w, 4)] + \
             [(0, y) for y in range(0, h, 4)] + [(w - 1, y) for y in range(0, h, 4)]
    for (x, y) in border:
        if flooded[y, x] == 255:
            cv2.floodFill(flooded, ffmask, (x, y), 128)
    background = (flooded == 128)

    fg = np.where(background, np.uint8(0), np.uint8(255))   # ink + enclosed interior
    fg = cv2.erode(fg, np.ones((k, k), np.uint8), 1)        # undo the line-thickening
    fg = _fill_holes(fg)
    fg = _keep_components(fg)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # repair interiors that leaked out through a gap in the outline
    fg_sealed, seal_info = apply_seal(fg, gap_seal, w)

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "01_ink.png"), ink)
        cv2.imwrite(os.path.join(debug_dir, "02_ink_closed.png"), ink_closed)
        cv2.imwrite(os.path.join(debug_dir, "03_background.png"), background.astype("uint8") * 255)
        cv2.imwrite(os.path.join(debug_dir, "04_alpha_prefill.png"), fg)
    return fg_sealed, ink, seal_info


def fill_alpha(panel_bgr: np.ndarray, debug_dir: str | None = None, gap_seal="auto") -> np.ndarray:
    """Scissor-cut-out matte for LINE DRAWINGS — a filled silhouette, as if the figure were cut out
    with scissors, rather than a thin tracing of the pencil strokes. See _fill_foreground.

    gap_seal: after the flood, repair interiors that leaked through a gap in the outline (e.g. an
    "implicit" belly line the child never drew). "auto" finds the leak's neck width automatically;
    an int forces that seal radius; 0 disables. Wide openings (between legs) are preserved.
    """
    fg_sealed, _ink, seal_info = _fill_foreground(panel_bgr, gap_seal, debug_dir)
    alpha = cv2.GaussianBlur(fg_sealed, (5, 5), 0)          # soft cut edge
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "04b_alpha.png"), alpha)
        with open(os.path.join(debug_dir, "seal.txt"), "w") as fh:
            fh.write(f"gap_seal={gap_seal} -> {seal_info}\n")
    return alpha


def panel_alpha(panel_bgr: np.ndarray, method: str = "fill", debug_dir: str | None = None,
                gap_seal="auto") -> np.ndarray:
    """Single alpha for the WHOLE rectified panel. 'fill' = scissor cut-out silhouette (best for line
    drawings); 'matte' = BiRefNet learned matte unioned with ink; 'luma' = luminance key. gap_seal
    controls leak repair on the fill path (see fill_alpha)."""
    if method == "fill":
        return fill_alpha(panel_bgr, debug_dir=debug_dir, gap_seal=gap_seal)
    if method == "matte" and _REMBG and _session() is not None:
        try:
            rgba = remove(cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB),
                          session=_session(), post_process_mask=True)
            alpha = np.asarray(rgba)[:, :, 3]
        except Exception:
            alpha = _luminance_key(panel_bgr)
    else:
        alpha = _luminance_key(panel_bgr)
    gray = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2GRAY)
    alpha = np.maximum(alpha, (_ink_map(gray) > 0).astype("uint8") * 255)
    return _keep_components((alpha > 10).astype("uint8") * 255, 0.0005) & alpha


# ----------------------------------------------------------------------- 3b. cuttability decision
# Quality thresholds for "is this a clean cut-out?" — calibrated on the sample panels (see
# PROCESS.md "Cuttability gate"). A clean filled silhouette is solid (high area/convex-hull),
# encloses real interior (filled area well above the raw ink), is one dominant mass, and does not
# blanket the whole panel. A broken-outline drawing leaks the flood inward (enclosure ~1, stringy);
# a scribble buries the panel in ink (ink_frac high); a failed matte goes hollow (solidity ~0).
_CLEAN = dict(solidity=0.40, enclosure=1.25, largest_frac=0.70, n_comp=3,
              fg_lo=0.06, fg_hi=0.85, ink_hi=0.50)


def silhouette_metrics(fg_bin: np.ndarray, ink_bin: np.ndarray) -> dict:
    """Shape-quality metrics for a candidate silhouette (fg_bin, ink_bin are uint8). Returns the
    metrics plus a boolean `clean` and a scalar `score` used to rank fill vs matte."""
    area = fg_bin.size
    fgb = (fg_bin > 127).astype("uint8")
    ink_px = int((ink_bin > 0).sum())
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fgb, 8)
    if n <= 1 or fgb.sum() == 0:
        return dict(fg_frac=0.0, ink_frac=ink_px / area, enclosure=0.0, solidity=0.0,
                    largest_frac=0.0, n_comp=0, clean=False, score=0.0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    big = int(np.argmax(areas)) + 1
    fg_area = int(fgb.sum())
    largest = int(areas.max())
    cnts, _ = cv2.findContours((lab == big).astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull_area = cv2.contourArea(cv2.convexHull(np.vstack(cnts))) if cnts else 1.0
    m = dict(
        fg_frac=fg_area / area,
        ink_frac=ink_px / area,
        enclosure=fg_area / max(ink_px, 1),
        solidity=largest / max(hull_area, 1.0),
        largest_frac=largest / max(fg_area, 1),
        n_comp=int((areas >= 0.004 * area).sum()),
    )
    m["clean"] = bool(
        m["solidity"] >= _CLEAN["solidity"] and m["enclosure"] >= _CLEAN["enclosure"]
        and m["largest_frac"] >= _CLEAN["largest_frac"] and m["n_comp"] <= _CLEAN["n_comp"]
        and _CLEAN["fg_lo"] <= m["fg_frac"] <= _CLEAN["fg_hi"] and m["ink_frac"] <= _CLEAN["ink_hi"]
    )
    m["score"] = 0.5 * m["solidity"] + 0.3 * m["largest_frac"] + 0.2 * min(m["enclosure"], 4.0) / 4.0
    return m


def _interior_shaded_ratio(panel_bgr: np.ndarray, fg_bin: np.ndarray) -> float:
    """Fraction of the filled silhouette's *interior* (eroded, so the outline doesn't count) that is
    dark or colour-saturated rather than blank paper. High -> bold/shaded/coloured art, where the
    learned matte gives cleaner edges than fill; low -> a faint outline drawing only fill can solidify.
    Lets choose_cutout skip the slow matte pass when fill already cuts a faint drawing cleanly."""
    gray = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2GRAY)
    inside = cv2.erode((fg_bin > 127).astype("uint8"), np.ones((9, 9), np.uint8), 1) > 0
    if int(inside.sum()) == 0:
        return 0.0
    hsv = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2HSV)
    nonpaper = (gray < 150) | (hsv[:, :, 1] > 60)
    return float((nonpaper & inside).sum()) / float(inside.sum())


def choose_cutout(panel_bgr: np.ndarray, gap_seal="auto", debug_dir: str | None = None) -> dict:
    """Decide how to background-remove a region, hands-off. Scores the `fill` silhouette and (if the
    BiRefNet model is available) the `matte` silhouette, then:
      * prefer `matte` when it is clean AND actually filled the figure (so we get its cleaner edges
        on bold/shaded drawings, e.g. the shaded cat);
      * else use `fill` when it is clean (best on faint line art with empty interiors);
      * if neither yields a clean silhouette, return method "crop" — the caller then just slices the
        region into thirds on a solid background instead of exporting a broken cut-out.
    Returns {method, cuttable, picked, candidates:{name:metrics}}."""
    gray = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2GRAY)
    ink = (_ink_map(gray) > 0).astype("uint8") * 255

    cand: dict[str, dict] = {}
    fg_fill, _ink2, seal_info = _fill_foreground(panel_bgr, gap_seal, debug_dir)
    cand["fill"] = silhouette_metrics(fg_fill, ink)

    # Only spend a (slow) BiRefNet matte pass where it can actually help: when fill failed (matte may
    # rescue it) or the art is bold/shaded (matte gives cleaner edges). Faint line art that fill cuts
    # cleanly skips matte entirely — the common case, so a batch isn't bottlenecked on the model.
    shaded = _interior_shaded_ratio(panel_bgr, fg_fill)
    run_matte = (_REMBG and _session() is not None
                 and (not cand["fill"]["clean"] or shaded >= 0.30))
    matte_alpha = None
    if run_matte:
        try:
            matte_alpha = panel_alpha(panel_bgr, method="matte")
            cand["matte"] = silhouette_metrics((matte_alpha > 30).astype("uint8") * 255, ink)
        except Exception:
            matte_alpha = None

    fillm = cand["fill"]
    mattem = cand.get("matte")
    picked = "fill"
    if mattem and mattem["clean"] and mattem["fg_frac"] >= 0.6 * max(fillm["fg_frac"], 1e-6):
        picked = "matte"               # matte captured the body -> use its cleaner edges
    elif not fillm["clean"] and mattem and mattem["clean"]:
        picked = "matte"
    cuttable = cand[picked]["clean"]
    # carry the already-computed alpha for the picked method so the caller need not recompute it
    if not cuttable:
        alpha = None
    elif picked == "matte":
        alpha = matte_alpha
    else:
        alpha = cv2.GaussianBlur(fg_fill, (5, 5), 0)
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, "04b_alpha.png"), alpha)
            with open(os.path.join(debug_dir, "seal.txt"), "w") as fh:
                fh.write(f"gap_seal={gap_seal} -> {seal_info}\n")
    return {"method": picked if cuttable else "crop", "cuttable": cuttable,
            "picked": picked, "candidates": cand, "alpha": alpha,
            "shaded_ratio": round(shaded, 3), "ran_matte": bool(run_matte)}


# --------------------------------------------------------------------- 3c. solid-background crop
def _paper_color(region_bgr: np.ndarray) -> tuple[int, int, int]:
    """Estimate the page colour from a thin border frame of the region (median, robust to the
    drawing). Used as the solid background when a figure can't be cleanly cut out."""
    h, w = region_bgr.shape[:2]
    t = max(2, int(0.04 * min(h, w)))
    frame = np.concatenate([region_bgr[:t].reshape(-1, 3), region_bgr[-t:].reshape(-1, 3),
                            region_bgr[:, :t].reshape(-1, 3), region_bgr[:, -t:].reshape(-1, 3)])
    return tuple(int(c) for c in np.median(frame, axis=0))   # BGR


def tile_square_solid(region_bgr: np.ndarray, bg: tuple, pad_frac: float = 0.04) -> np.ndarray:
    """Pad a full region (no masking) to a centred, opaque square on solid `bg` (BGR). The crop
    fallback: when the character can't be cut out, we keep the whole rectangle on a solid page-colour
    background rather than a broken silhouette."""
    h, w = region_bgr.shape[:2]
    side = int(max(h, w) * (1 + 2 * pad_frac))
    canvas = np.zeros((side, side, 4), dtype="uint8")
    canvas[:, :, :3] = cv2.cvtColor(np.uint8([[bg]]), cv2.COLOR_BGR2RGB)[0, 0]
    canvas[:, :, 3] = 255
    oy, ox = (side - h) // 2, (side - w) // 2
    canvas[oy:oy + h, ox:ox + w, :3] = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2RGB)
    return canvas


def mask_region(region_bgr: np.ndarray, use_model: bool = True) -> np.ndarray:
    """Per-cell matte (legacy path). Whole-panel masking via panel_alpha() is now preferred."""
    if use_model and _REMBG and _session() is not None:
        try:
            rgba = remove(cv2.cvtColor(region_bgr, cv2.COLOR_BGR2RGB),
                          session=_session(), post_process_mask=True)
            alpha = np.asarray(rgba)[:, :, 3]
        except Exception:
            alpha = _luminance_key(region_bgr)
    else:
        alpha = _luminance_key(region_bgr)

    alpha = np.maximum(alpha, (_ink_pass(region_bgr) > 0).astype("uint8") * 255)
    # keep only sizeable connected components (drop speckle from paper texture)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((alpha > 10).astype("uint8"), 8)
    keep = np.zeros_like(alpha)
    min_area = max(40, int(0.0005 * alpha.size))
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = alpha[lab == i]
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    rgb = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2RGB)
    return np.dstack([rgb, keep])


# ----------------------------------------------------------------------------- 4. square tiling
def tile_square(rgba: np.ndarray, pad_frac: float = 0.06, bg=None) -> np.ndarray:
    """Crop to alpha bounding box, pad to a centered square. bg=None -> transparent; else (r,g,b)."""
    alpha = rgba[:, :, 3]
    ys, xs = np.where(alpha > 12)
    if len(xs) == 0:
        side = max(rgba.shape[:2]); crop = rgba
    else:
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        crop = rgba[y0:y1 + 1, x0:x1 + 1]
        side = int(max(crop.shape[:2]) * (1 + 2 * pad_frac))
    canvas = np.zeros((side, side, 4), dtype="uint8")
    if bg is not None:
        canvas[:, :, :3] = bg
        canvas[:, :, 3] = 255
    oy = (side - crop.shape[0]) // 2
    ox = (side - crop.shape[1]) // 2
    a = crop[:, :, 3:4].astype("float32") / 255.0
    canvas[oy:oy + crop.shape[0], ox:ox + crop.shape[1], :3] = (
        crop[:, :, :3] * a + canvas[oy:oy + crop.shape[0], ox:ox + crop.shape[1], :3] * (1 - a)
    ).astype("uint8")
    if bg is None:
        canvas[oy:oy + crop.shape[0], ox:ox + crop.shape[1], 3] = np.maximum(
            canvas[oy:oy + crop.shape[0], ox:ox + crop.shape[1], 3], crop[:, :, 3])
    return canvas


# ------------------------------------------------------------------------------------ orchestrate
PARTS = ["top", "middle", "bottom"]


@dataclass
class ImageState:
    source: str
    panel_size: list           # [w, h] of rectified panel / subject region
    dividers: list             # 2 fractions of panel height
    rectify: dict              # info from rectify_panel
    kind: str = "workbook"     # "workbook" (bordered panel) | "plain" (non-workbook drawing)
    trim: dict | None = None   # printed-border trim info (workbook only)
    method: str = "auto"       # masking method requested ("auto" lets the gate decide)
    cut_method: str | None = None   # resolved: "fill" | "matte" | "crop"
    cuttable: bool | None = None    # did a clean cut-out succeed? (False -> cropped on solid bg)
    scores: dict | None = None      # per-candidate cuttability metrics (audit trail)


def _write_png(path: str, rgba: np.ndarray) -> None:
    cv2.imwrite(path, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))


def _resolve_method(method: str | None, use_model) -> str:
    """Back-compat: callers may pass use_model bool. Default is "auto" — let the cuttability gate
    pick fill/matte per image and fall back to crop when nothing cuts cleanly."""
    if method:
        return method
    if use_model is True:
        return "matte"
    return "auto"


def _content_crop(img_bgr: np.ndarray, margin: float = 0.06):
    """Bounding box of the drawn content over a whole page (largest ink masses), with a margin.
    Used for non-workbook single drawings that have no bordered panel to rectify to."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ink = _keep_components(_ink_map(gray), 0.0008)
    ys, xs = np.where(ink > 0)
    if len(xs) < 50:
        return img_bgr.copy(), None
    H, W = gray.shape
    mx, my = int(margin * W), int(margin * H)
    x0, x1 = max(0, int(xs.min()) - mx), min(W, int(xs.max()) + mx)
    y0, y1 = max(0, int(ys.min()) - my), min(H, int(ys.max()) + my)
    return img_bgr[y0:y1, x0:x1].copy(), [x0, y0, x1, y1]


def _looks_like_textpage(img_bgr: np.ndarray, left_frac: float = 0.45) -> bool:
    """Does the left part of the page hold handwriting (many small ink marks)? This is what tells a
    two-page workbook spread (left page = chapters) apart from a single centred drawing."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    W = gray.shape[1]
    left = (_ink_map(gray[:, :int(left_frac * W)]) > 0).astype("uint8")
    n, _lab, stats, _c = cv2.connectedComponentsWithStats(left, 8)
    area = left.size
    small = sum(1 for i in range(1, n) if 2e-5 * area <= stats[i, cv2.CC_STAT_AREA] <= 1e-2 * area)
    return small >= 25                 # many small marks -> a page of writing


def locate_subject(img_bgr: np.ndarray) -> tuple[np.ndarray, str, dict]:
    """Find the region to process and its kind, hands-off.

    * "workbook": rectify finds the bordered right-page panel -> warp it upright, then trim any
      residual printed frame left by an imperfect rectify (see trim_printed_border).
    * "plain": rectify finds no qualifying panel and the page's drawn content does not span both
      pages -> a non-workbook single drawing; crop to its content bounding box.
    A rectify miss whose content DOES span the page is most likely an undetected workbook spread, so
    we keep the conservative right-half fallback rather than cropping across the fold."""
    panel, info = rectify_panel(img_bgr)
    if info.get("method") in ("quad", "rrect"):
        panel, trim = trim_printed_border(panel)
        return panel, "workbook", {"rectify": info, "trim": trim}
    # rectify found no clean panel: either a non-workbook drawing, or a workbook whose corners it
    # couldn't detect (it fell back to a right-half crop). A workbook has a text-heavy left page; a
    # single drawing doesn't -> crop to its drawn content.
    region, bbox = _content_crop(img_bgr)
    if bbox is not None and not _looks_like_textpage(img_bgr):
        return region, "plain", {"rectify": info, "content_bbox": bbox}
    # an (undetected) workbook spread -> keep the right-half fallback, but still trim its frame
    panel, trim = trim_printed_border(panel)
    return panel, "workbook", {"rectify": info, "trim": trim}


def retile(panel_bgr: np.ndarray, dividers: list[float], out_dir: str,
           use_model=None, method: str | None = None, mask_overrides: dict | None = None,
           debug: bool = False, gap_seal="auto") -> dict:
    """(Re)generate the 3 square tiles. With method="auto" the cuttability gate picks fill/matte or
    falls back to "crop"; "crop" slices the region into thirds on a solid page-colour background
    (no transparency) — used when the character can't be cleanly cut out. Otherwise the figure is
    masked ONCE over the whole panel (so the silhouette stays closed head-to-toe), then sliced at the
    dividers and squared. Returns {parts, method (resolved), decision}."""
    method = _resolve_method(method, use_model)
    debug_dir = os.path.join(out_dir, "debug") if debug else None
    decision = None
    reuse_alpha = None
    if method == "auto":
        decision = choose_cutout(panel_bgr, gap_seal=gap_seal, debug_dir=debug_dir)
        effective = decision["method"]
        reuse_alpha = decision.get("alpha")        # already computed during scoring
    else:
        effective = method

    h = panel_bgr.shape[0]
    ys = [0] + [int(round(d * h)) for d in sorted(dividers)] + [h]
    written: dict[str, str] = {}

    if effective == "crop":
        bg = _paper_color(panel_bgr)
        for i, part in enumerate(PARTS):
            tile = tile_square_solid(panel_bgr[ys[i]:ys[i + 1]], bg)
            p = os.path.join(out_dir, f"{part}.png")
            _write_png(p, tile)
            written[part] = p
    else:
        alpha = reuse_alpha if reuse_alpha is not None else \
            panel_alpha(panel_bgr, method=effective, debug_dir=debug_dir, gap_seal=gap_seal)
        rgb = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB)
        panel_rgba = np.dstack([rgb, alpha])
        if debug_dir:
            _write_png(os.path.join(debug_dir, "05_panel_rgba.png"), panel_rgba)
        for i, part in enumerate(PARTS):
            region = panel_rgba[ys[i]:ys[i + 1]].copy()
            if mask_overrides and part in mask_overrides:
                region[:, :, 3] = mask_overrides[part]
            tile = tile_square(region)
            p = os.path.join(out_dir, f"{part}.png")
            _write_png(p, tile)
            written[part] = p
    return {"parts": written, "method": effective, "decision": decision}


def propose(image_path: str, out_dir: str, use_model=None, method: str | None = None,
            debug: bool = True, gap_seal="auto") -> ImageState:
    """Full auto pass for one image -> writes panel.png, debug intermediates, the 3 tile proposals,
    and image_state.json (incl. the cuttability decision). debug=True keeps intermediates for review."""
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    region, kind, loc = locate_subject(img)
    cv2.imwrite(os.path.join(out_dir, "panel.png"), region)
    dividers = detect_dividers(region) if kind == "workbook" else [1 / 3.0, 2 / 3.0]

    req = _resolve_method(method, use_model)
    res = retile(region, dividers, out_dir, method=req, debug=debug, gap_seal=gap_seal)
    dec = res.get("decision") or {}
    state = ImageState(source=os.path.abspath(image_path),
                       panel_size=[region.shape[1], region.shape[0]],
                       dividers=dividers, rectify=loc.get("rectify", {}),
                       kind=kind, trim=loc.get("trim"),
                       method=req, cut_method=res["method"],
                       cuttable=dec.get("cuttable"),
                       scores=dec.get("candidates"))
    with open(os.path.join(out_dir, "image_state.json"), "w") as fh:
        json.dump(asdict(state), fh, indent=2)
    return state


if __name__ == "__main__":  # tiny manual test: python image_ops.py <image> <out_dir>
    import sys
    st = propose(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "._imgtest")
    print(json.dumps(asdict(st), indent=2))
