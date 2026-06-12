#!/usr/bin/env python3
"""
match.py — Stage 1 of the Exquisite Stories pipeline.

Pairs each audio recording with the image taken just before it, using the capture time
*embedded inside* each file (NOT the filesystem timestamp — those are identical across a batch
once files have been copied/unzipped). Gives each pair a shared ``__pair-NNN`` suffix, tags
leftovers ``__nopair``, and MOVES the tagged files from INBOX into PROCESSING.

Why embedded time: audio .m4a carry a ``creation_time`` tag (read with ffprobe); images carry an
EXIF ``DateTimeOriginal`` (read with exiftool, or Pillow as a fallback). Empirically each photo is
snapped a handful of seconds BEFORE its narration starts, so we match an image to the audio that
follows it within a short window.

Portable: uses ffprobe + exiftool/Pillow only (no macOS ``mdls``). Idempotent: files already
carrying a ``__pair-`` / ``__nopair`` token are skipped, so re-running is a no-op.

Usage:
    python match.py --inbox INBOX --out PROCESSING
    python match.py --inbox INBOX --out PROCESSING --dry-run     # report pairs, move nothing
    python match.py --inbox INBOX --out PROCESSING --window 25   # max secs image-before-audio
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass

AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".aac", ".mp4", ".m4b"}
IMAGE_EXTS = {".jpeg", ".jpg", ".png", ".heic", ".tif", ".tiff", ".avif"}

TAG_RE = re.compile(r"__(pair-\d+|nopair)$")

# Image is captured a few seconds BEFORE the audio it belongs to. Accept this directional window
# (audio_time - image_time) in seconds. Validated on the first batch: 2..15s typical; 25 is slack.
DEFAULT_WINDOW = 25.0
DEFAULT_BACK = -3.0  # allow tiny negative (clock jitter) so image marginally after audio still pairs


# --------------------------------------------------------------------------------------------- io
def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def is_tagged(name: str) -> bool:
    stem, _ = os.path.splitext(name)
    return bool(TAG_RE.search(stem))


def audio_capture_time(path: str) -> dt.datetime | None:
    """Embedded recording time from container metadata via ffprobe."""
    out = run([
        "ffprobe", "-v", "quiet",
        "-show_entries", "format_tags=creation_time",
        "-of", "default=nw=1:nk=1", path,
    ])
    if not out:
        return None
    out = out.strip().replace("Z", "+00:00")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return dt.datetime.strptime(out, fmt)
        except ValueError:
            continue
    try:
        return dt.datetime.fromisoformat(out)
    except ValueError:
        return None


def _parse_exif_dt(s: str, offset: str | None = None) -> dt.datetime | None:
    s = s.strip()
    if not s or s.startswith("0000"):
        return None
    # EXIF format: "2026:05:18 00:02:45" (optionally with sub-seconds / offset)
    s = s.split(".")[0]
    try:
        base = dt.datetime.strptime(s, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        try:
            base = dt.datetime.fromisoformat(s)
        except ValueError:
            return None
    if offset and re.match(r"^[+-]\d{2}:?\d{2}$", offset.strip()):
        off = offset.strip().replace(":", "")
        sign = 1 if off[0] == "+" else -1
        hrs, mins = int(off[1:3]), int(off[3:5])
        base = base.replace(tzinfo=dt.timezone(sign * dt.timedelta(hours=hrs, minutes=mins)))
    else:
        base = base.replace(tzinfo=dt.timezone.utc)
    return base


def image_capture_time(path: str) -> dt.datetime | None:
    """EXIF DateTimeOriginal — exiftool preferred, Pillow fallback."""
    if shutil.which("exiftool"):
        out = run([
            "exiftool", "-s3", "-d", "%Y:%m:%d %H:%M:%S",
            "-DateTimeOriginal", "-OffsetTimeOriginal", path,
        ])
        if out:
            lines = out.splitlines()
            ts = lines[0] if lines else ""
            off = lines[1] if len(lines) > 1 else None
            parsed = _parse_exif_dt(ts, off)
            if parsed:
                return parsed
    # Fallback: Pillow
    try:
        from PIL import Image, ExifTags  # noqa: WPS433 (lazy import keeps match usable without PIL)
        tagmap = {v: k for k, v in ExifTags.TAGS.items()}
        exif = Image.open(path).getexif()
        ifd = exif.get_ifd(0x8769) if hasattr(exif, "get_ifd") else {}
        raw = ifd.get(tagmap.get("DateTimeOriginal")) or exif.get(tagmap.get("DateTime"))
        if raw:
            return _parse_exif_dt(str(raw))
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------------------- matching
@dataclass
class Item:
    path: str
    name: str
    t: dt.datetime


def collect(inbox: str, exts: set[str], reader) -> tuple[list[Item], list[str]]:
    items, skipped = [], []
    for root, _dirs, files in os.walk(inbox):
        for f in sorted(files):
            ext = os.path.splitext(f)[1].lower()
            if ext not in exts or f.startswith("."):
                continue
            if is_tagged(f):
                skipped.append(f)
                continue
            p = os.path.join(root, f)
            t = reader(p)
            if t is None:
                print(f"  [warn] no embedded capture time, skipping: {f}", file=sys.stderr)
                continue
            items.append(Item(p, f, t))
    items.sort(key=lambda x: x.t)
    return items, skipped


def pair(audio: list[Item], images: list[Item], window: float, back: float):
    """Directional greedy: each audio takes the closest unused image captured just before it."""
    used: set[int] = set()
    pairs: list[tuple[Item, Item]] = []
    audio_unmatched: list[Item] = []
    for a in audio:
        best = None  # (gap, idx)
        for i, im in enumerate(images):
            if i in used:
                continue
            gap = (a.t - im.t).total_seconds()
            if back <= gap <= window and (best is None or gap < best[0]):
                best = (gap, i)
        if best is not None:
            used.add(best[1])
            pairs.append((a, images[best[1]]))
        else:
            audio_unmatched.append(a)
    image_unmatched = [im for i, im in enumerate(images) if i not in used]
    return pairs, audio_unmatched, image_unmatched


# ------------------------------------------------------------------------------------------ apply
def tagged_name(name: str, token: str) -> str:
    stem, ext = os.path.splitext(name)
    return f"{stem}__{token}{ext}"


def move(src: str, dst_dir: str, new_name: str, dry: bool) -> str:
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, new_name)
    if os.path.exists(dst):
        raise FileExistsError(dst)
    if not dry:
        shutil.move(src, dst)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description="Pair audio+images by embedded capture time, tag, move.")
    ap.add_argument("--inbox", required=True, help="folder of raw audio + image files")
    ap.add_argument("--out", required=True, help="PROCESSING folder to move tagged files into")
    ap.add_argument("--window", type=float, default=DEFAULT_WINDOW, help="max secs image-before-audio")
    ap.add_argument("--back", type=float, default=DEFAULT_BACK, help="min gap (allow small negative)")
    ap.add_argument("--dry-run", action="store_true", help="report only; move nothing")
    args = ap.parse_args()

    print(f"Scanning {args.inbox} …")
    audio, a_skip = collect(args.inbox, AUDIO_EXTS, audio_capture_time)
    images, i_skip = collect(args.inbox, IMAGE_EXTS, image_capture_time)
    if a_skip or i_skip:
        print(f"  (skipped {len(a_skip)+len(i_skip)} already-tagged files)")
    print(f"  {len(audio)} audio, {len(images)} images with embedded times")

    pairs, a_un, i_un = pair(audio, images, args.window, args.back)
    print(f"\n{len(pairs)} pairs | {len(a_un)} audio w/o image | {len(i_un)} images w/o audio")

    plan = []
    audio_out = os.path.join(args.out, "audio")
    image_out = os.path.join(args.out, "images")
    for n, (a, im) in enumerate(pairs, 1):
        tok = f"pair-{n:03d}"
        gap = (a.t - im.t).total_seconds()
        print(f"  {tok}: {a.name}  <-  {im.name}  (+{gap:.0f}s)")
        plan.append((a.path, audio_out, tagged_name(a.name, tok)))
        plan.append((im.path, image_out, tagged_name(im.name, tok)))
    for a in a_un:
        plan.append((a.path, audio_out, tagged_name(a.name, "nopair")))
    for im in i_un:
        plan.append((im.path, image_out, tagged_name(im.name, "nopair")))

    # collision guard before touching anything
    for _src, d, name in plan:
        if os.path.exists(os.path.join(d, name)):
            print(f"[abort] target already exists: {os.path.join(d, name)}", file=sys.stderr)
            return 2

    for src, d, name in plan:
        move(src, d, name, args.dry_run)

    manifest = {
        "pairs": [
            {"pair": f"pair-{n:03d}",
             "audio": tagged_name(a.name, f"pair-{n:03d}"),
             "image": tagged_name(im.name, f"pair-{n:03d}"),
             "gap_seconds": round((a.t - im.t).total_seconds(), 1)}
            for n, (a, im) in enumerate(pairs, 1)
        ],
        "audio_unpaired": [tagged_name(a.name, "nopair") for a in a_un],
        "image_unpaired": [tagged_name(im.name, "nopair") for im in i_un],
    }
    if not args.dry_run:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "match_manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=2)

    verb = "Would move" if args.dry_run else "Moved"
    print(f"\n{verb} {len(plan)} files into {args.out} "
          f"({len(pairs)} pairs + {len(a_un)+len(i_un)} nopair).")
    if not args.dry_run:
        print(f"Wrote {os.path.join(args.out, 'match_manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
