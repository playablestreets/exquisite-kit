#!/usr/bin/env python3
"""
export.py — final step of Stage 2.

Collects the reviewed proposals in WORK/<id>/ and sorts them into READY_FOR_DRIVE/ with the 6 asset
folders that mirror EXAMPLE_COMPLETE, naming every file with its `pair-NNN` token so the website can
re-associate a story's head/body/legs and 3 audio clips. Also writes manifest.json — the ingest
index for the website.

By default only ids marked reviewed in status.json are exported (use --include-unreviewed to force).
Staging only: it does not upload to Google Drive — a human does that.

Usage:
    python export.py --work WORK --out READY_FOR_DRIVE
    python export.py --work WORK --out READY_FOR_DRIVE --include-unreviewed
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

IMG_FOLDER = {"top": "IMAGES_TOP", "middle": "IMAGES_MIDDLE", "bottom": "IMAGES_BOTTOM"}
AUD_FOLDER = {"beginning": "AUDIO_BEGINNING", "middle": "AUDIO_MIDDLE", "end": "AUDIO_END"}


def _id_token(tid: str) -> str:
    # pair-007 -> pair-007 ; nopair-image-003 -> nopair-image-003 (kept verbatim, still unique)
    return tid


def export_one(tid: str, work: str, out: str) -> dict | None:
    d = os.path.join(work, tid)
    status_p = os.path.join(d, "status.json")
    if not os.path.exists(status_p):
        return None
    status = json.load(open(status_p))
    tok = _id_token(tid)
    entry = {"id": tid, "unpaired": status.get("unpaired", False), "images": {}, "audio": {},
             "chapters": None, "method": None}

    if status.get("image"):
        idir = status["image"]["dir"]
        for part, folder in IMG_FOLDER.items():
            src = os.path.join(idir, f"{part}.png")
            if os.path.exists(src):
                dst_dir = os.path.join(out, folder)
                os.makedirs(dst_dir, exist_ok=True)
                name = f"{tok}_{part}.png"
                shutil.copy2(src, os.path.join(dst_dir, name))
                entry["images"][part] = f"{folder}/{name}"

    if status.get("audio"):
        adir = status["audio"]["dir"]
        ast = os.path.join(adir, "audio_state.json")
        if os.path.exists(ast):
            a = json.load(open(ast))
            entry["chapters"] = a.get("chapters")
            entry["method"] = a.get("method")
        for part, folder in AUD_FOLDER.items():
            src = os.path.join(adir, f"{part}.wav")
            if os.path.exists(src):
                dst_dir = os.path.join(out, folder)
                os.makedirs(dst_dir, exist_ok=True)
                name = f"{tok}_{part}.wav"
                shutil.copy2(src, os.path.join(dst_dir, name))
                entry["audio"][part] = f"{folder}/{name}"
    return entry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="WORK")
    ap.add_argument("--out", default="READY_FOR_DRIVE")
    ap.add_argument("--include-unreviewed", action="store_true")
    args = ap.parse_args()

    ids = sorted(d for d in os.listdir(args.work) if os.path.isdir(os.path.join(args.work, d)))
    os.makedirs(args.out, exist_ok=True)
    manifest = {"stories": [], "generated_with": "exquisite-kit"}
    exported = skipped = 0
    for tid in ids:
        status_p = os.path.join(args.work, tid, "status.json")
        if not os.path.exists(status_p):
            continue
        reviewed = json.load(open(status_p)).get("reviewed", False)
        if not reviewed and not args.include_unreviewed:
            skipped += 1
            continue
        entry = export_one(tid, args.work, args.out)
        if entry:
            entry["reviewed"] = reviewed
            manifest["stories"].append(entry)
            exported += 1

    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Exported {exported} ids into {args.out} ({skipped} unreviewed skipped).")
    print(f"Wrote {os.path.join(args.out, 'manifest.json')} — ready to upload to Google Drive.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
