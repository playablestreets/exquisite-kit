#!/usr/bin/env python3
"""
process.py — Stage 2 orchestrator.

Discovers paired (and unpaired) files produced by match, and builds *proposals* into WORK/<id>/ for
the reviewer to confirm/adjust. Also hosts environment utilities used by SETUP.md.

Discovery is tolerant of layout: it searches --in recursively for audio + image files and groups by
the shared `pair-NNN` token, so it works on both `PROCESSING/{audio,images}/` and the existing
`TO_PROCESS/NEW AUDIO MPRG/ + NEW IMAGES MPRG/`.

Usage:
    python process.py --in PROCESSING --work WORK            # build proposals for every pair
    python process.py --in PROCESSING --work WORK --only pair-001,pair-002
    python process.py --selfcheck                            # verify environment
    python process.py --download-models                      # fetch/cache model checkpoints
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".aac", ".mp4"}
IMAGE_EXTS = {".jpeg", ".jpg", ".png", ".heic", ".tif", ".tiff", ".avif"}
TOKEN_RE = re.compile(r"__(pair-\d+|nopair)")


def token_of(name: str) -> str | None:
    m = TOKEN_RE.search(name)
    return m.group(1) if m else None


def discover(root: str) -> dict[str, dict]:
    """token -> {'audio': path|None, 'image': path|None, 'unpaired': bool}. nopair files each get a
    unique synthetic id so they're processed individually."""
    groups: dict[str, dict] = {}
    nop = 0
    for dirpath, _d, files in os.walk(root):
        for f in sorted(files):
            if f.startswith("."):
                continue
            ext = os.path.splitext(f)[1].lower()
            kind = "audio" if ext in AUDIO_EXTS else ("image" if ext in IMAGE_EXTS else None)
            if not kind:
                continue
            tok = token_of(f)
            if tok is None:
                continue
            if tok == "nopair":
                nop += 1
                tok = f"nopair-{kind}-{nop:03d}"
                groups[tok] = {"audio": None, "image": None, "unpaired": True}
            g = groups.setdefault(tok, {"audio": None, "image": None, "unpaired": tok.startswith("nopair")})
            g[kind] = os.path.join(dirpath, f)
    return groups


def build_one(tid: str, g: dict, work: str, method: str) -> dict:
    import image_ops
    import audio_ops
    out = os.path.join(work, tid)
    os.makedirs(out, exist_ok=True)
    status = {"id": tid, "unpaired": g["unpaired"], "image": None, "audio": None, "reviewed": False}

    if g.get("image"):
        img_dir = os.path.join(out, "image")
        st = image_ops.propose(g["image"], img_dir, method=method, debug=True)
        status["image"] = {"dir": img_dir, "source": g["image"], "dividers": st.dividers,
                           "method": method, "kind": st.kind, "cut_method": st.cut_method,
                           "cuttable": st.cuttable}
    if g.get("audio"):
        aud_dir = os.path.join(out, "audio")
        st = audio_ops.propose(g["audio"], g.get("image"), aud_dir)
        status["audio"] = {"dir": aud_dir, "source": g["audio"],
                           "edges": st.edges, "method": st.method,
                           "chapters_known": st.chapters is not None}
    with open(os.path.join(out, "status.json"), "w") as fh:
        json.dump(status, fh, indent=2)
    return status


# ------------------------------------------------------------------------------------- env utils
def selfcheck() -> int:
    import shutil
    ok = True

    def line(good, msg):
        nonlocal ok
        ok = ok and good
        print(f"[{'ok' if good else '!!'}] {msg}")

    line(bool(shutil.which("ffmpeg") and shutil.which("ffprobe")), "ffmpeg / ffprobe found")
    line(bool(shutil.which("exiftool")), "exiftool found (image EXIF; Pillow is fallback)")
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        line(True, f"torch {torch.__version__}  device={dev}")
    except Exception as e:
        line(False, f"torch missing: {e}")
    try:
        from rembg import new_session
        new_session(os.environ.get("REMBG_MODEL", "birefnet-general"))
        line(True, "rembg BiRefNet session created")
    except Exception as e:
        line(False, f"rembg/BiRefNet: {e}")
    try:
        import faster_whisper  # noqa: F401  (primary transcription backend)
        line(True, "faster-whisper importable")
    except Exception as e:
        try:
            import whisperx  # noqa: F401
            line(True, "whisperx importable (faster-whisper missing)")
        except Exception:
            line(False, f"no Whisper backend: {e}")
    ckpt = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    has_sam = any(f.endswith(".pt") for f in os.listdir(ckpt)) if os.path.isdir(ckpt) else False
    line(has_sam, "SAM2 checkpoint present in checkpoints/")
    mode = "anthropic-api" if os.environ.get("ANTHROPIC_API_KEY") else "claude-orchestrator"
    print(f"[ok] chapter reading: {mode}")
    print("SELFCHECK PASSED" if ok else "SELFCHECK INCOMPLETE — see SETUP.md")
    return 0 if ok else 1


def download_models() -> int:
    print("Caching models …")
    try:
        from rembg import new_session
        new_session(os.environ.get("REMBG_MODEL", "birefnet-general"))
        print("  [ok] BiRefNet cached")
    except Exception as e:
        print(f"  [!!] BiRefNet: {e}")
    try:
        from faster_whisper import WhisperModel       # primary transcription backend
        WhisperModel("small", device="cpu", compute_type="int8")
        print("  [ok] faster-whisper(small) cached")
    except Exception as e:
        print(f"  [!!] faster-whisper: {e}")
        try:
            import whisperx
            whisperx.load_model("small", "cpu", compute_type="int8")
            print("  [ok] whisperx(small) cached (fallback)")
        except Exception as e2:
            print(f"  [!!] whisperx: {e2}")
    ckpt = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints"))
    os.makedirs(ckpt, exist_ok=True)
    sam = os.path.join(ckpt, "sam2.1_hiera_base_plus.pt")
    if not os.path.exists(sam):
        url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
        try:
            import urllib.request
            print("  downloading SAM2 checkpoint …")
            urllib.request.urlretrieve(url, sam)
            print("  [ok] SAM2 checkpoint")
        except Exception as e:
            print(f"  [!!] SAM2 download failed ({e}). Fetch manually into checkpoints/:\n      {url}")
    else:
        print("  [ok] SAM2 checkpoint present")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", help="folder of matched/tagged files (PROCESSING)")
    ap.add_argument("--work", default="WORK", help="where proposals are written")
    ap.add_argument("--only", help="comma-separated ids to (re)build, e.g. pair-001,pair-002")
    ap.add_argument("--method", choices=["auto", "fill", "matte", "luma", "crop"], default="auto",
                    help="masking: auto=cuttability gate picks fill/matte else crop (default); "
                         "fill=scissor cut-out (line art); matte=BiRefNet; luma=key; "
                         "crop=no cut, thirds on solid page-colour bg")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--download-models", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        return selfcheck()
    if args.download_models:
        return download_models()
    if not args.inp:
        ap.error("--in is required (unless --selfcheck/--download-models)")

    groups = discover(args.inp)
    ids = sorted(groups)
    if args.only:
        want = set(args.only.split(","))
        ids = [i for i in ids if i in want]
    print(f"Discovered {len(groups)} ids; building {len(ids)} into {args.work}")
    os.makedirs(args.work, exist_ok=True)
    for tid in ids:
        done = os.path.join(args.work, tid, "status.json")
        if os.path.exists(done) and not args.only:
            print(f"  {tid}: already built (skip; use --only to rebuild)")
            continue
        print(f"  {tid}: building …")
        try:
            build_one(tid, groups[tid], args.work, method=args.method)
        except Exception as e:
            print(f"  {tid}: ERROR {e.__class__.__name__}: {e}", file=sys.stderr)
    print("Done. Open the reviewer to confirm/adjust, then run export.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
