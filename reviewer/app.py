#!/usr/bin/env python3
"""
reviewer/app.py — human review/correct tool for Stage 2 (FastAPI backend).

Per story id it lets a person:
  - drag the 2 divider lines on the rectified panel and recompute the 3 tiles,
  - fix a tile's mask (brush erase/restore in the browser, or SAM-2 point refine on the panel),
  - drag the 2 audio cut points and re-split,
  - play each of the 3 audio clips,
  - Save -> marks the id reviewed so export.py will include it.

Run:  uvicorn reviewer.app:app --app-dir exquisite-kit --port 8765
Env:  WORK=<work dir>  (default ./WORK)
"""
from __future__ import annotations

import io
import json
import os
import sys

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
WORK = os.environ.get("WORK", "WORK")

app = FastAPI(title="Exquisite Stories reviewer")

# Lazy/global heavy handles
_sam = None


def item_dir(tid: str) -> str:
    d = os.path.join(WORK, tid)
    if not os.path.isdir(d):
        raise HTTPException(404, f"unknown id {tid}")
    return d


def load_status(tid: str) -> dict:
    return json.load(open(os.path.join(item_dir(tid), "status.json")))


def save_status(tid: str, status: dict) -> None:
    with open(os.path.join(item_dir(tid), "status.json"), "w") as fh:
        json.dump(status, fh, indent=2)


# ------------------------------------------------------------------------------------- pages/api
@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return open(os.path.join(HERE, "static", "index.html")).read()


@app.get("/api/ids")
def ids():
    out = []
    for tid in sorted(os.listdir(WORK)) if os.path.isdir(WORK) else []:
        sp = os.path.join(WORK, tid, "status.json")
        if os.path.exists(sp):
            s = json.load(open(sp))
            out.append({"id": tid, "reviewed": s.get("reviewed", False),
                        "unpaired": s.get("unpaired", False),
                        "has_image": bool(s.get("image")), "has_audio": bool(s.get("audio"))})
    return out


@app.get("/api/item/{tid}")
def item(tid: str):
    status = load_status(tid)
    resp = {"id": tid, "status": status}
    d = item_dir(tid)
    img_state = os.path.join(d, "image", "image_state.json")
    aud_state = os.path.join(d, "audio", "audio_state.json")
    if os.path.exists(img_state):
        resp["image_state"] = json.load(open(img_state))
    if os.path.exists(aud_state):
        resp["audio_state"] = json.load(open(aud_state))
    return resp


@app.get("/api/file/{tid}/{kind}/{name}")
def file(tid: str, kind: str, name: str):
    if "/" in name or ".." in name:
        raise HTTPException(400, "bad name")
    path = os.path.join(item_dir(tid), kind, name)
    if not os.path.exists(path):
        raise HTTPException(404, name)
    return FileResponse(path)


@app.post("/api/retile/{tid}")
async def retile(tid: str, req: Request):
    """Body: {dividers:[f1,f2], method:"fill"|"matte"|"luma"}. Recompute the 3 tiles from the panel."""
    import cv2
    import image_ops
    body = await req.json()
    dividers = sorted(float(x) for x in body["dividers"])
    method = body.get("method", "fill")
    gap_seal = body.get("gap_seal", "auto")          # "auto" | int radius | 0
    if isinstance(gap_seal, str) and gap_seal.isdigit():
        gap_seal = int(gap_seal)
    d = item_dir(tid)
    panel = cv2.imread(os.path.join(d, "image", "panel.png"), cv2.IMREAD_COLOR)
    if panel is None:
        raise HTTPException(400, "no panel")
    res = image_ops.retile(panel, dividers, os.path.join(d, "image"), method=method, debug=True,
                           gap_seal=gap_seal)
    dec = res.get("decision") or {}
    st_path = os.path.join(d, "image", "image_state.json")
    st = json.load(open(st_path)); st["dividers"] = dividers; st["method"] = method
    st["gap_seal"] = gap_seal
    st["cut_method"] = res["method"]                  # resolved (fill/matte/crop)
    if "cuttable" in dec:
        st["cuttable"] = dec["cuttable"]
        st["scores"] = dec.get("candidates")
    json.dump(st, open(st_path, "w"), indent=2)
    return {"ok": True, "dividers": dividers, "method": method,
            "cut_method": res["method"], "gap_seal": gap_seal}


@app.post("/api/savetile/{tid}/{part}")
async def savetile(tid: str, part: str, req: Request):
    """Body = raw PNG bytes of an edited tile (browser brush result). Overwrites the tile."""
    if part not in ("top", "middle", "bottom"):
        raise HTTPException(400, "bad part")
    data = await req.body()
    with open(os.path.join(item_dir(tid), "image", f"{part}.png"), "wb") as fh:
        fh.write(data)
    return {"ok": True}


@app.post("/api/sam/{tid}")
async def sam(tid: str, req: Request):
    """SAM-2 point refine on the panel. Body: {points:[[x,y,label],...]}. Returns a PNG mask.
    Falls back to 503 if SAM 2 isn't installed (the UI then uses the brush)."""
    import cv2
    import numpy as np
    global _sam
    body = await req.json()
    pts = body.get("points", [])
    panel = cv2.imread(os.path.join(item_dir(tid), "image", "panel.png"), cv2.IMREAD_COLOR)
    if panel is None:
        raise HTTPException(400, "no panel")
    try:
        if _sam is None:
            from sam2.build_sam import build_sam2  # type: ignore
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
            ckpt = os.path.join(HERE, "..", "checkpoints", "sam2.1_hiera_base_plus.pt")
            cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
            _sam = SAM2ImagePredictor(build_sam2(cfg, ckpt))
        _sam.set_image(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        coords = np.array([[p[0], p[1]] for p in pts], dtype="float32")
        labels = np.array([int(p[2]) for p in pts], dtype="int32")
        masks, scores, _ = _sam.predict(point_coords=coords, point_labels=labels, multimask_output=True)
        m = (masks[int(np.argmax(scores))] * 255).astype("uint8")
        ok, buf = cv2.imencode(".png", m)
        return Response(content=buf.tobytes(), media_type="image/png")
    except Exception as e:
        raise HTTPException(503, f"SAM2 unavailable: {e}")


@app.post("/api/resplit/{tid}")
async def resplit(tid: str, req: Request):
    """Body: {edges:[e0,e1,e2,e3]}. Re-cut the 3 audio clips (story start/end + 2 interior cuts)."""
    import audio_ops
    body = await req.json()
    edges = sorted(float(x) for x in body["edges"])
    d = item_dir(tid)
    ast_p = os.path.join(d, "audio", "audio_state.json")
    ast = json.load(open(ast_p))
    audio_ops.split_and_trim(ast["source"], edges, os.path.join(d, "audio"))
    ast["edges"] = edges; ast["method"] = "manual"
    json.dump(ast, open(ast_p, "w"), indent=2)
    return {"ok": True, "edges": edges}


@app.post("/api/chapters/{tid}")
async def chapters(tid: str, req: Request):
    """Body: {chapters:[c1,c2,c3]}. Lets the orchestrator/human supply chapter texts."""
    body = await req.json()
    d = item_dir(tid)
    aud = os.path.join(d, "audio")
    os.makedirs(aud, exist_ok=True)
    json.dump({"chapters": body["chapters"]}, open(os.path.join(aud, "chapters.json"), "w"), indent=2)
    return {"ok": True}


@app.post("/api/save/{tid}")
def save(tid: str):
    status = load_status(tid)
    status["reviewed"] = True
    save_status(tid, status)
    return {"ok": True}
