#!/usr/bin/env python3
"""
reviewer/app.py — human review/correct tool for Stage 2 (FastAPI backend).

Per story id it lets a person:
  - drag/resize the crop box over the source photo and recompute the 3 tiles,
  - drag the 2 audio cut points and re-split,
  - play each of the 3 audio clips,
  - Save -> marks the id reviewed so export.py will include it.

Run:  uvicorn reviewer.app:app --app-dir exquisite-kit --port 8765
Env:  WORK=<work dir>  (default ./WORK)
"""
from __future__ import annotations

import json
import os
import sys

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
WORK = os.environ.get("WORK", "WORK")

app = FastAPI(title="Exquisite Stories reviewer")


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
    """Body is EITHER {box:[x0,y0,x1,y1]} — one crop box sliced into equal thirds — OR
    {boxes:{top:[...],middle:[...],bottom:[...]}} — three INDEPENDENT crop boxes, one per part,
    placed anywhere on the page. Optional out_size. All coords in ORIGINAL source-image pixels."""
    import cv2
    import image_ops
    body = await req.json()
    out_size = int(body.get("out_size", 1024))
    d = item_dir(tid)
    img_dir = os.path.join(d, "image")
    st_path = os.path.join(img_dir, "image_state.json")
    st = json.load(open(st_path))
    img = cv2.imread(st["source"], cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "source image unreadable")
    if body.get("boxes"):
        boxes = {p: [int(round(float(v))) for v in body["boxes"][p]] for p in image_ops.PARTS}
        image_ops.retile_boxes(img, boxes, img_dir, out_size=out_size)
        st["boxes"] = boxes; st["out_size"] = out_size; st["source_engine"] = "manual"
        json.dump(st, open(st_path, "w"), indent=2)
        return {"ok": True, "boxes": boxes}
    box = [int(round(float(v))) for v in body["box"]]
    image_ops.retile(img, box, img_dir, out_size=out_size)
    # clear any prior per-part boxes so the state reflects single-box (equal-thirds) mode
    st["box"] = box; st["boxes"] = None; st["out_size"] = out_size; st["source_engine"] = "manual"
    json.dump(st, open(st_path, "w"), indent=2)
    return {"ok": True, "box": box}


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
