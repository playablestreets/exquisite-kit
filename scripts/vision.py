#!/usr/bin/env python3
"""
vision.py — hard-case escalation to the local `claude` CLI.

The image pipeline (image_ops.py) runs a cheap deterministic crop on every photo. Only when that
result fails a confidence gate do we spend tokens asking Claude to look at the page and place the
crop box like a human would. This module is that single call, kept dependency-light (stdlib +
subprocess) so it never weighs the batch down.

`box_from_claude(image_path)` returns a normalized box dict, or None on ANY failure (no CLI,
timeout, malformed output, low confidence) so the caller transparently keeps its OpenCV box.

Set EXQ_CLAUDE_BIN to override the binary, EXQ_CLAUDE_MODEL to pick a model, EXQ_CLAUDE_TIMEOUT
for the per-call timeout (seconds).

NOTE: we deliberately do NOT use the CLI's `--json-schema` flag — combined with the Read tool it
hangs (structured output conflicts with the multi-turn tool use). Instead we ask for a bare JSON
object in the reply text and parse it ourselves (the model reads the image in ~10s this way).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess

_PROMPT = """Read the image at {path}. It is a phone photo of a child's "exquisite corpse" story page.

Most pages are a two-page workbook spread: the LEFT page is handwritten story text (ignore it),
and the RIGHT page is a printed rectangular drawing panel with two faint printed dotted divider
lines that split it into three equal parts for the child's drawn character — head (top), torso
(middle), legs/feet (bottom). Some pages are NOT this workbook: just a plain drawing of a
character. Use your judgement either way; ignore the table, shadows, fingers and background.

Find the rectangle that tightly bounds the DRAWN CHARACTER plus a small margin, positioned and
sized so that splitting it into three EQUAL horizontal thirds puts the head in the top third, the
torso in the middle third, and the legs/feet in the bottom third. For workbook pages, align the
box to the printed panel and its dotted dividers.

Reply with ONLY a JSON object (no prose, no markdown fences) of exactly this shape:
{{"kind": "workbook" | "plain",
  "box": {{"x0": <num>, "y0": <num>, "x1": <num>, "y1": <num>}},
  "cuts": {{"neck": <num>, "waist": <num>}},
  "confidence": <num 0..1>}}
All coordinates are normalized fractions of the FULL image (0..1): x0<x1 left-to-right, y0<y1
top-to-bottom. `cuts.neck`/`cuts.waist` are the normalized y of the head/torso and torso/legs
boundaries you actually see (used only to fine-tune placement). `confidence` is how sure you are
the box frames the whole character with head/torso/legs in the right thirds."""


def _bin() -> str | None:
    return os.environ.get("EXQ_CLAUDE_BIN") or shutil.which("claude")


def available() -> bool:
    """Is the claude CLI on PATH? Used by selfcheck (escalation is optional)."""
    return _bin() is not None


def _valid_box(b: dict) -> bool:
    try:
        x0, y0, x1, y1 = float(b["x0"]), float(b["y0"]), float(b["x1"]), float(b["y1"])
    except (KeyError, TypeError, ValueError):
        return False
    return all(0.0 <= v <= 1.0 for v in (x0, y0, x1, y1)) and x1 > x0 and y1 > y0


def _extract_result(stdout: str) -> dict | None:
    """The --output-format json envelope wraps the model's reply text in `result`. That text should
    be a bare JSON object; tolerate stray prose or ```json fences by grabbing the outermost {...}."""
    try:
        env = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    res = env.get("result", env) if isinstance(env, dict) else env
    if isinstance(res, dict):
        return res
    if not isinstance(res, str):
        return None
    try:
        return json.loads(res)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", res, re.DOTALL)        # first {...} block (handles fences/prose)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None


def box_from_claude(image_path: str, min_confidence: float = 0.4) -> dict | None:
    """Ask the local claude CLI to place the 1:3 crop box. Returns
    {kind, box:{x0,y0,x1,y1}, cuts?:{neck,waist}, confidence, notes} (normalized), or None on any
    failure so the caller keeps its deterministic box."""
    claude = _bin()
    if not claude:
        return None
    timeout = float(os.environ.get("EXQ_CLAUDE_TIMEOUT", "240"))
    cmd = [claude, "-p", _PROMPT.format(path=os.path.abspath(image_path)),
           "--output-format", "json", "--allowedTools", "Read"]
    model = os.environ.get("EXQ_CLAUDE_MODEL")
    if model:
        cmd += ["--model", model]
    try:
        # stdin=DEVNULL: `claude -p` otherwise waits ~3s for piped stdin (and can hang in batch).
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              stdin=subprocess.DEVNULL)
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return None
    if proc.returncode != 0:
        return None
    res = _extract_result(proc.stdout)
    if not res or not isinstance(res.get("box"), dict) or not _valid_box(res["box"]):
        return None
    if float(res.get("confidence", 0.0)) < min_confidence:
        return None
    return res


if __name__ == "__main__":  # tiny manual smoke test: python vision.py <image>
    import sys
    print(json.dumps(box_from_claude(sys.argv[1]), indent=2))
