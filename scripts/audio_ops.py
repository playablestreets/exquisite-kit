#!/usr/bin/env python3
"""
audio_ops.py — audio side of Stage 2.

For one recording (the child reading 3 chapters) produce 3 trimmed clips: beginning / middle / end,
cut where the narration crosses chapter 1->2->3.

How the cut points are found:
  1. transcribe()        — WhisperX gives word-level timestamps (fallbacks: faster-whisper, whisper).
  2. read_chapters()     — the 3 chapter texts come from a sidecar `chapters.json` (written by the
                           Claude orchestrator) or, if ANTHROPIC_API_KEY is set, read directly from
                           the page image via the API.
  3. align_boundaries()  — fuzzy-match the opening words of chapter 2 and chapter 3 against the
                           transcript word stream -> 2 boundary timestamps.
  4. split_and_trim()    — ffmpeg cuts the 3 spans and trims leading/trailing silence.

If chapters are unavailable the splitter falls back to silence-based, then to even thirds, so the
reviewer always has draggable starting cut points to correct.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, asdict

PARTS = ["beginning", "middle", "end"]


# --------------------------------------------------------------------------------------- helpers
def duration(path: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", path], capture_output=True, text=True).stdout
    try:
        return float(out.strip())
    except ValueError:
        return 0.0


# ------------------------------------------------------------------------------ 1. transcription
def transcribe(audio_path: str, model_size: str = "small") -> list[dict]:
    """Return a flat list of words: [{"word","start","end"}].

    Uses faster-whisper (the same CTranslate2 Whisper backend WhisperX wraps) — it gives word-level
    timestamps directly and has a stable API across versions. Falls back to whisperx, then openai-
    whisper. Returns [] if none are available (the splitter then uses silence/thirds)."""
    try:
        from faster_whisper import WhisperModel  # type: ignore
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segs, _ = model.transcribe(audio_path, word_timestamps=True, vad_filter=True)
        words = []
        for s in segs:
            for w in (s.words or []):
                words.append({"word": w.word.strip(), "start": float(w.start), "end": float(w.end)})
        if words:
            return words
    except Exception as e:  # pragma: no cover
        print(f"  [audio] faster-whisper unavailable ({e.__class__.__name__}); trying whisperx")

    try:
        import whisperx  # type: ignore
        model = whisperx.load_model(model_size, "cpu", compute_type="int8")
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=8)
        a_model, meta = whisperx.load_align_model(language_code=result.get("language", "en"), device="cpu")
        aligned = whisperx.align(result["segments"], a_model, meta, audio, "cpu")
        return [{"word": w["word"].strip(), "start": float(w["start"]), "end": float(w["end"])}
                for seg in aligned.get("segments", []) for w in seg.get("words", [])
                if "start" in w and "end" in w]
    except Exception:
        return []


# ------------------------------------------------------------------------------ 2. chapter texts
def read_chapters(image_path: str | None, sidecar: str) -> list[str] | None:
    """Prefer a sidecar chapters.json (written by the Claude orchestrator). Else use the Anthropic
    API if a key is present. Else None."""
    if os.path.exists(sidecar):
        try:
            data = json.load(open(sidecar))
            ch = data.get("chapters") if isinstance(data, dict) else data
            if ch and len(ch) >= 3:
                return [str(c).strip() for c in ch[:3]]
        except Exception:
            pass
    if image_path and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return _read_chapters_api(image_path)
        except Exception as e:  # pragma: no cover
            print(f"  [audio] chapter API read failed: {e}")
    return None


def _read_chapters_api(image_path: str) -> list[str]:
    """Read the 3 chapter texts from the left page via the Anthropic vision API.
    NOTE: confirm the current vision model id via the `claude-api` skill before relying on this."""
    import base64
    import anthropic  # type: ignore
    media = "image/jpeg" if image_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
    b64 = base64.standard_b64encode(open(image_path, "rb").read()).decode()
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8"),
        max_tokens=400,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media, "data": b64}},
            {"type": "text", "text": "This page has 3 handwritten chapters (CHAPTER 1, 2, 3) on the "
             "left. Return ONLY a JSON array of the 3 chapter texts as strings, in order."},
        ]}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    start, end = text.find("["), text.rfind("]")
    return [str(c).strip() for c in json.loads(text[start:end + 1])][:3]


# --------------------------------------------------------------------------------- 3. alignment
def _norm(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def align_edges(words: list[dict], chapters: list[str] | None, total: float) -> tuple[list[float], str]:
    """Return ([e0, e1, e2, e3], method): the 3 segments are [e0,e1], [e1,e2], [e2,e3].

    With chapter texts we locate where each chapter is spoken in the transcript, which lets us trim
    BOTH the adult preamble before the story (e0 > 0) and the trailing chatter after it (e3 < total),
    not just place the 2 interior cuts. Falls back to silence, then even thirds (full span)."""
    if words and chapters and len(chapters) >= 3:
        try:
            from rapidfuzz import fuzz
        except Exception:
            fuzz = None
        if fuzz is not None:
            stream = [_norm(w["word"]) for w in words]
            starts = [w["start"] for w in words]
            ends = [w["end"] for w in words]

            def match_span(chapter: str, lo: int) -> tuple[int, int, float] | None:
                """Best contiguous window for a chapter's words, searched at/after index lo."""
                probe = _norm(chapter)
                k = max(2, len(probe.split()))
                best, bi = 0.0, None
                hi = max(lo + 1, len(stream) - k + 1)
                for i in range(lo, hi):
                    cand = " ".join(stream[i:i + k])
                    sc = fuzz.ratio(cand, probe)
                    if sc > best:
                        best, bi = sc, i
                if bi is None or best < 55:
                    return None
                return bi, min(len(stream) - 1, bi + k - 1), best

            s1 = match_span(chapters[0], 0)
            s2 = match_span(chapters[1], (s1[1] + 1) if s1 else 0)
            s3 = match_span(chapters[2], (s2[1] + 1) if s2 else (s1[1] + 1 if s1 else 0))
            if s1 and s2 and s3 and s1[0] <= s2[0] <= s3[0]:
                e0 = starts[s1[0]]
                e1 = starts[s2[0]]
                e2 = starts[s3[0]]
                e3 = min(total, ends[s3[1]] + 0.25)
                if e0 < e1 < e2 < e3:
                    return [round(e0, 2), round(e1, 2), round(e2, 2), round(e3, 2)], "chapter-align"

    # fallback A: 2 largest interior silences, keep full span
    if len(words) > 4:
        gaps = sorted(((b["start"] - a["end"], (a["end"] + b["start"]) / 2)
                       for a, b in zip(words, words[1:])), reverse=True)[:2]
        cuts = sorted(t for _g, t in gaps)
        if len(cuts) == 2 and 0 < cuts[0] < cuts[1] < total:
            return [0.0, round(cuts[0], 2), round(cuts[1], 2), round(total, 2)], "silence"

    # fallback B: even thirds
    return [0.0, round(total / 3, 2), round(2 * total / 3, 2), round(total, 2)], "even-thirds"


# ----------------------------------------------------------------------------- 4. split + trim
_TRIM = ("silenceremove=start_periods=1:start_silence=0.15:start_threshold=-45dB,"
         "areverse,"
         "silenceremove=start_periods=1:start_silence=0.15:start_threshold=-45dB,"
         "areverse")


def cut(audio_path: str, start: float, end: float, out_path: str, trim: bool = True) -> None:
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
           "-i", audio_path]
    if trim:
        cmd += ["-af", _TRIM]
    cmd += ["-ar", "44100", "-ac", "1", out_path]
    subprocess.run(cmd, check=True)


def split_and_trim(audio_path: str, edges: list[float], out_dir: str) -> dict:
    """edges = [e0,e1,e2,e3] -> 3 clips. e0/e3 may be inside the recording (preamble/chatter trimmed)."""
    os.makedirs(out_dir, exist_ok=True)
    edges = sorted(float(e) for e in edges)
    written = {}
    for i, part in enumerate(PARTS):
        p = os.path.join(out_dir, f"{part}.wav")
        cut(audio_path, edges[i], edges[i + 1], p)
        written[part] = {"path": p, "src_start": round(edges[i], 2), "src_end": round(edges[i + 1], 2),
                         "duration": round(duration(p), 2)}
    return written


def split_spans(audio_path: str, spans: list[float], out_dir: str) -> dict:
    """spans = [b0,b1, m0,m1, e0,e1] -> 3 clips from INDEPENDENT [start,end] pairs (one per part), so
    silence/chatter BETWEEN sections (b1->m0, m1->e0) is dropped, not just at the very ends. Each clip
    is still silence-trimmed at its own edges. Used by the reviewer's 6-handle editor."""
    os.makedirs(out_dir, exist_ok=True)
    vals = [float(v) for v in spans]
    written = {}
    for i, part in enumerate(PARTS):
        s, e = sorted((vals[2 * i], vals[2 * i + 1]))   # start<=end within each section
        p = os.path.join(out_dir, f"{part}.wav")
        cut(audio_path, s, e, p)
        written[part] = {"path": p, "src_start": round(s, 2), "src_end": round(e, 2),
                         "duration": round(duration(p), 2)}
    return written


# ------------------------------------------------------------------------------------ orchestrate
@dataclass
class AudioState:
    source: str
    total: float
    edges: list          # [e0,e1,e2,e3] — segment boundaries incl. story start/end (preamble trimmed)
    method: str
    chapters: list | None
    transcript: list


def propose(audio_path: str, image_path: str | None, out_dir: str) -> AudioState:
    """Full auto pass for one recording -> writes the 3 clip proposals + audio_state.json."""
    os.makedirs(out_dir, exist_ok=True)
    total = duration(audio_path)
    words = transcribe(audio_path)
    chapters = read_chapters(image_path, os.path.join(out_dir, "chapters.json"))
    edges, method = align_edges(words, chapters, total)
    split_and_trim(audio_path, edges, out_dir)
    state = AudioState(source=os.path.abspath(audio_path), total=round(total, 2),
                       edges=edges, method=method, chapters=chapters, transcript=words)
    with open(os.path.join(out_dir, "audio_state.json"), "w") as fh:
        json.dump(asdict(state), fh, indent=2)
    return state


if __name__ == "__main__":  # python audio_ops.py <audio> [image] [out_dir]
    import sys
    a = sys.argv[1]
    img = sys.argv[2] if len(sys.argv) > 2 else None
    out = sys.argv[3] if len(sys.argv) > 3 else "._audtest"
    st = propose(a, img, out)
    print(json.dumps(asdict(st), indent=2)[:800])
