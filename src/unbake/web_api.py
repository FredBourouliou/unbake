"""JSON bridge used by the browser editor (runs inside Pyodide).

reconstruct_gif(bytes) -> scene dict with editable keyframes
export_scene(scene)    -> {"lottie": str, "svg": str}

The editing model is the keyframe model: each object has position, rotation and scale
keyframes; frames in between are linear interpolations, exactly what the exports play.
"""
from __future__ import annotations

import json
import math
import tempfile

import numpy as np

from .export import keyframes, to_lottie, to_svg, trim_keys
from .reconstruct import reconstruct
from .scene import Shape, holes


def reconstruct_gif(data: bytes) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".gif") as f:
        f.write(data)
        f.flush()
        rec = reconstruct(f.name)
    n, h, w = rec["frames"].shape[:3]
    objects = []
    for i, s in enumerate(rec["shapes"]):
        if s.kind == "stroke":
            visible, start, length, ks, kl = trim_keys(s)
            at = {int(f): j for j, f in enumerate(visible)}
            line = s.outer[:: max(1, len(s.outer) // 200)]
            objects.append({
                "id": i, "kind": "stroke",
                "color": "#%02x%02x%02x" % tuple(s.color),
                "line": np.round(line, 2).tolist(),
                "width": round(float(s.width), 2), "closed": bool(s.closed), "cap": s.cap,
                "span": [int(visible[0]), int(visible[-1])],
                "start": [[f, round(float(start[at[f]]), 4)] for f in ks],
                "len": [[f, round(float(length[at[f]]), 4)] for f in kl],
            })
            continue
        visible, keys = keyframes(s)
        tr = s.track
        rot = np.full(n, np.nan)
        rot[visible] = np.unwrap(np.deg2rad(tr[visible, 2])) * 180 / math.pi
        pos = sorted(set(keys["x"]) | set(keys["y"]))
        objects.append({
            "id": i,
            "color": "#%02x%02x%02x" % tuple(s.color),
            "outer": np.round(s.outer, 2).tolist(),
            "holes": [np.round(h, 2).tolist() for h in holes(s)],
            "span": [int(visible[0]), int(visible[-1])],
            "pos": [[f, round(float(tr[f, 0]), 2), round(float(tr[f, 1]), 2)] for f in pos],
            "rot": [[f, round(float(rot[f]), 2)] for f in keys["rot"]],
            "scale": [[f, round(float(tr[f, 3]), 4)] for f in keys["scale"]],
        })
    return {"width": w, "height": h, "frames": n, "fps": 24,
            "background": "#%02x%02x%02x" % rec["background"], "objects": objects}


def _hex(c: str) -> tuple[int, int, int]:
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def _interp(keys: list, n: int, span: tuple[int, int], cols: int) -> np.ndarray:
    k = np.array(keys, float)
    out = np.full((n, cols), np.nan)
    f = np.arange(span[0], span[1] + 1)
    for c in range(cols):
        out[f, c] = np.interp(f, k[:, 0], k[:, 1 + c])
    return out


def scene_to_shapes(scene: dict) -> list[Shape]:
    n = scene["frames"]
    shapes = []
    for o in scene["objects"]:
        if o.get("hidden"):
            continue
        span = tuple(o["span"])
        if o.get("kind") == "stroke":
            start = _interp(o["start"], n, span, 1)[:, 0]
            length = _interp(o["len"], n, span, 1)[:, 0]
            if o["closed"]:
                trim = np.c_[start % 1.0, (start + length) % 1.0]
                full = length >= 0.999
                trim[full] = (0.0, 1.0)
            else:
                trim = np.c_[start, np.minimum(start + length, 1.0)]
            track = np.full((n, 4), np.nan)
            track[~np.isnan(start)] = [0.0, 0.0, 0.0, 1.0]
            shapes.append(Shape("stroke", _hex(o["color"]), np.array(o["line"], float), None, track,
                                width=float(o["width"]), closed=bool(o["closed"]), cap=o["cap"], trim=trim))
            continue
        pos = _interp(o["pos"], n, span, 2)
        rot = _interp(o["rot"], n, span, 1)
        scale = _interp(o["scale"], n, span, 1)
        track = np.c_[pos, rot, scale]
        hole_list = [np.array(h, float) for h in o.get("holes", [])]
        shapes.append(Shape("polygon", _hex(o["color"]), np.array(o["outer"], float), hole_list or None, track))
    return shapes


def export_scene(scene: dict) -> dict:
    shapes = scene_to_shapes(scene)
    size = (scene["width"], scene["height"])
    bg = _hex(scene["background"])
    n, fps = scene["frames"], scene["fps"]
    return {"lottie": json.dumps(to_lottie(shapes, size, fps, n, bg)), "svg": to_svg(shapes, size, fps, n, bg)}


def reconstruct_for_browser(data: bytes) -> tuple[str, bytes]:
    """Scene as JSON plus the decoded input frames (RGB bytes, frames × height × width × 3)."""
    import time
    from .reconstruct import load_frames
    t0 = time.perf_counter()
    scene = reconstruct_gif(data)
    scene["seconds"] = round(time.perf_counter() - t0, 2)
    with tempfile.NamedTemporaryFile(suffix=".gif") as f:
        f.write(data)
        f.flush()
        frames = load_frames(f.name)
    return json.dumps(scene), frames.astype(np.uint8).tobytes()
