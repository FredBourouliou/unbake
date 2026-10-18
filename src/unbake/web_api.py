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

from .easing import sample_channel
from .export import eased_channels, to_lottie, to_svg
from .reconstruct import reconstruct
from .scene import Shape, holes


def _keys_json(keys, digits=4):
    return [[f, round(v, digits), [round(x, 4) for x in c]] for f, v, c in keys]


def reconstruct_gif(data: bytes) -> dict:
    """Scene for the editor: every channel as eased keyframes [frame, value, easing to next]."""
    with tempfile.NamedTemporaryFile(suffix=".gif") as f:
        f.write(data)
        f.flush()
        rec = reconstruct(f.name)
    n, h, w = rec["frames"].shape[:3]
    objects = []
    for i, s in enumerate(rec["shapes"]):
        visible, ch = eased_channels(s)
        base = {"id": i, "color": "#%02x%02x%02x" % tuple(s.color), "span": [int(visible[0]), int(visible[-1])]}
        if s.kind == "stroke":
            line = s.outer[:: max(1, len(s.outer) // 200)]
            base.update({"kind": "stroke", "line": np.round(line, 2).tolist(), "width": round(float(s.width), 2),
                         "closed": bool(s.closed), "cap": s.cap})
            base.update({name: _keys_json(keys) for name, keys in ch.items()})
        else:
            base.update({"outer": np.round(s.outer, 2).tolist(), "holes": [np.round(hp, 2).tolist() for hp in holes(s)],
                         "x": _keys_json(ch["x"], 2), "y": _keys_json(ch["y"], 2),
                         "rot": _keys_json(ch["rot"], 2), "scale": _keys_json(ch["scale"], 4)})
        objects.append(base)
    return {"width": w, "height": h, "frames": n, "fps": 24,
            "background": "#%02x%02x%02x" % rec["background"], "objects": objects}


def _hex(c: str) -> tuple[int, int, int]:
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def _sample(keys: list, n: int, span: tuple[int, int]) -> np.ndarray:
    """Per-frame values of an eased channel; NaN outside the object's span."""
    out = np.full(n, np.nan)
    f = np.arange(span[0], span[1] + 1)
    frames = np.array([k[0] for k in keys], float)
    values = np.array([k[1] for k in keys], float)
    out[f] = sample_channel(frames, values, [(i, tuple(k[2])) for i, k in enumerate(keys)], f.astype(float))
    return out


def scene_to_shapes(scene: dict) -> list[Shape]:
    n = scene["frames"]
    shapes = []
    for o in scene["objects"]:
        if o.get("hidden"):
            continue
        span = tuple(o["span"])
        if o.get("kind") == "stroke":
            start = _sample(o["start"], n, span)
            if o["closed"]:
                length = _sample(o["len"], n, span)
                trim = np.c_[start % 1.0, (start + length) % 1.0]
                trim[length >= 0.999] = (0.0, 1.0)
            else:
                trim = np.c_[np.clip(start, 0, 1), np.clip(_sample(o["end"], n, span), 0, 1)]
            track = np.full((n, 4), np.nan)
            track[~np.isnan(start)] = [0.0, 0.0, 0.0, 1.0]
            shapes.append(Shape("stroke", _hex(o["color"]), np.array(o["line"], float), None, track,
                                width=float(o["width"]), closed=bool(o["closed"]), cap=o["cap"], trim=trim))
            continue
        track = np.c_[_sample(o["x"], n, span), _sample(o["y"], n, span), _sample(o["rot"], n, span), _sample(o["scale"], n, span)]
        hole_list = [np.array(hp, float) for hp in o.get("holes", [])]
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
