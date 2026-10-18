"""Export a reconstruction as Lottie JSON and as an animated SVG.

Per-frame transforms are reduced to keyframes with Ramer–Douglas–Peucker on each
channel (x, y, rotation, scale); playback interpolates linearly between keyframes.
"""
from __future__ import annotations

import json
import math

import numpy as np

from .scene import Shape

# Maximum deviation allowed when dropping a frame, per channel.
TOLERANCE = {"x": 0.35, "y": 0.35, "rot": 1.5, "scale": 0.015}


def _rdp(t: np.ndarray, v: np.ndarray, tol: float) -> list[int]:
    """Indices of the samples kept by Ramer–Douglas–Peucker on a 1-D signal."""
    keep = {0, len(v) - 1}
    stack = [(0, len(v) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        seg = v[a] + (v[b] - v[a]) * (t[a + 1:b] - t[a]) / (t[b] - t[a])
        err = np.abs(v[a + 1:b] - seg)
        i = int(err.argmax())
        if err[i] > tol:
            m = a + 1 + i
            keep.add(m)
            stack += [(a, m), (m, b)]
    return sorted(keep)


def keyframes(shape: Shape) -> tuple[np.ndarray, list[dict[str, list[int]]]]:
    """Return the visible frame span and the kept keyframe indices per channel."""
    visible = np.flatnonzero(~np.isnan(shape.track[:, 0]))
    t = visible.astype(float)
    keys = {}
    for i, name in enumerate(("x", "y", "rot", "scale")):
        v = shape.track[visible, i]
        if name == "rot":
            v = np.unwrap(np.deg2rad(v)) * 180 / math.pi
        keys[name] = [int(visible[k]) for k in _rdp(t, v, TOLERANCE[name])]
    return visible, keys


def keyframe_count(shapes: list[Shape]) -> tuple[int, int]:
    """(keyframes kept, frames × channels before simplification)."""
    kept = total = 0
    for s in shapes:
        visible, keys = keyframes(s)
        kept += sum(len(k) for k in keys.values())
        total += 4 * len(visible)
    return kept, total


# Lottie ---------------------------------------------------------------------------------

def _path(points: np.ndarray) -> dict:
    pts = [[round(float(x), 3), round(float(y), 3)] for x, y in points]
    zeros = [[0, 0]] * len(pts)
    return {"ty": "sh", "ks": {"a": 0, "k": {"c": True, "v": pts, "i": zeros, "o": zeros}}}


def _anim(frames: list[int], values: list) -> dict:
    if len(frames) == 1:
        return {"a": 0, "k": values[0]}
    k = [{"t": f, "s": v if isinstance(v, list) else [v],
          "i": {"x": [1], "y": [1]}, "o": {"x": [0], "y": [0]}} for f, v in zip(frames, values)]
    return {"a": 1, "k": k}


def to_lottie(shapes: list[Shape], size: tuple[int, int], fps: int, n_frames: int,
              background: tuple[int, int, int]) -> dict:
    w, h = size
    layers = []
    for idx, s in enumerate(reversed(shapes)):  # Lottie draws the first layer on top
        visible, keys = keyframes(s)
        tr = s.track
        rot = np.full(len(tr), np.nan)
        rot[visible] = np.unwrap(np.deg2rad(tr[visible, 2])) * 180 / math.pi
        pos_frames = sorted(set(keys["x"]) | set(keys["y"]))
        items = [_path(s.outer)]
        if s.hole is not None:
            items.append(_path(s.hole))
        items.append({"ty": "fl", "c": {"a": 0, "k": [c / 255 for c in s.color] + [1]},
                      "o": {"a": 0, "k": 100}, "r": 2})  # even-odd: the hole stays empty
        items.append({"ty": "tr", "p": {"a": 0, "k": [0, 0]}, "a": {"a": 0, "k": [0, 0]},
                      "s": {"a": 0, "k": [100, 100]}, "r": {"a": 0, "k": 0}, "o": {"a": 0, "k": 100}})
        layers.append({
            "ddd": 0, "ind": idx + 1, "ty": 4, "nm": f"object {len(shapes) - idx}",
            "ip": int(visible[0]), "op": int(visible[-1]) + 1, "st": 0, "sr": 1, "bm": 0,
            "ks": {
                "o": {"a": 0, "k": 100},
                "a": {"a": 0, "k": [0, 0, 0]},
                "p": _anim(pos_frames, [[round(float(tr[f, 0]), 3), round(float(tr[f, 1]), 3), 0] for f in pos_frames]),
                "r": _anim(keys["rot"], [round(float(rot[f]), 3) for f in keys["rot"]]),
                "s": _anim(keys["scale"], [[round(float(tr[f, 3]) * 100, 3)] * 2 + [100] for f in keys["scale"]]),
            },
            "shapes": [{"ty": "gr", "nm": "outline", "it": items}],
        })
    layers.append({"ddd": 0, "ind": len(shapes) + 1, "ty": 1, "nm": "background",
                   "sc": "#%02x%02x%02x" % background, "sw": w, "sh": h,
                   "ip": 0, "op": n_frames, "st": 0, "sr": 1, "bm": 0,
                   "ks": {"o": {"a": 0, "k": 100}, "a": {"a": 0, "k": [0, 0, 0]},
                          "p": {"a": 0, "k": [0, 0, 0]}, "r": {"a": 0, "k": 0}, "s": {"a": 0, "k": [100, 100, 100]}}})
    return {"v": "5.7.4", "fr": fps, "ip": 0, "op": n_frames, "w": w, "h": h, "nm": "unbake",
            "ddd": 0, "assets": [], "layers": layers}


# SVG -------------------------------------------------------------------------------------

def _d(points: np.ndarray) -> str:
    return "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in points) + " Z"


def to_svg(shapes: list[Shape], size: tuple[int, int], fps: int, n_frames: int,
           background: tuple[int, int, int]) -> str:
    w, h = size
    dur = n_frames / fps
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
           f'<rect width="{w}" height="{h}" fill="rgb{tuple(background)}"/>']
    for s in shapes:
        visible, keys = keyframes(s)
        tr = s.track.copy()
        tr[visible, 2] = np.unwrap(np.deg2rad(tr[visible, 2])) * 180 / math.pi

        def anim(kind, frames, fmt):
            if len(frames) == 1:
                return ""
            kt = ";".join(f"{f / n_frames:.4f}" for f in frames)  # frame f is shown at f / fps
            vals = ";".join(fmt(f) for f in frames)
            # Hold the first/last value outside the visible span.
            return (f'<animateTransform attributeName="transform" type="{kind}" additive="sum" '
                    f'dur="{dur:.3f}s" repeatCount="indefinite" calcMode="linear" keyTimes="{_pad(kt, frames, n_frames)}" '
                    f'values="{_padv(vals, frames, n_frames, fmt)}"/>')

        pos = sorted(set(keys["x"]) | set(keys["y"]))
        first = visible[0]
        static = (f'translate({tr[first, 0]:.2f} {tr[first, 1]:.2f}) rotate({tr[first, 2]:.2f}) scale({tr[first, 3]:.4f})')
        d = _d(s.outer) + ("" if s.hole is None else " " + _d(s.hole))
        animations = (anim("translate", pos, lambda f: f"{tr[f, 0]:.2f} {tr[f, 1]:.2f}")
                      + anim("rotate", keys["rot"], lambda f: f"{tr[f, 2]:.2f}")
                      + anim("scale", keys["scale"], lambda f: f"{tr[f, 3]:.4f}"))
        vis = ""
        if visible[0] > 0 or visible[-1] < n_frames - 1:
            a, b = visible[0] / n_frames, (visible[-1] + 1) / n_frames
            vis = (f'<animate attributeName="visibility" dur="{dur:.3f}s" repeatCount="indefinite" calcMode="discrete" '
                   f'keyTimes="0;{a:.4f};{b:.4f}" values="hidden;visible;hidden"/>')
        transform_attr = "" if animations else f' transform="{static}"'
        out.append(f'<g{transform_attr}>{animations}{vis}'
                   f'<path d="{d}" fill="rgb{tuple(s.color)}" fill-rule="evenodd"/></g>')
    out.append("</svg>")
    return "\n".join(out)


def _pad(kt: str, frames: list[int], n: int) -> str:
    """SMIL needs keyTimes from exactly 0 to exactly 1: hold the end values."""
    parts = kt.split(";")
    if frames[0] > 0:
        parts = ["0"] + parts
    return ";".join(parts + ["1"])


def _padv(vals: str, frames: list[int], n: int, fmt) -> str:
    parts = vals.split(";")
    if frames[0] > 0:
        parts = [fmt(frames[0])] + parts
    return ";".join(parts + [fmt(frames[-1])])


def save(rec: dict, size: tuple[int, int], fps: int, stem: str) -> tuple[str, str]:
    n = len(rec["frames"])
    lottie = to_lottie(rec["shapes"], size, fps, n, rec["background"])
    svg = to_svg(rec["shapes"], size, fps, n, rec["background"])
    with open(stem + ".json", "w") as f:
        json.dump(lottie, f)
    with open(stem + ".svg", "w") as f:
        f.write(svg)
    return stem + ".json", stem + ".svg"
