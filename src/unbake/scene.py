"""Scene model and renderer shared by the corpus generator and the reconstruction.

A scene is a list of shapes drawn in z-order on a flat background. Each shape has a
local outline (polygon, possibly with holes) and per-frame transforms
(x, y, rotation in degrees, uniform scale).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

SUPERSAMPLE = 4

# Rotational symmetry order of each primitive (0 = any angle).
SYMMETRY = {"circle": 0, "ring": 0, "rect": 2, "square": 4, "triangle": 3, "star": 5, "polygon": 1}


def outline(kind: str, size: float, aspect: float = 1.0) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (outer, hole) polygons in local coordinates, centred on the origin."""
    if kind in ("circle", "ring"):
        a = np.linspace(0, 2 * math.pi, 96, endpoint=False)
        outer = np.c_[np.cos(a), np.sin(a)] * size
        hole = outer * 0.55 if kind == "ring" else None
        return outer, hole
    if kind in ("rect", "square"):
        w, h = size, size * (aspect if kind == "rect" else 1.0)
        return np.array([[-w, -h], [w, -h], [w, h], [-w, h]], float), None
    if kind == "triangle":
        a = np.deg2rad([-90, 30, 150])
        return np.c_[np.cos(a), np.sin(a)] * size, None
    if kind == "star":
        a = np.deg2rad(np.arange(10) * 36 - 90)
        r = np.where(np.arange(10) % 2 == 0, size, size * 0.45)
        return np.c_[np.cos(a) * r, np.sin(a) * r], None
    raise ValueError(kind)


def transform(points: np.ndarray, x: float, y: float, rot: float, scale: float) -> np.ndarray:
    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    rotated = points @ np.array([[c, s], [-s, c]])
    return rotated * scale + [x, y]


@dataclass
class Shape:
    kind: str
    color: tuple[int, int, int]
    outer: np.ndarray
    hole: np.ndarray | list | None = None  # one polygon, a list of polygons, or None
    # One (x, y, rot, scale) row per frame.
    track: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    # Strokes only (kind == "stroke"): `outer` is the centre line, drawn with this width.
    width: float = 0.0
    closed: bool = False
    cap: str = "round"                      # "round" or "butt"
    trim: np.ndarray | None = None          # per frame (start, end) as path fractions; start > end wraps a closed path


def holes(shape: "Shape") -> list[np.ndarray]:
    if shape.hole is None:
        return []
    return list(shape.hole) if isinstance(shape.hole, list) else [shape.hole]


def sub_path(points: np.ndarray, closed: bool, start: float, end: float) -> list[np.ndarray]:
    """Pieces of a polyline between two arc-length fractions (a closed path may wrap)."""
    pts = np.vstack([points, points[:1]]) if closed else points
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.r_[0, np.cumsum(seg)]
    total = cum[-1]
    if total <= 0:
        return []

    def cut(a: float, b: float) -> np.ndarray:
        sa, sb = a * total, b * total
        inner = pts[(cum > sa) & (cum < sb)]
        pa = np.array([np.interp(sa, cum, pts[:, 0]), np.interp(sa, cum, pts[:, 1])])
        pb = np.array([np.interp(sb, cum, pts[:, 0]), np.interp(sb, cum, pts[:, 1])])
        return np.vstack([pa, inner, pb])

    start, end = float(np.clip(start, 0, 1)), float(np.clip(end, 0, 1))
    if start <= end:
        return [cut(start, end)] if end - start > 1e-4 else []
    if not closed:
        return []
    return [cut(start, 1.0), cut(0.0, end)]


def _draw_stroke(draw, shape: "Shape", frame: int, ss: int) -> None:
    x, y, rot, scale = shape.track[frame]
    start, end = shape.trim[frame]
    if math.isnan(start):
        return
    line = transform(shape.outer, x, y, rot, scale) * ss
    w = max(1.0, shape.width * scale * ss)
    for piece in sub_path(line, shape.closed, start, end):
        draw.line([tuple(p) for p in piece], fill=255, width=int(round(w)), joint="curve")
        if shape.cap == "round":
            for p in (piece[0], piece[-1]):
                draw.ellipse([p[0] - w / 2, p[1] - w / 2, p[0] + w / 2, p[1] + w / 2], fill=255)


def render_frame(shapes: list[Shape], frame: int, size: tuple[int, int],
                 background: tuple[int, int, int]) -> np.ndarray:
    """Render one antialiased RGB frame (supersampled, then downscaled)."""
    w, h = size
    canvas = Image.new("RGB", (w * SUPERSAMPLE, h * SUPERSAMPLE), background)
    for shape in shapes:
        x, y, rot, scale = shape.track[frame]
        if math.isnan(x):  # not present on this frame
            continue
        mask = Image.new("L", canvas.size, 0)
        draw = ImageDraw.Draw(mask)
        if shape.kind == "stroke":
            _draw_stroke(draw, shape, frame, SUPERSAMPLE)
            canvas.paste(Image.new("RGB", canvas.size, shape.color), mask=mask)
            continue
        outer = transform(shape.outer, x, y, rot, scale) * SUPERSAMPLE
        draw.polygon([tuple(p) for p in outer], fill=255)
        for hole_poly in holes(shape):
            hole = transform(hole_poly, x, y, rot, scale) * SUPERSAMPLE
            draw.polygon([tuple(p) for p in hole], fill=0)
        canvas.paste(Image.new("RGB", canvas.size, shape.color), mask=mask)
    return np.asarray(canvas.resize((w, h), Image.LANCZOS))


def render(shapes: list[Shape], n_frames: int, size: tuple[int, int],
           background: tuple[int, int, int]) -> list[np.ndarray]:
    return [render_frame(shapes, f, size, background) for f in range(n_frames)]
