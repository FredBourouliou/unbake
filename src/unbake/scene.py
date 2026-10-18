"""Scene model and renderer shared by the corpus generator and the reconstruction.

A scene is a list of shapes drawn in z-order on a flat background. Each shape has a
local outline (polygon, possibly with one hole) and per-frame transforms
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
    hole: np.ndarray | None = None
    # One (x, y, rot, scale) row per frame.
    track: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))


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
        outer = transform(shape.outer, x, y, rot, scale) * SUPERSAMPLE
        draw.polygon([tuple(p) for p in outer], fill=255)
        if shape.hole is not None:
            hole = transform(shape.hole, x, y, rot, scale) * SUPERSAMPLE
            draw.polygon([tuple(p) for p in hole], fill=0)
        canvas.paste(Image.new("RGB", canvas.size, shape.color), mask=mask)
    return np.asarray(canvas.resize((w, h), Image.LANCZOS))


def render(shapes: list[Shape], n_frames: int, size: tuple[int, int],
           background: tuple[int, int, int]) -> list[np.ndarray]:
    return [render_frame(shapes, f, size, background) for f in range(n_frames)]
