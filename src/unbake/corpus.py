"""Synthetic motion-graphics corpus with known ground truth.

Each sample is a GIF (what Unbake sees) plus a JSON file with the true shapes and
their per-frame transforms (what the metrics compare against).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .scene import Shape, outline, render

SIZE = (256, 256)
FPS = 24
PALETTE = [(231, 76, 60), (46, 134, 222), (39, 174, 96), (243, 156, 18), (142, 68, 173),
           (26, 188, 156), (52, 73, 94), (236, 240, 241)]
KINDS = ["circle", "ring", "rect", "square", "triangle", "star"]

LEVELS = {
    # name: (min shapes, max shapes, allow overlap, close colours, dither)
    "easy": (1, 3, False, False, False),
    "medium": (3, 5, True, False, True),
    "hard": (4, 6, True, True, True),
}


def _ease(t: np.ndarray, kind: str) -> np.ndarray:
    if kind == "linear":
        return t
    return 0.5 - 0.5 * np.cos(np.pi * t)  # ease-in-out


def _keyframed(rng, n_frames: int, n_keys: int, lo: float, hi: float) -> np.ndarray:
    keys_t = np.sort(np.r_[0, rng.choice(np.arange(1, n_frames - 1), n_keys - 2, replace=False), n_frames - 1])
    keys_v = rng.uniform(lo, hi, n_keys)
    out = np.empty(n_frames)
    ease = rng.choice(["linear", "ease"])
    for a, b, va, vb in zip(keys_t[:-1], keys_t[1:], keys_v[:-1], keys_v[1:]):
        t = _ease(np.linspace(0, 1, b - a + 1), ease)
        out[a:b + 1] = va + (vb - va) * t
    return out


def _stroke_path(rng) -> tuple[np.ndarray, bool]:
    """Centre line of a stroke, centred on the origin: circle/arc loop, wave, or zigzag."""
    kind = rng.choice(["circle", "wave", "zigzag"], p=[0.45, 0.35, 0.2])
    if kind == "circle":
        r = rng.uniform(28, 60)
        a = np.linspace(0, 2 * math.pi, 160, endpoint=False)
        return np.c_[np.cos(a), np.sin(a)] * r, True
    length = rng.uniform(110, 190)
    x = np.linspace(-length / 2, length / 2, 120)
    if kind == "wave":
        y = rng.uniform(12, 30) * np.sin(x / length * 2 * math.pi * rng.uniform(1, 2))
    else:
        y = np.interp(x, np.linspace(x[0], x[-1], 5), rng.uniform(-30, 30, 5))
    return np.c_[x, y], False


def _trim(rng, n: int, closed: bool) -> np.ndarray:
    """Per-frame (start, end): draw on, draw off, draw on then off, or a spinning loader."""
    t = np.linspace(0, 1, n)
    ease = 0.5 - 0.5 * np.cos(np.pi * t)
    styles = ["on", "off", "onoff"] + (["loader"] if closed else [])
    style = rng.choice(styles)
    if style == "on":
        return np.c_[np.zeros(n), np.clip(ease * 1.15, 0, 1)]
    if style == "off":
        return np.c_[np.clip(ease * 1.15, 0, 1) * 0.999, np.ones(n)]
    if style == "onoff":
        a = np.clip(t * 2, 0, 1)
        b = np.clip(t * 2 - 1, 0, 1)
        return np.c_[0.5 - 0.5 * np.cos(np.pi * b), 0.5 - 0.5 * np.cos(np.pi * a)]
    # Loader: a gap that grows and shrinks while the arc spins around the loop.
    spin = (t * rng.uniform(1.0, 2.0)) % 1.0
    length = 0.2 + 0.55 * (0.5 - 0.5 * np.cos(2 * np.pi * t))
    return np.c_[spin, (spin + length) % 1.0]


def make_stroke_sample(seed: int) -> dict:
    """1-3 animated strokes, plus 0-2 rigid shapes so both models coexist."""
    rng = np.random.default_rng(seed)
    n_frames = int(rng.integers(36, 61))
    background = (250, 248, 244) if rng.random() < 0.7 else (24, 26, 32)
    used, shapes = set(), []

    def colour():
        c = next(PALETTE[i] for i in rng.permutation(len(PALETTE))
                 if PALETTE[i] not in used and sum(abs(a - b) for a, b in zip(PALETTE[i], background)) >= 60)
        used.add(c)
        return c

    for _ in range(int(rng.integers(0, 3))):
        kind = str(rng.choice(KINDS))
        outer, hole = outline(kind, float(rng.uniform(14, 24)), float(rng.uniform(0.45, 0.8)))
        n_keys = int(rng.integers(2, 4))
        tr = np.c_[_keyframed(rng, n_frames, n_keys, 40, SIZE[0] - 40), _keyframed(rng, n_frames, n_keys, 40, SIZE[1] - 40),
                   _keyframed(rng, n_frames, n_keys, -180, 180), np.ones(n_frames)]
        shapes.append(Shape(kind, colour(), outer, hole, tr))
    for _ in range(int(rng.integers(1, 4))):
        line, closed = _stroke_path(rng)
        cx, cy = rng.uniform(70, SIZE[0] - 70), rng.uniform(70, SIZE[1] - 70)
        tr = np.tile([cx, cy, float(rng.uniform(0, 360)), 1.0], (n_frames, 1))
        shapes.append(Shape("stroke", colour(), line, None, tr, width=float(rng.uniform(4, 10)), closed=closed,
                            cap=str(rng.choice(["round", "butt"])), trim=_trim(rng, n_frames, closed)))
    frames = render(shapes, n_frames, SIZE, background)
    return {"level": "strokes", "seed": seed, "background": background, "frames": frames,
            "shapes": shapes, "dither": True, "n_frames": n_frames}


def make_gradient_sample(seed: int) -> dict:
    """2-4 overlapping shapes, about half filled with a two-colour linear gradient, dithered."""
    rng = np.random.default_rng(seed)
    n_frames = int(rng.integers(36, 61))
    background = (250, 248, 244) if rng.random() < 0.7 else (24, 26, 32)
    used, shapes = set(), []

    def colour():
        c = next(PALETTE[i] for i in rng.permutation(len(PALETTE))
                 if PALETTE[i] not in used and sum(abs(a - b) for a, b in zip(PALETTE[i], background)) >= 60)
        used.add(c)
        return c

    n_shapes = int(rng.integers(2, 5))
    flags = rng.random(n_shapes) < 0.5
    flags[rng.integers(n_shapes)] = True  # at least one gradient per sample
    # Each gradient takes two palette colours: keep only as many as the palette can serve
    # (without this, seeds with four gradients could not be generated).
    usable = sum(sum(abs(a - b) for a, b in zip(c, background)) >= 60 for c in PALETTE)
    while 2 * flags.sum() + (n_shapes - flags.sum()) > usable:
        flags[np.flatnonzero(flags)[-1]] = False
    for grad in flags:
        kind = str(rng.choice([k for k in KINDS if k != "ring"] if grad else KINDS))
        size = float(rng.uniform(18, 32))
        outer, hole = outline(kind, size, float(rng.uniform(0.5, 0.8)))
        n_keys = int(rng.integers(2, 5))
        tr = np.c_[_keyframed(rng, n_frames, n_keys, 45, SIZE[0] - 45), _keyframed(rng, n_frames, n_keys, 45, SIZE[1] - 45),
                   _keyframed(rng, n_frames, n_keys, -180, 180) if rng.random() < 0.8 else np.zeros(n_frames),
                   _keyframed(rng, n_frames, n_keys, 0.8, 1.2) if rng.random() < 0.5 else np.ones(n_frames)]
        if grad:
            a = rng.uniform(0, 2 * math.pi)
            d = np.array([math.cos(a), math.sin(a)]) * size
            c0, c1 = colour(), colour()
            gradient = (-d, d, c0, c1)
            col = tuple(int(round((p + q) / 2)) for p, q in zip(c0, c1))
            shapes.append(Shape(kind, col, outer, hole, tr, gradient=gradient))
        else:
            shapes.append(Shape(kind, colour(), outer, hole, tr))
    frames = render(shapes, n_frames, SIZE, background)
    return {"level": "gradients", "seed": seed, "background": background, "frames": frames,
            "shapes": shapes, "dither": True, "n_frames": n_frames}


def make_sample(level: str, seed: int) -> dict:
    if level == "strokes":
        return make_stroke_sample(seed)
    if level == "gradients":
        return make_gradient_sample(seed)
    rng = np.random.default_rng(seed)
    lo, hi, overlap, close, dither = LEVELS[level]
    n_shapes = int(rng.integers(lo, hi + 1))
    n_frames = int(rng.integers(36, 61))
    background = (250, 248, 244) if rng.random() < 0.7 else (24, 26, 32)

    colors = rng.permutation(len(PALETTE))
    shapes: list[Shape] = []
    lanes = np.linspace(40, SIZE[1] - 40, n_shapes) if not overlap else None
    for i in range(n_shapes):
        kind = str(rng.choice(KINDS))
        if close and i > 0 and rng.random() < 0.5:
            base = np.array(shapes[-1].color, float)
            color = tuple(int(c) for c in np.clip(base + rng.uniform(-28, 28, 3), 0, 255))
        else:
            # Distinct colours, none too close to the background (same-colour shapes are a "hard" feature).
            used = {s.color for s in shapes}
            color = next(PALETTE[c] for c in colors
                         if PALETTE[c] not in used and sum(abs(a - b) for a, b in zip(PALETTE[c], background)) >= 60)
        size = float(rng.uniform(14, 30))
        outer, hole = outline(kind, size, float(rng.uniform(0.45, 0.8)))
        n_keys = int(rng.integers(2, 5))
        if overlap:
            xs = _keyframed(rng, n_frames, n_keys, 45, SIZE[0] - 45)
            ys = _keyframed(rng, n_frames, n_keys, 45, SIZE[1] - 45)
        else:
            xs = _keyframed(rng, n_frames, n_keys, 45, SIZE[0] - 45)
            ys = lanes[i] + _keyframed(rng, n_frames, n_keys, -6, 6)
        rot = _keyframed(rng, n_frames, n_keys, -180, 180) if rng.random() < 0.8 else np.zeros(n_frames)
        scale = _keyframed(rng, n_frames, n_keys, 0.7, 1.3) if rng.random() < 0.6 else np.ones(n_frames)
        shapes.append(Shape(kind, color, outer, hole, np.c_[xs, ys, rot, scale]))

    frames = render(shapes, n_frames, SIZE, background)
    return {"level": level, "seed": seed, "background": background, "frames": frames,
            "shapes": shapes, "dither": dither, "n_frames": n_frames}


def save_sample(sample: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{sample['level']}-{sample['seed']:04d}"
    images = [Image.fromarray(f) for f in sample["frames"]]
    dither = Image.Dither.FLOYDSTEINBERG if sample["dither"] else Image.Dither.NONE
    pal = [im.quantize(colors=64, method=Image.Quantize.MEDIANCUT, dither=dither) for im in images]
    gif = out_dir / f"{name}.gif"
    pal[0].save(gif, save_all=True, append_images=pal[1:], duration=round(1000 / FPS), loop=0, disposal=1)
    truth = {
        "level": sample["level"], "seed": sample["seed"], "size": SIZE, "fps": FPS,
        "background": sample["background"], "n_frames": sample["n_frames"],
        "shapes": [{"kind": s.kind, "color": s.color, "outer": s.outer.round(3).tolist(),
                    "hole": None if s.hole is None else s.hole.round(3).tolist(),
                    "track": s.track.round(3).tolist(),
                    **({"width": s.width, "closed": s.closed, "cap": s.cap, "trim": s.trim.round(4).tolist()}
                       if s.kind == "stroke" else {}),
                    **({"gradient": [np.round(s.gradient[0], 3).tolist(), np.round(s.gradient[1], 3).tolist(),
                                     list(s.gradient[2]), list(s.gradient[3])]} if s.gradient else {})}
                   for s in sample["shapes"]],
    }
    (out_dir / f"{name}.json").write_text(json.dumps(truth))
    return gif


def build(root: Path, per_level: dict[str, int], holdout: bool) -> list[Path]:
    """Dev seeds start at 0, holdout seeds at 10 000: the two sets never share a sample."""
    base = 10_000 if holdout else 0
    paths = []
    for level, n in per_level.items():
        for k in range(n):
            paths.append(save_sample(make_sample(level, base + k), root / ("holdout" if holdout else "dev")))
    return paths


if __name__ == "__main__":
    import sys
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    counts = {"easy": 20, "medium": 20, "hard": 10, "strokes": 30}
    build(root, counts, holdout=False)
    build(root, counts, holdout=True)
    print(f"corpus written to {root}")
