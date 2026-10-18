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


def make_sample(level: str, seed: int) -> dict:
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
            color = PALETTE[colors[i % len(PALETTE)]]
            if sum(abs(a - b) for a, b in zip(color, background)) < 60:
                color = PALETTE[colors[(i + 3) % len(PALETTE)]]
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
                    "track": s.track.round(3).tolist()} for s in sample["shapes"]],
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
    counts = {"easy": 20, "medium": 20, "hard": 10}
    build(root, counts, holdout=False)
    build(root, counts, holdout=True)
    print(f"corpus written to {root}")
