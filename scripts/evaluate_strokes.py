"""v0.2 evaluation: strokes level + false strokes on the medium level.

Usage: uv run python scripts/evaluate_strokes.py data/dev
"""
import json
import sys
from pathlib import Path

import numpy as np

from unbake.reconstruct import label_frames, reconstruct
from unbake.scene import Shape, render, render_frame, transform

split = Path(sys.argv[1])


def gt_shape(g):
    if g["kind"] != "stroke":
        return None
    return Shape("stroke", tuple(g["color"]), np.array(g["outer"]), None, np.array(g["track"]),
                 width=g["width"], closed=g["closed"], cap=g["cap"], trim=np.array(g["trim"]))


def mask(shape, t, size):
    """Coverage of one shape on a frame (drawn in white, so dark colours count too)."""
    white = Shape(shape.kind, (255, 255, 255), shape.outer, shape.hole, shape.track,
                  width=shape.width, closed=shape.closed, cap=shape.cap, trim=shape.trim)
    return render_frame([white], t, size, (0, 0, 0)).max(axis=2) > 127


def region_iou(rec, size):
    T = len(rec["frames"])
    out = label_frames(np.stack(render(rec["shapes"], T, size, rec["background"])),
                       np.array(rec["background"], float), rec["colours"])
    vals = [np.count_nonzero((rec["labels"][t] == k) & (out[t] == k)) / u
            for t in range(T) for k in range(1, len(rec["colours"]) + 1)
            if (u := np.count_nonzero((rec["labels"][t] == k) | (out[t] == k))) > 20]
    return float(np.mean(vals)) if vals else 0.0


found = total = 0
stroke_ious, region = [], []
for gif in sorted(split.glob("strokes-*.gif")):
    truth = json.loads(gif.with_suffix(".json").read_text())
    size = tuple(truth["size"])
    rec = reconstruct(str(gif))
    region.append(region_iou(rec, size))
    rec_strokes = [s for s in rec["shapes"] if s.kind == "stroke"]
    for g in truth["shapes"]:
        gs = gt_shape(g)
        if gs is None:
            continue
        total += 1
        line = transform(gs.outer, *gs.track[0])
        best, best_d = None, 1e9
        for s in rec_strokes:
            if np.abs(np.array(s.color) - np.array(gs.color)).sum() > 60:
                continue
            d = np.mean([np.min(np.linalg.norm(s.outer - p, axis=1)) for p in line[::4]])
            if d < best_d:
                best, best_d = s, d
        if best is None or best_d > 6:
            stroke_ious.append(0.0)
            continue
        found += 1
        vals = []
        for t in range(truth["n_frames"]):
            a, b = mask(gs, t, size), mask(best, t, size)
            u = np.count_nonzero(a | b)
            if u > 10:
                vals.append(np.count_nonzero(a & b) / u)
        stroke_ious.append(float(np.mean(vals)) if vals else 0.0)

false_strokes = 0
medium = sorted(split.glob("medium-*.gif"))
for gif in medium:
    rec = reconstruct(str(gif))
    false_strokes += any(s.kind == "stroke" for s in rec["shapes"])

summary = {"strokes_recovered": found / total, "stroke_iou": float(np.mean(stroke_ious)),
           "region_iou": float(np.mean(region)), "n_strokes": total,
           "medium_with_false_stroke": f"{false_strokes}/{len(medium)}"}
print(json.dumps(summary, indent=2))
Path("results").joinpath(f"strokes-{split.name}.json").write_text(json.dumps(summary, indent=2))
