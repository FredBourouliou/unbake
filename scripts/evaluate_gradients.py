"""Gradient evaluation: gradients level + false gradients on the medium level.

Usage: uv run python scripts/evaluate_gradients.py data/dev
"""
import json
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from unbake.export import to_lottie
from unbake.lottie_render import render_lottie
from unbake.metrics import score
from unbake.reconstruct import reconstruct
from unbake.scene import Shape, render


def gt_shapes(truth):
    out = []
    for g in truth["shapes"]:
        grad = g.get("gradient")
        out.append(Shape(g["kind"], tuple(g["color"]), np.array(g["outer"]),
                         None if g["hole"] is None else np.array(g["hole"]), np.array(g["track"]),
                         gradient=None if grad is None else (np.array(grad[0]), np.array(grad[1]), tuple(grad[2]), tuple(grad[3]))))
    return out


def run_gradient(gif):
    truth = json.loads(gif.with_suffix(".json").read_text())
    rec = reconstruct(str(gif))
    s = score(rec, str(gif.with_suffix(".json")))
    size, T, bg = tuple(truth["size"]), truth["n_frames"], tuple(truth["background"])
    gts = gt_shapes(truth)
    # Colour error on the foreground, undithered ground truth vs reconstruction.
    a = np.stack(render(gts, T, size, bg)).astype(float)
    b = np.stack(render(rec["shapes"], T, size, rec["background"])).astype(float)
    fg = np.abs(a - np.array(bg, float)).sum(-1) > 30
    err = float(np.abs(a - b)[fg].mean())
    # Export fidelity: the Lottie file rendered by lottie-web vs the reconstruction's own render.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.json"
        path.write_text(json.dumps(to_lottie(rec["shapes"], size, truth["fps"], T, rec["background"])))
        lot = render_lottie(str(path), size, T, rec["background"]).astype(float)
    fg_rec = np.abs(b - np.array(rec["background"], float)).sum(-1) > 30
    lottie_err = float(np.abs(lot - b)[fg_rec].mean())
    # Gradient objects recovered: matched by centroid (as in metrics.score) and filled with a gradient.
    est = rec["shapes"]
    found = total = 0
    if est:
        gxy = np.array([g.track[:, :2] for g in gts])
        exy = np.array([e.track[:, :2] for e in est])
        cost = np.nanmean(np.linalg.norm(gxy[:, None] - exy[None], axis=-1), axis=-1)
        cost = np.nan_to_num(cost, nan=1e9)
        pairs = {g: e for g, e in zip(*linear_sum_assignment(cost)) if cost[g, e] < 25}
    else:
        pairs = {}
    for g, gt in enumerate(gts):
        if gt.gradient is None:
            continue
        total += 1
        e = pairs.get(g)
        found += e is not None and est[e].gradient is not None
    return {"identity": s["identity"], "pos": s["pos_median"], "colour_error": err, "lottie_error": lottie_err,
            "found": found, "total": total}


def run_medium(gif):
    rec = reconstruct(str(gif))
    return any(s.gradient is not None for s in rec["shapes"])


if __name__ == "__main__":
    split = Path(sys.argv[1])
    with Pool(16) as pool:
        rows = pool.map(run_gradient, sorted(split.glob("gradients-*.gif")))
        medium = pool.map(run_medium, sorted(split.glob("medium-*.gif")))
    summary = {
        "samples": len(rows),
        "gradients_recovered": sum(r["found"] for r in rows) / max(1, sum(r["total"] for r in rows)),
        "n_gradients": sum(r["total"] for r in rows),
        "identity": float(np.mean([r["identity"] for r in rows])),
        "pos_median": float(np.median([r["pos"] for r in rows])),
        "colour_error": float(np.mean([r["colour_error"] for r in rows])),
        "lottie_error": float(np.mean([r["lottie_error"] for r in rows])),
        "medium_with_false_gradient": f"{sum(medium)}/{len(medium)}",
    }
    print(json.dumps(summary, indent=2))
    Path("results").joinpath(f"gradients-{split.name}.json").write_text(json.dumps(summary, indent=2))
