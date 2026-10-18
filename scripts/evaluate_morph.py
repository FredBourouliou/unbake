"""Morph evaluation: morph level + false morphs on the medium level.

Usage: uv run python scripts/evaluate_morph.py data/dev
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
from unbake.scene import Shape, render, render_frame


def gt_shapes(truth):
    return [Shape(g["kind"], tuple(g["color"]), np.array(g["outer"]), None if g["hole"] is None else np.array(g["hole"]),
                  np.array(g["track"]), morph=None if "morph" not in g else np.array(g["morph"])) for g in truth["shapes"]]


def mask(s, t, size):
    white = Shape(s.kind, (255, 255, 255), s.outer, s.hole, s.track, morph=s.morph)
    return render_frame([white], t, size, (0, 0, 0))[..., 0] > 127


def run_morph(gif):
    truth = json.loads(gif.with_suffix(".json").read_text())
    rec = reconstruct(str(gif))
    s = score(rec, str(gif.with_suffix(".json")))
    size, T = tuple(truth["size"]), truth["n_frames"]
    gts, est = gt_shapes(truth), rec["shapes"]
    pairs = {}
    if est:
        cost = np.nanmean(np.linalg.norm(np.array([g.track[:, :2] for g in gts])[:, None]
                                         - np.array([e.track[:, :2] for e in est])[None], axis=-1), axis=-1)
        cost = np.nan_to_num(cost, nan=1e9)
        pairs = {g: e for g, e in zip(*linear_sum_assignment(cost)) if cost[g, e] < 25}
    found = total = 0
    ious = []
    for g, gt in enumerate(gts):
        if gt.morph is None:
            continue
        total += 1
        e = pairs.get(g)
        found += e is not None and est[e].morph is not None
        vals = []
        for t in range(T):
            a = mask(gt, t, size)
            b = mask(est[e], t, size) if e is not None and not np.isnan(est[e].track[t, 0]) else np.zeros_like(a)
            u = np.count_nonzero(a | b)
            if u > 20:
                vals.append(np.count_nonzero(a & b) / u)
        ious.append(float(np.mean(vals)) if vals else 0.0)
    # Export fidelity: the Lottie file rendered by lottie-web vs the reconstruction's own render.
    own = np.stack(render(est, T, size, rec["background"])).astype(float)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.json"
        path.write_text(json.dumps(to_lottie(est, size, truth["fps"], T, rec["background"])))
        lot = render_lottie(str(path), size, T, rec["background"]).astype(float)
    fg = np.abs(own - np.array(rec["background"], float)).sum(-1) > 30
    return {"identity": s["identity"], "found": found, "total": total, "ious": ious,
            "lottie_error": float(np.abs(lot - own)[fg].mean())}


def run_medium(gif):
    return any(s.morph is not None for s in reconstruct(str(gif))["shapes"])


if __name__ == "__main__":
    split = Path(sys.argv[1])
    with Pool(16) as pool:
        rows = pool.map(run_morph, sorted(split.glob("morph-*.gif")))
        medium = pool.map(run_medium, sorted(split.glob("medium-*.gif")))
    summary = {
        "samples": len(rows),
        "morphs_recovered": sum(r["found"] for r in rows) / max(1, sum(r["total"] for r in rows)),
        "n_morphs": sum(r["total"] for r in rows),
        "morph_iou": float(np.mean([v for r in rows for v in r["ious"]])),
        "identity": float(np.mean([r["identity"] for r in rows])),
        "lottie_error": float(np.mean([r["lottie_error"] for r in rows])),
        "medium_with_false_morph": f"{sum(medium)}/{len(medium)}",
    }
    print(json.dumps(summary, indent=2))
    Path("results").joinpath(f"morph-{split.name}.json").write_text(json.dumps(summary, indent=2))
