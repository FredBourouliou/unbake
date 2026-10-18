"""Run the reconstruction on a corpus split and print per-level metrics.

Usage: uv run python scripts/evaluate.py data/dev [easy medium hard]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from unbake.metrics import score
from unbake.reconstruct import reconstruct

split = Path(sys.argv[1])
levels = sys.argv[2:] or ["easy", "medium", "hard"]
summary = {}
for level in levels:
    rows = []
    for gif in sorted(split.glob(f"{level}-*.gif")):
        t0 = time.time()
        rec = reconstruct(str(gif))
        s = score(rec, str(gif.with_suffix(".json")))
        s["seconds"] = time.time() - t0
        s.pop("rerender")
        s["name"] = gif.stem
        rows.append(s)
    agg = {
        "samples": len(rows),
        "identity": float(np.mean([r["identity"] for r in rows])),
        "pos_median": float(np.median([r["pos_median"] for r in rows])),
        "rot_median": float(np.median([r["rot_median"] for r in rows])),
        "scale_median": float(np.median([r["scale_median"] for r in rows])),
        "iou": float(np.mean([r["iou"] for r in rows])),
        "objects_found/true": f'{sum(r["n_est"] for r in rows)}/{sum(r["n_gt"] for r in rows)}',
        "seconds_per_gif": float(np.mean([r["seconds"] for r in rows])),
    }
    summary[level] = agg
    worst = sorted(rows, key=lambda r: r["identity"])[:3]
    print(f"== {level}: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in agg.items()))
    print("   worst:", ", ".join(f'{r["name"]} id={r["identity"]:.2f} iou={r["iou"]:.2f} est/gt={r["n_est"]}/{r["n_gt"]}' for r in worst))
out = Path("results") / f"metrics-{split.name}.json"
out.write_text(json.dumps(summary, indent=2))
