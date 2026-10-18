"""Compare Unbake and OmniLottie on the same holdout GIFs.

Both outputs are Lottie JSON rendered by lottie-web at the input's size and frame count,
then scored with the same per-colour-region IoU against the input frames.

Usage: uv run python scripts/compare_omnilottie.py OMNILOTTIE_DIR [n=10]
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from unbake.export import save
from unbake.lottie_render import region_iou, render_lottie
from unbake.reconstruct import reconstruct

omni = Path(sys.argv[1]).resolve()
n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
out = Path("results/compare")
out.mkdir(parents=True, exist_ok=True)
rows = []
for gif in sorted(Path("data/holdout").glob("medium-*.gif"))[:n]:
    t0 = time.time()
    rec = reconstruct(str(gif))
    h, w = rec["frames"].shape[1:3]
    T = len(rec["frames"])
    u_json, _ = save(rec, (w, h), 24, str(out / f"{gif.stem}.unbake"))
    t_unbake = time.time() - t0
    iou_u = region_iou(rec["labels"], render_lottie(u_json, (w, h), T), rec["background"], rec["colours"])

    o_json = out / f"{gif.stem}.omnilottie.json"
    t0 = time.time()
    proc = subprocess.run([str(omni / ".venv/bin/python"), "inference_hf.py", "--model_path", "./model",
                           "--video", str(gif.resolve()), "--output", str(o_json.resolve()), "--device", "mps"],
                          cwd=omni, capture_output=True, text=True)
    t_omni = time.time() - t0
    iou_o, err = 0.0, ""
    if proc.returncode == 0 and o_json.exists():
        try:
            iou_o = region_iou(rec["labels"], render_lottie(str(o_json), (w, h), T), rec["background"], rec["colours"])
        except Exception as e:  # invalid Lottie counts as a failure
            err = f"render failed: {e}"
    else:
        err = (proc.stderr or proc.stdout)[-300:]
    rows.append({"gif": gif.stem, "unbake_iou": iou_u, "unbake_s": t_unbake,
                 "omnilottie_iou": iou_o, "omnilottie_s": t_omni, "error": err})
    print(f"{gif.stem}: unbake IoU {iou_u:.3f} ({t_unbake:.1f}s) | omnilottie IoU {iou_o:.3f} ({t_omni:.0f}s) {err[:80]}", flush=True)

summary = {k: float(np.mean([r[k] for r in rows])) for k in ("unbake_iou", "unbake_s", "omnilottie_iou", "omnilottie_s")}
summary["omnilottie_failures"] = sum(1 for r in rows if r["error"])
(Path("results") / "compare-omnilottie.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
print(json.dumps(summary, indent=2))
