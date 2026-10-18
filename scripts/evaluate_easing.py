"""v0.3 evaluation: keyframes and Lottie fidelity, eased export vs v0.2 linear export.

Same reconstruction for both; only the export changes. Lottie files are rendered by lottie-web.
Usage: uv run python scripts/evaluate_easing.py data/dev [medium strokes]
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from unbake.export import keyframe_count, to_lottie
from unbake.lottie_render import region_iou, render_lottie
from unbake.reconstruct import reconstruct

split = Path(sys.argv[1])
levels = sys.argv[2:] or ["medium", "strokes"]
keys = {False: 0, True: 0}
ious = {False: [], True: []}
with tempfile.TemporaryDirectory() as tmp:
    for level in levels:
        for gif in sorted(split.glob(f"{level}-*.gif")):
            rec = reconstruct(str(gif))
            h, w = rec["frames"].shape[1:3]
            T = len(rec["frames"])
            for easing in (False, True):
                doc = to_lottie(rec["shapes"], (w, h), 24, T, rec["background"], easing=easing)
                path = Path(tmp) / f"{gif.stem}-{int(easing)}.json"
                path.write_text(json.dumps(doc))
                frames = render_lottie(str(path), (w, h), T, rec["background"])
                ious[easing].append(region_iou(rec["labels"], frames, rec["background"], rec["colours"]))
                keys[easing] += keyframe_count(rec["shapes"], easing=easing)[0]

summary = {"keyframes_linear": keys[False], "keyframes_eased": keys[True],
           "keyframe_ratio": keys[True] / keys[False],
           "lottie_iou_linear": float(np.mean(ious[False])), "lottie_iou_eased": float(np.mean(ious[True])),
           "iou_drop": float(np.mean(ious[False]) - np.mean(ious[True])), "samples": len(ious[True])}
print(json.dumps(summary, indent=2))
Path("results").joinpath(f"easing-{split.name}.json").write_text(json.dumps(summary, indent=2))
