"""Real-world test: community Lottie animations rendered to GIFs, then unbaked.

There is no object-level ground truth here, so the scores are:
- IoU between the input frames and the re-rendered reconstruction (per colour region),
- number of objects and keyframes,
plus a contact sheet per animation (input above, reconstruction below) for visual review.

Usage: uv run python scripts/real_gifs.py data/real/lottie data/real/gif results/real
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from unbake.export import keyframe_count
from unbake.lottie_render import render_lottie
from unbake.metrics import score  # noqa: F401  (kept for parity with evaluate.py)
from unbake.reconstruct import reconstruct, label_frames
from unbake.scene import render

src, gif_dir, out_dir = (Path(p) for p in sys.argv[1:4])
gif_dir.mkdir(parents=True, exist_ok=True)
out_dir.mkdir(parents=True, exist_ok=True)
SIZE, MAX_FRAMES = (256, 256), 72
rows = []
for js in sorted(src.glob("*.json")):
    meta = json.loads(js.read_text())
    n = int(min(MAX_FRAMES, round(meta["op"] - meta["ip"])))
    frames = render_lottie(str(js), SIZE, n, background=(255, 255, 255))
    pal = [Image.fromarray(f).quantize(colors=64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG)
           for f in frames]
    gif = gif_dir / f"{js.stem}.gif"
    pal[0].save(gif, save_all=True, append_images=pal[1:], duration=round(1000 / meta.get("fr", 24)), loop=0)

    t0 = time.time()
    try:
        rec = reconstruct(str(gif))
    except Exception as e:  # a crash is a result too
        rows.append({"name": js.stem, "error": repr(e)[:120]})
        print(f"{js.stem}: CRASH {e!r}"[:160])
        continue
    secs = time.time() - t0
    T = len(rec["frames"])
    rerender = np.stack(render(rec["shapes"], T, SIZE, rec["background"]))
    lab_in = rec["labels"]
    lab_out = label_frames(rerender, np.array(rec["background"], float), rec["colours"])
    ious = [np.count_nonzero((lab_in[t] == k) & (lab_out[t] == k)) / u
            for t in range(T) for k in range(1, len(rec["colours"]) + 1)
            if (u := np.count_nonzero((lab_in[t] == k) | (lab_out[t] == k))) > 20]
    # Share of non-background input pixels whose colour was kept in the palette at all.
    covered = float(np.mean(lab_in != 0))
    kept, total = keyframe_count(rec["shapes"])
    row = {"name": js.stem, "frames": T, "colours": len(rec["colours"]), "objects": len(rec["shapes"]),
           "iou": float(np.mean(ious)) if ious else 0.0, "keyframes": kept, "seconds": round(secs, 2),
           "foreground_share": round(covered, 3)}
    rows.append(row)
    print(f'{row["name"]}: objects {row["objects"]}, colours {row["colours"]}, IoU {row["iou"]:.3f}, {secs:.1f}s')

    idx = np.linspace(0, T - 1, 6).astype(int)
    sheet = Image.new("RGB", (SIZE[0] * 6, SIZE[1] * 2), "white")
    for c, t in enumerate(idx):
        sheet.paste(Image.fromarray(rec["frames"][t]), (c * SIZE[0], 0))
        sheet.paste(Image.fromarray(rerender[t]), (c * SIZE[0], SIZE[1]))
    sheet.save(out_dir / f"{js.stem}.png")

ok = [r for r in rows if "iou" in r]
summary = {"animations": len(rows), "crashes": len(rows) - len(ok),
           "mean_iou": float(np.mean([r["iou"] for r in ok])),
           "iou_ge_0.9": sum(r["iou"] >= 0.9 for r in ok), "iou_0.7_0.9": sum(0.7 <= r["iou"] < 0.9 for r in ok),
           "iou_lt_0.7": sum(r["iou"] < 0.7 for r in ok)}
(out_dir / "summary.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
print(json.dumps(summary, indent=2))
