"""Reconstruct a GIF and export it as Lottie JSON + animated SVG.

Usage: uv run python scripts/unbake_cli.py input.gif [output_stem] [--fps 24]
"""
import argparse
from pathlib import Path

from unbake.export import keyframe_count, save
from unbake.reconstruct import reconstruct

p = argparse.ArgumentParser()
p.add_argument("gif")
p.add_argument("stem", nargs="?")
p.add_argument("--fps", type=int, default=24)
a = p.parse_args()
rec = reconstruct(a.gif)
h, w = rec["frames"].shape[1:3]
stem = a.stem or str(Path(a.gif).with_suffix(""))
json_path, svg_path = save(rec, (w, h), a.fps, stem)
kept, total = keyframe_count(rec["shapes"])
print(f"{len(rec['shapes'])} objects, {kept} keyframes (from {total} frame values) -> {json_path}, {svg_path}")
