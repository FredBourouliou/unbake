"""Command line: reconstruct a GIF and export it as Lottie JSON and animated SVG.

    unbake animation.gif                 # writes animation.json and animation.svg
    unbake animation.gif out/anim --fps 30
"""
import argparse
from pathlib import Path

from .export import keyframe_count, save
from .reconstruct import reconstruct


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="unbake", description="Turn a motion-graphics GIF back into editable objects.")
    p.add_argument("gif", help="input GIF")
    p.add_argument("stem", nargs="?", help="output path without extension (default: next to the GIF)")
    p.add_argument("--fps", type=int, default=24, help="frame rate of the exported animation (default: 24)")
    a = p.parse_args(argv)
    rec = reconstruct(a.gif)
    h, w = rec["frames"].shape[1:3]
    stem = a.stem or str(Path(a.gif).with_suffix(""))
    json_path, svg_path = save(rec, (w, h), a.fps, stem)
    kept, total = keyframe_count(rec["shapes"])
    strokes = sum(s.kind == "stroke" for s in rec["shapes"])
    print(f"{len(rec['shapes'])} objects ({strokes} strokes), {kept} keyframes (from {total} frame values)")
    print(f"-> {json_path}\n-> {svg_path}")


if __name__ == "__main__":
    main()
