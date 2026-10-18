"""Render Lottie JSON to frames through lottie-web (tools/lottie-render), for scoring."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

TOOL = Path(__file__).resolve().parents[2] / "tools" / "lottie-render" / "render.js"


def render_lottie(json_path: str, size: tuple[int, int], n_frames: int, background=(255, 255, 255)) -> np.ndarray:
    """Frames as RGB; transparent areas are composited over `background`."""
    with tempfile.TemporaryDirectory() as out:
        subprocess.run(["node", str(TOOL), json_path, out, str(size[0]), str(size[1]), str(n_frames)],
                       check=True, capture_output=True, text=True, timeout=180)
        files = sorted(Path(out).glob("*.png"))
        out = []
        for f in files:
            rgba = Image.open(f).convert("RGBA")
            base = Image.new("RGBA", rgba.size, tuple(int(c) for c in background) + (255,))
            out.append(np.asarray(Image.alpha_composite(base, rgba).convert("RGB")))
        return np.stack(out)


def region_iou(input_labels: np.ndarray, frames: np.ndarray, background, colours) -> float:
    """Mean IoU per colour region between the input labels and rendered frames."""
    from .reconstruct import label_frames
    out = label_frames(frames, np.asarray(background, float), colours)
    ious = []
    for t in range(len(out)):
        for k in range(1, len(colours) + 1):
            a, b = input_labels[t] == k, out[t] == k
            union = np.count_nonzero(a | b)
            if union > 20:
                ious.append(np.count_nonzero(a & b) / union)
    return float(np.mean(ious)) if ious else 0.0
