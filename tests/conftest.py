import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from unbake.corpus import make_sample, save_sample


@pytest.fixture(scope="session")
def corpus(tmp_path_factory):
    """A few small corpus samples written to disk once per test session."""
    root = tmp_path_factory.mktemp("corpus")
    paths = {}
    for level, seed in [("easy", 1), ("medium", 2), ("strokes", 3), ("gradients", 4)]:
        gif = save_sample(make_sample(level, seed), root)
        paths[level] = (gif, json.loads(gif.with_suffix(".json").read_text()))
    return paths


@pytest.fixture
def upscaled(tmp_path):
    """Save a copy of a GIF enlarged by an integer factor."""
    def _make(gif: Path, factor: int) -> Path:
        from PIL import ImageSequence
        frames = [f.convert("RGB") for f in ImageSequence.Iterator(Image.open(gif))]
        big = [f.resize((f.width * factor, f.height * factor), Image.LANCZOS).quantize(colors=64) for f in frames]
        out = tmp_path / f"x{factor}.gif"
        big[0].save(out, save_all=True, append_images=big[1:], duration=42, loop=0)
        return out
    return _make
