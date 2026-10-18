import numpy as np
from PIL import Image

from unbake.metrics import score
from unbake.reconstruct import load_frames, reconstruct


def test_easy_sample_is_reconstructed(corpus):
    gif, truth = corpus["easy"]
    rec = reconstruct(str(gif))
    s = score(rec, str(gif.with_suffix(".json")))
    assert s["n_est"] == s["n_gt"]
    assert s["identity"] > 0.95 and s["pos_median"] < 0.5 and s["iou"] > 0.9


def test_medium_sample_handles_occlusions(corpus):
    gif, _ = corpus["medium"]
    s = score(reconstruct(str(gif)), str(gif.with_suffix(".json")))
    assert s["identity"] > 0.85 and s["iou"] > 0.85


def test_strokes_are_detected(corpus):
    gif, truth = corpus["strokes"]
    rec = reconstruct(str(gif))
    expected = sum(1 for g in truth["shapes"] if g["kind"] == "stroke")
    assert sum(1 for s in rec["shapes"] if s.kind == "stroke") == expected


def test_large_gif_gives_the_same_objects(corpus, upscaled):
    gif, _ = corpus["easy"]
    small = reconstruct(str(gif))
    big = reconstruct(str(upscaled(gif, 3)))
    assert len(big["shapes"]) == len(small["shapes"])
    assert big["analysis_scale"] < 1
    # Geometry comes back in the large GIF's pixels.
    a = np.nanmean([s.track[:, 0] for s in small["shapes"]])
    b = np.nanmean([s.track[:, 0] for s in big["shapes"]])
    assert abs(b - 3 * a) < 3


def test_gif_frame_durations_are_respected(tmp_path):
    frames = [Image.new("RGB", (8, 8), c) for c in [(255, 0, 0), (0, 255, 0)]]
    path = tmp_path / "d.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=[40, 120], loop=0)
    assert len(load_frames(str(path))) == 4  # 1 + 3 frames at the shortest duration
