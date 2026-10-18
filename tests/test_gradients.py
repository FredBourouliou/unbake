import json

import numpy as np

from unbake.export import to_lottie, to_svg
from unbake.reconstruct import reconstruct
from unbake.web_api import export_scene, reconstruct_gif


def test_gradient_shapes_come_back_as_gradients(corpus):
    gif, truth = corpus["gradients"]
    rec = reconstruct(str(gif))
    n_true = sum("gradient" in g for g in truth["shapes"])
    found = [s for s in rec["shapes"] if s.gradient is not None]
    assert len(found) == n_true
    for s in found:
        p0, p1, c0, c1 = s.gradient
        assert np.linalg.norm(np.asarray(p1) - np.asarray(p0)) > 10   # spans the shape
        assert sum(abs(a - b) for a, b in zip(c0, c1)) > 60            # two distinct colours


def test_flat_samples_get_no_gradient(corpus):
    for level in ("easy", "medium", "strokes"):
        gif, _ = corpus[level]
        assert all(s.gradient is None for s in reconstruct(str(gif))["shapes"])


def test_gradient_export_and_editor_round_trip(corpus):
    gif, _ = corpus["gradients"]
    rec = reconstruct(str(gif))
    n = len(rec["frames"])
    lottie = to_lottie(rec["shapes"], (256, 256), 24, n, rec["background"])
    fills = [it["ty"] for layer in lottie["layers"] if layer["ty"] == 4 for it in layer["shapes"][0]["it"]]
    assert "gf" in fills
    assert "<linearGradient" in to_svg(rec["shapes"], (256, 256), 24, n, rec["background"])

    scene = reconstruct_gif(gif.read_bytes())
    obj = next(o for o in scene["objects"] if "gradient" in o)
    obj["gradient"]["colors"][1] = "#123456"
    out = export_scene(json.loads(json.dumps(scene)))
    assert 'stop-color="rgb(18, 52, 86)"' in out["svg"]          # the edited end colour is exported
