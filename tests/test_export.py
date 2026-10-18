import json

from unbake.export import to_lottie, to_svg
from unbake.reconstruct import reconstruct
from unbake.web_api import export_scene, reconstruct_gif


def test_lottie_and_svg_for_rigid_and_strokes(corpus):
    for level in ("medium", "strokes"):
        gif, truth = corpus[level]
        rec = reconstruct(str(gif))
        n = len(rec["frames"])
        lottie = to_lottie(rec["shapes"], (256, 256), 24, n, rec["background"])
        assert lottie["op"] == n
        assert len(lottie["layers"]) == len(rec["shapes"]) + 1  # + background
        svg = to_svg(rec["shapes"], (256, 256), 24, n, rec["background"])
        assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
        if level == "strokes":
            kinds = {it["ty"] for layer in lottie["layers"] if layer["ty"] == 4 for it in layer["shapes"][0]["it"]}
            assert {"st", "tm"} <= kinds                      # stroke + trim path
            assert 'pathLength="1"' in svg


def test_editor_round_trip(corpus):
    gif, _ = corpus["strokes"]
    scene = reconstruct_gif(gif.read_bytes())
    assert scene["objects"]
    scene["objects"][0]["color"] = "#123456"
    out = export_scene(scene)
    assert "rgb(18, 52, 86)" in out["svg"]                     # the edited colour is exported
    layers = json.loads(out["lottie"])["layers"]
    assert len(layers) == len(scene["objects"]) + 1
