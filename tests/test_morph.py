import json

from unbake.export import morph_keys, to_lottie, to_svg
from unbake.reconstruct import reconstruct
from unbake.web_api import export_scene, reconstruct_gif


def test_morphing_shape_comes_back_with_an_animated_outline(corpus):
    gif, truth = corpus["morph"]
    rec = reconstruct(str(gif))
    n_true = sum("morph" in g for g in truth["shapes"])
    morphs = [s for s in rec["shapes"] if s.morph is not None]
    assert len(morphs) == n_true
    for s in morphs:
        assert 2 <= len(morph_keys(s)) < 0.5 * (~(s.track[:, 0] != s.track[:, 0])).sum()  # a few keys, not one per frame


def test_rigid_samples_get_no_morph(corpus):
    for level in ("easy", "medium", "strokes", "gradients"):
        gif, _ = corpus[level]
        assert all(s.morph is None for s in reconstruct(str(gif))["shapes"])


def test_morph_export_and_editor_round_trip(corpus):
    gif, _ = corpus["morph"]
    rec = reconstruct(str(gif))
    n = len(rec["frames"])
    lottie = to_lottie(rec["shapes"], (256, 256), 24, n, rec["background"])
    paths = [it for layer in lottie["layers"] if layer["ty"] == 4 for it in layer["shapes"][0]["it"] if it["ty"] == "sh"]
    assert any(p["ks"]["a"] == 1 for p in paths)                 # an animated path
    assert '<animate attributeName="d"' in to_svg(rec["shapes"], (256, 256), 24, n, rec["background"])

    scene = reconstruct_gif(gif.read_bytes())
    obj = next(o for o in scene["objects"] if o.get("morph"))
    assert len(obj["morph"]) >= 2
    out = export_scene(json.loads(json.dumps(scene)))
    assert '<animate attributeName="d"' in out["svg"]
