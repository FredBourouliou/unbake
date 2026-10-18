import numpy as np

from unbake.scene import Shape, holes, outline, render_frame, sub_path


def test_sub_path_open_and_wrapping():
    line = np.c_[np.linspace(0, 100, 101), np.zeros(101)]
    (piece,) = sub_path(line, False, 0.25, 0.75)
    assert piece[0][0] == 25 and piece[-1][0] == 75
    loop = np.c_[np.cos(np.linspace(0, 2 * np.pi, 64, endpoint=False)), np.sin(np.linspace(0, 2 * np.pi, 64, endpoint=False))]
    assert len(sub_path(loop, True, 0.9, 0.1)) == 2      # wraps around the start of a loop
    assert sub_path(line, False, 0.9, 0.1) == []         # an open path cannot wrap


def test_ring_has_a_transparent_hole():
    outer, hole = outline("ring", 40)
    ring = Shape("ring", (255, 0, 0), outer, hole, np.array([[64.0, 64.0, 0.0, 1.0]]))
    img = render_frame([ring], 0, (128, 128), (0, 0, 0))
    assert img[64, 64].sum() == 0          # centre: background shows through
    assert img[64, 64 + 32, 0] > 200       # on the band: red
    assert len(holes(ring)) == 1


def test_stroke_renders_only_its_visible_part():
    line = np.c_[np.linspace(-50, 50, 101), np.zeros(101)]
    stroke = Shape("stroke", (255, 255, 255), line, None, np.array([[64.0, 64.0, 0.0, 1.0]]),
                   width=6, closed=False, cap="butt", trim=np.array([[0.0, 0.5]]))
    img = render_frame([stroke], 0, (128, 128), (0, 0, 0))
    assert img[64, 30].sum() > 600 and img[64, 100].sum() == 0
