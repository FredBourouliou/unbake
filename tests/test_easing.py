import re
from pathlib import Path

import numpy as np

from unbake.easing import ease, fit_channel, fit_segment, sample_channel


def test_fit_segment_recovers_a_known_easing():
    t = np.arange(25, dtype=float)
    v = 10 + 90 * ease(t / 24, (0.42, 0.0, 0.58, 1.0))
    c, err = fit_segment(t, v)
    assert err < 0.2


def test_bounce_needs_one_key_per_extreme():
    t = np.arange(49, dtype=float)
    v = 200 - 140 * np.abs(np.sin(t / 48 * np.pi * 2.5))   # 2.5 bounces: 6 extrema + ends
    keys = fit_channel(t, v, 0.5)
    assert len(keys) <= 9
    assert np.max(np.abs(sample_channel(t, v, keys, t) - v)) <= 0.5 + 1e-6


def test_worker_loads_every_module_the_web_api_needs():
    """The browser worker must ship every module imported (transitively) by web_api."""
    src = Path(__file__).resolve().parents[1] / "src" / "unbake"
    worker = (Path(__file__).resolve().parents[1] / "web" / "worker.js").read_text()
    shipped = set(re.search(r"MODULES = \[(.*?)\]", worker).group(1).replace("'", "").replace(" ", "").split(","))
    needed, todo = set(), ["web_api"]
    while todo:
        mod = todo.pop()
        if mod in needed:
            continue
        needed.add(mod)
        todo += re.findall(r"^\s*from \.(\w+) import", (src / f"{mod}.py").read_text(), re.M)
    assert {f"{m}.py" for m in needed} <= shipped
