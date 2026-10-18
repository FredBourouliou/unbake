"""Keyframes joined by cubic-Bézier easing curves, as in CSS, After Effects and Lottie.

A segment between two keyframes (t_a, v_a) -> (t_b, v_b) plays
    v(t) = v_a + (v_b - v_a) * E((t - t_a) / (t_b - t_a))
where E is the easing curve through (0, 0), (x1, y1), (x2, y2), (1, 1). Control points are
kept inside the unit square so the curve is monotonic, which SVG keySplines require and
which matches segments split at extrema.

A channel is fitted like Ramer–Douglas–Peucker, but with eased segments: split at the
signal's extrema first (where an animator puts keys), then recursively at the worst
frame until every segment fits one easing curve within tolerance.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

LINEAR = (1 / 3, 1 / 3, 2 / 3, 2 / 3)
_STARTS = [LINEAR, (0.42, 0.0, 0.58, 1.0), (0.42, 0.0, 1.0, 1.0), (0.0, 0.0, 0.58, 1.0),
           (0.25, 0.1, 0.25, 1.0), (0.55, 0.0, 0.85, 0.3), (0.15, 0.7, 0.45, 1.0)]
_S = np.linspace(0, 1, 129)


def ease(u: np.ndarray, c) -> np.ndarray:
    """Evaluate the easing curve with control points c = (x1, y1, x2, y2) at times u in [0, 1]."""
    x1, y1, x2, y2 = c
    s, m = _S, 1 - _S
    bx = 3 * m * m * s * x1 + 3 * m * s * s * x2 + s ** 3
    by = 3 * m * m * s * y1 + 3 * m * s * s * y2 + s ** 3
    return np.interp(u, bx, by)


def fit_segment(t: np.ndarray, v: np.ndarray) -> tuple[tuple, float]:
    """Best easing curve for samples (t, v) between their first and last point, and its max error."""
    span = v[-1] - v[0]
    u = (t - t[0]) / (t[-1] - t[0])
    if abs(span) < 1e-9:
        return LINEAR, float(np.max(np.abs(v - v[0])))
    y = (v - v[0]) / span
    best, best_err = LINEAR, np.inf
    for c0 in _STARTS:
        err0 = np.max(np.abs(ease(u, c0) - y))
        if err0 < best_err:
            best, best_err = c0, err0
    if len(t) > 3:
        res = least_squares(lambda c: ease(u, c) - y, best, bounds=([0, 0, 0, 0], [1, 1, 1, 1]), max_nfev=200)
        err = np.max(np.abs(ease(u, res.x) - y))
        if err < best_err:
            best, best_err = tuple(float(x) for x in res.x), err
    return tuple(float(x) for x in best), float(best_err * abs(span))


def _extrema(v: np.ndarray, tol: float) -> list[int]:
    """Indices where the signal turns around by more than `tol` (noise below it is ignored)."""
    keys, direction, anchor = [0], 0, 0
    for i in range(1, len(v)):
        if direction >= 0 and v[i] > v[anchor]:
            anchor = i if direction > 0 or v[i] - v[keys[-1]] > tol else anchor
            direction = 1 if v[i] - v[keys[-1]] > tol else direction
        elif direction <= 0 and v[i] < v[anchor]:
            anchor = i if direction < 0 or v[keys[-1]] - v[i] > tol else anchor
            direction = -1 if v[keys[-1]] - v[i] > tol else direction
        if direction > 0 and v[anchor] - v[i] > tol:
            keys.append(anchor)
            direction, anchor = -1, i
        elif direction < 0 and v[i] - v[anchor] > tol:
            keys.append(anchor)
            direction, anchor = 1, i
    if keys[-1] != len(v) - 1:
        keys.append(len(v) - 1)
    return sorted(set(keys))


def fit_channel(t: np.ndarray, v: np.ndarray, tol: float) -> list[tuple[int, tuple]]:
    """Keyframes as (index, easing to the next key); the last key's easing is unused."""
    if len(v) == 1:
        return [(0, LINEAR)]
    breaks = _extrema(v, 2 * tol)  # turn-arounds smaller than twice the tolerance are noise
    segments: dict[int, tuple] = {}
    stack = [(a, b) for a, b in zip(breaks[:-1], breaks[1:])]
    while stack:
        a, b = stack.pop()
        c, err = fit_segment(t[a:b + 1], v[a:b + 1])
        if err <= tol or b - a < 2:
            segments[a] = c
            continue
        # Split at the frame farthest from the best curve.
        u = (t[a:b + 1] - t[a]) / (t[b] - t[a])
        span = v[b] - v[a]
        model = v[a] + span * ease(u, c) if abs(span) > 1e-9 else np.full(b - a + 1, v[a])
        m = a + 1 + int(np.argmax(np.abs(v[a + 1:b] - model[1:-1])))
        stack += [(a, m), (m, b)]
    keys = sorted(segments)
    return [(k, segments[k]) for k in keys] + [(len(v) - 1, LINEAR)]


def sample_channel(frames: np.ndarray, values: np.ndarray, keys: list[tuple[int, tuple]], t: np.ndarray) -> np.ndarray:
    """Evaluate the keyframed channel at times t (frames and values are the keys' own)."""
    out = np.empty(len(t))
    kf = np.array([frames[k] for k, _ in keys], float)
    kv = np.array([values[k] for k, _ in keys], float)
    for i, x in enumerate(t):
        j = int(np.clip(np.searchsorted(kf, x, side="right") - 1, 0, len(kf) - 2)) if len(kf) > 1 else 0
        if len(kf) == 1 or x <= kf[0]:
            out[i] = kv[0]
        elif x >= kf[-1]:
            out[i] = kv[-1]
        else:
            u = (x - kf[j]) / (kf[j + 1] - kf[j])
            out[i] = kv[j] + (kv[j + 1] - kv[j]) * float(ease(np.array([u]), keys[j][1])[0])
    return out
