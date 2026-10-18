"""Linear gradient fills.

Idea: once dithering is blurred away, the inside of a flat shape has a constant colour and
the inside of a gradient-filled shape has a colour that is an affine function of position,
c(p) = a + B p, with B of rank one (all channels change along the same direction). Edges,
where colours jump, separate the two.

1. Per frame, the smooth regions (no colour edge) of the foreground are fitted with an affine
   colour model; a region whose colour changes by a lot, with a small residual, is a gradient.
2. Gradient regions are grouped across frames by their colour line (the segment of colour
   space they cover), so each group behaves like one more palette colour for the tracker.
3. Per tracked object, the gradient is fitted in the object's local frame, pooled over its
   views, and read off as two end points and two colours. A disc has no visible orientation,
   but its gradient does: there, the gradient's direction gives the rotation.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

EDGE = 20.0         # colour change per pixel (sum over channels, after blur) that marks an edge
MIN_RANGE = 40.0    # a gradient region spans at least this colour change
MAX_RESIDUAL = 8.0  # RMS distance to the affine colour model
MIN_REGION = 40
MIN_FRAMES = 3
LINE_GATE = 30.0    # colour distance under which two regions share a colour line


def blur(frames: np.ndarray) -> np.ndarray:
    return np.stack([cv2.GaussianBlur(f.astype(np.float32), (0, 0), 1.2) for f in frames])


def _edges(f: np.ndarray) -> np.ndarray:
    return sum(np.abs(cv2.Sobel(f[..., c], cv2.CV_32F, 1, 0, ksize=3)) +
               np.abs(cv2.Sobel(f[..., c], cv2.CV_32F, 0, 1, ksize=3)) for c in range(3)) / 8


def _affine(xs: np.ndarray, ys: np.ndarray, colours: np.ndarray):
    """Least-squares colour model c = a + B (x, y): returns a, B (2 x 3) and the RMS residual."""
    X = np.c_[np.ones(len(xs)), xs, ys]
    coef, *_ = np.linalg.lstsq(X, colours, rcond=None)
    res = colours - X @ coef
    return coef[0], coef[1:], float(np.sqrt((res ** 2).mean()))


def _line_distance(a0, a1, b0, b1) -> float:
    """How far two colour segments are from being parts of one colour line.

    A partly hidden gradient shows only part of its colours, so two views of the same fill are
    collinear, overlapping segments: the shorter one's ends lie near the longer one's line,
    within (or just past) its extent.
    """
    if np.linalg.norm(a1 - a0) > np.linalg.norm(b1 - b0):
        a0, a1, b0, b1 = b0, b1, a0, a1
    ab = b1 - b0
    L = float(np.linalg.norm(ab))
    if L < 1e-6:
        return float(max(np.linalg.norm(a0 - b0), np.linalg.norm(a1 - b0)))
    u = ab / L
    off = max(float(np.linalg.norm((p - b0) - ((p - b0) @ u) * u)) for p in (a0, a1))
    lo, hi = sorted(float((p - b0) @ u) for p in (a0, a1))
    gap = max(0.0, lo - L, -hi)  # distance between the two extents along the line
    return max(off, gap)


def _absorb(g: list, e0, e1, colour_sum, n) -> None:
    """Add a segment to a group: keep the two farthest-apart ends, accumulate the mean colour."""
    ends = [g[0], g[1], e0, e1]
    i, j = max(((i, j) for i in range(4) for j in range(i + 1, 4)),
               key=lambda ij: np.linalg.norm(ends[ij[0]] - ends[ij[1]]))
    g[0], g[1] = ends[i], ends[j]
    g[2] = g[2] + colour_sum
    g[3] += n


def _frame_regions(f: np.ndarray, raw: np.ndarray, background: np.ndarray) -> list:
    """Gradient regions of one frame: (mask, end colour 0, end colour 1, pixel count)."""
    fg = np.abs(f - background).sum(-1) > 30
    edges = _edges(f)
    found = []  # (mask, a, B, e0, e1, n)
    # Two gradients that touch along a faint edge come out as one region with a large residual:
    # such regions are split again with a lower edge threshold.
    todo = [fg]
    for threshold in (EDGE, 0.6 * EDGE, 0.35 * EDGE):
        retry = []
        for domain in todo:
            smooth = (edges < threshold) & domain
            n, comp, stats, _ = cv2.connectedComponentsWithStats(smooth.astype(np.uint8), connectivity=4)
            for i in range(1, n):
                if stats[i, cv2.CC_STAT_AREA] < MIN_REGION:
                    continue
                mask = comp == i
                ys, xs = np.nonzero(mask)
                a, B, rms = _affine(xs, ys, f[ys, xs])
                proj = (np.c_[xs, ys] - [xs.mean(), ys.mean()]) @ B  # colour offsets from the mean
                if np.ptp(proj, axis=0).sum() < MIN_RANGE:
                    continue  # flat: left to the palette
                if rms > MAX_RESIDUAL:
                    retry.append(mask)
                    continue
                centre = f[ys, xs].mean(0)
                u = np.linalg.svd(B)[2][0]  # main colour direction
                sc = proj @ u
                found.append((mask, a, B, centre + sc.min() * u, centre + sc.max() * u, len(xs)))
        todo = retry
    # The antialiased edge band: nearby pixels whose colour is closer to the region's own model
    # than to the background, and not inside another region.
    taken = np.zeros(fg.shape, bool)
    for r in found:
        taken |= r[0]
    H, W = fg.shape
    yy, xx = np.mgrid[0:H, 0:W]
    out = []
    for mask, a, B, e0, e1, n in found:
        band = (cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0) & ~taken
        ys, xs = np.nonzero(band)
        pred = a + np.c_[xs, ys] @ B
        d_own = np.abs(raw[ys, xs] - pred).sum(1)
        d_bg = np.abs(raw[ys, xs] - background).sum(1)
        keep = (d_own < d_bg) & (d_own < 90)
        grown = mask.copy()
        grown[ys[keep], xs[keep]] = True
        out.append((grown, e0, e1, n))
    return out


def detect_gradients(frames: np.ndarray, background: np.ndarray):
    """Gradient regions per frame, grouped by colour line.

    Returns (group map T x H x W with 0 = none and j + 1 = group j, mean colour per group).
    """
    T, H, W = frames.shape[:3]
    soft = blur(frames)
    regions = []  # (t, mask, end colour 0, end colour 1, pixel count)
    for t in range(T):
        regions += [(t, *r) for r in _frame_regions(soft[t], frames[t].astype(np.float32), background)]

    groups: list[list] = []  # [end colour 0, end colour 1, colour sum, pixel count]
    owner = []               # group of each region
    for t, mask, e0, e1, n in regions:
        best, best_d = None, LINE_GATE
        for j, g in enumerate(groups):
            d = _line_distance(e0, e1, g[0], g[1])
            if d < best_d:
                best, best_d = j, d
        if best is None:
            groups.append([e0, e1, (e0 + e1) / 2 * n, n])
            best = len(groups) - 1
        else:
            _absorb(groups[best], e0, e1, (e0 + e1) / 2 * n, n)
        owner.append(best)
    # A group may have grown until it covers another one seen earlier: merge until stable.
    parent = list(range(len(groups)))
    merged = True
    while merged:
        merged = False
        live = [j for j in range(len(groups)) if parent[j] == j]
        for a in live:
            for b in live:
                if a < b and parent[a] == a and parent[b] == b and \
                        _line_distance(groups[b][0], groups[b][1], groups[a][0], groups[a][1]) < LINE_GATE:
                    _absorb(groups[a], groups[b][0], groups[b][1], groups[b][2], groups[b][3])
                    parent[b] = a
                    merged = True
    root = lambda j: j if parent[j] == j else root(parent[j])  # noqa: E731
    # A fill seen on one or two frames only (a motion-blurred transition) is not a gradient object.
    seen: dict[int, set] = {}
    for (t, *_), j in zip(regions, owner):
        seen.setdefault(root(j), set()).add(t)
    keep = sorted(j for j in seen if len(seen[j]) >= MIN_FRAMES)
    index = {j: i + 1 for i, j in enumerate(keep)}
    group_map = np.zeros((T, H, W), np.uint8)
    for (t, mask, *_), j in zip(regions, owner):
        if root(j) in index:
            group_map[t][mask & (group_map[t] == 0)] = index[root(j)]
    means = np.array([groups[j][2] / groups[j][3] for j in keep]) if keep else np.zeros((0, 3))
    return group_map, means, soft


def fit_gradient(obj, masks: dict[int, np.ndarray], soft: np.ndarray, round_shape: bool):
    """Gradient of a tracked object in its local frame: (p0, p1, colour0, colour1).

    For a round shape the rotation is re-estimated from the gradient's direction (in place).
    """
    kernel = np.ones((5, 5), np.uint8)
    frames = sorted(t for t in masks if t in obj.params)
    views = {}
    for t in frames:
        inner = cv2.erode(masks[t].astype(np.uint8), kernel) > 0
        ys, xs = np.nonzero(inner)
        if len(xs) < 30:
            continue
        views[t] = (xs + 0.5, ys + 0.5, soft[t][ys, xs])
    if not views:
        return None

    def local(t, xs, ys):
        x, y, rot, scale = obj.params[t]
        c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
        q = np.c_[xs - x, ys - y] / scale
        return q @ np.array([[c, s], [-s, c]]).T

    if round_shape:
        # Direction of the gradient on each frame, in image space, relative to the reference view.
        # SVD signs are arbitrary: orient every view by the colour direction of the largest one.
        largest = max(views, key=lambda t: len(views[t][0]))
        ref_colour = np.linalg.svd(_affine(*views[largest])[1])[2][0]
        angles = {}
        for t, (xs, ys, col) in views.items():
            _, B, _ = _affine(xs, ys, col)
            U, S, Vt = np.linalg.svd(B)
            v = U[:, 0] * (1.0 if Vt[0] @ ref_colour >= 0 else -1.0)
            angles[t] = math.degrees(math.atan2(v[1], v[0]))
        ref = obj.ref_frame if obj.ref_frame in angles else max(views, key=lambda t: len(views[t][0]))
        prev = None
        for t in sorted(angles):
            a = angles[t] - angles[ref]
            if prev is not None:
                a = prev + ((a - prev + 180) % 360 - 180)  # unwrap
            obj.params[t][2] = a
            prev = a
        # Frames without a gradient view keep the nearest estimate.
        known = sorted(angles)
        for t in obj.params:
            if t not in angles:
                obj.params[t][2] = obj.params[min(known, key=lambda k: abs(k - t))][2]

    L = np.vstack([local(t, xs, ys) for t, (xs, ys, _) in views.items()])
    C = np.vstack([col for _, _, col in views.values()])
    a, B, _ = _affine(L[:, 0], L[:, 1], C)
    U, S, Vt = np.linalg.svd(B)
    v, colour_dir = U[:, 0], Vt[0] * S[0]
    proj = obj.points @ v
    lo, hi = float(proj.min()), float(proj.max())
    # a is the colour at the local origin; the model along v is a + colour_dir * t.
    c0 = np.clip(a + colour_dir * lo, 0, 255)
    c1 = np.clip(a + colour_dir * hi, 0, 255)
    return v * lo, v * hi, tuple(int(round(x)) for x in c0), tuple(int(round(x)) for x in c1)
