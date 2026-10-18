"""Animated strokes: lines that draw themselves (Lottie "trim paths").

Idea: over the whole animation, an animated stroke only ever covers its own path, so the
union of its pixels is a *thin* region, while its area changes from frame to frame. A rigid
shape that moves sweeps a thick region instead (even a thin clock hand turning sweeps a disc).

For each such region: skeleton -> ordered centre line (open or closed), width from the
distance transform, then per frame the visible interval along the line.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .scene import Shape, render_frame

MAX_WIDTH = 16.0
# A real stroke, rendered with its fitted width and trim, covers its observed pixels nearly as
# well as a one-pixel edge error allows: IoU / ((w - 1) / (w + 1)) is 0.81 or more for 289 of
# 290 dev strokes and every stroke in the real animations, even thin ones. Rigid shapes mistaken
# for strokes (a ring scaling in place, a thin arc inside a swept area) stay at 0.73 or less.
MIN_FIT = 0.77


def detect_strokes(labels: np.ndarray, colours: np.ndarray) -> tuple[list[Shape], np.ndarray]:
    """Return stroke shapes and a boolean array (frames × h × w) of the pixels they explain."""
    T, H, W = labels.shape
    shapes: list[Shape] = []
    consumed = np.zeros(labels.shape, bool)
    for k in range(1, int(labels.max()) + 1):
        on = labels == k
        union = on.any(0)
        n, comp = cv2.connectedComponents(union.astype(np.uint8), connectivity=8)
        for i in range(1, n):
            region = comp == i
            if region.sum() < 60:
                continue
            per_frame = (on & region).reshape(T, -1).sum(1)
            present = per_frame > 0
            if present.sum() < 3:
                continue
            areas = per_frame[present]
            if present.all() and areas.max() < 1.35 * np.percentile(areas, 10):
                continue  # constant area: a rigid object, not a line being drawn
            dt = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
            skel = skeletonize(region)
            half = dt[skel]
            if half.size < 10:
                continue
            width = 2 * float(np.median(half))
            if width > MAX_WIDTH or skel.sum() < 4 * width or np.percentile(half, 90) > 1.8 * np.median(half) + 0.75:
                continue
            path, closed = order_skeleton(skel, width)
            if path is None or len(path) < 8:
                continue
            # A branched line drawing (an outlined icon) is not one stroke: leave it to the rigid model.
            if len(path) < 0.75 * skel.sum():
                continue
            line = _smooth_resample(path, closed)
            line, width = _refine_centre(line, closed, region, width)
            raw = _visible_intervals(line, closed, on & region[None], width)
            if raw is None:
                continue
            # Short-lived thin slivers (mislabelled pixels between similar colours) are not strokes.
            visible = int(np.sum(~np.isnan(raw[:, 0])))
            if visible < max(8, 0.25 * T):
                continue
            lengths = np.where(np.isnan(raw[:, 0]), np.nan, (raw[:, 1] - raw[:, 0]) % 1.0)
            total = _length(line, closed)
            # A blob gliding along a path (a moving dot) is a rigid object, not a stroke.
            if np.nanmax(lengths) * total < 3 * width:
                continue
            colour = tuple(int(c) for c in colours[k - 1])
            shape = _best_cap(line, closed, width, raw, colour, on & region[None], (W, H))
            iou = _mean_iou(shape, on & region[None], np.flatnonzero(~np.isnan(raw[:, 0])), (W, H))
            if iou < MIN_FIT * (shape.width - 1) / (shape.width + 1):
                continue
            shapes.append(shape)
            consumed |= on & region[None]
    return shapes, consumed


# Skeleton -> ordered polyline ------------------------------------------------------------

_NEIGH = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def _graph(skel: np.ndarray) -> dict:
    ys, xs = np.nonzero(skel)
    pts = set(zip(ys.tolist(), xs.tolist()))
    return {p: [(p[0] + dy, p[1] + dx) for dy, dx in _NEIGH if (p[0] + dy, p[1] + dx) in pts] for p in pts}


def _prune(g: dict, min_len: float) -> dict:
    """Remove short spurs hanging off branch points."""
    changed = True
    while changed:
        changed = False
        for p in [p for p, nb in g.items() if len(nb) == 1]:
            chain, cur, prev = [p], p, None
            while True:
                nxt = [q for q in g[cur] if q != prev]
                if len(g[cur]) != 2 and cur != p or not nxt:
                    break
                prev, cur = cur, nxt[0]
                chain.append(cur)
                if len(g[cur]) > 2:
                    break
            if len(g.get(cur, [])) > 2 and len(chain) - 1 < min_len:
                for q in chain[:-1]:
                    for r in g.pop(q, []):
                        if r in g:
                            g[r] = [s for s in g[r] if s != q]
                changed = True
                break
    return g


def _bfs(g: dict, src) -> tuple[dict, object]:
    dist, prev, queue = {src: 0}, {src: None}, [src]
    for p in queue:
        for q in g[p]:
            if q not in dist:
                dist[q], prev[q] = dist[p] + 1, p
                queue.append(q)
    far = max(dist, key=dist.get)
    return prev, far


def order_skeleton(skel: np.ndarray, width: float) -> tuple[np.ndarray | None, bool]:
    g = _prune(_graph(skel), max(3.0, width))
    if not g:
        return None, False
    ends = [p for p, nb in g.items() if len(nb) == 1]
    if not ends:
        # Closed loop: walk around it.
        start = min(g)
        path, prev, cur = [start], None, start
        seen = {start}
        while True:
            nxt = [q for q in g[cur] if q not in seen]
            if not nxt:
                break
            nxt.sort(key=lambda q: abs(q[0] - cur[0]) + abs(q[1] - cur[1]))
            prev, cur = cur, nxt[0]
            path.append(cur)
            seen.add(cur)
        pts = np.array([(x, y) for y, x in path], float)
        return pts + 0.5, True
    # Open: longest geodesic path (two BFS passes).
    _, a = _bfs(g, ends[0])
    prev, b = _bfs(g, a)
    path, cur = [], b
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    pts = np.array([(x, y) for y, x in path], float)
    return pts + 0.5, False


def _smooth_resample(path: np.ndarray, closed: bool, step: float = 1.0) -> np.ndarray:
    k = 5
    if closed:
        ext = np.vstack([path[-k:], path, path[:k]])
    else:
        ext = np.vstack([np.repeat(path[:1], k, 0), path, np.repeat(path[-1:], k, 0)])
    kern = np.ones(2 * k + 1) / (2 * k + 1)
    sm = np.c_[np.convolve(ext[:, 0], kern, "same"), np.convolve(ext[:, 1], kern, "same")][k:-k]
    if not closed:
        sm[0], sm[-1] = path[0], path[-1]
    pts = np.vstack([sm, sm[:1]]) if closed else sm
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.r_[0, np.cumsum(seg)]
    s = np.arange(0, cum[-1], step)
    out = np.c_[np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])]
    return out


def _refine_centre(line: np.ndarray, closed: bool, region: np.ndarray, width: float) -> tuple[np.ndarray, float]:
    """Sub-pixel centre line and width: recentre each point between the region's edges along its normal."""
    H, W = region.shape
    if closed:
        tang = np.roll(line, -2, 0) - np.roll(line, 2, 0)
    else:
        tang = np.gradient(line, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    normal = np.c_[-tang[:, 1], tang[:, 0]]
    steps = np.arange(-1.5 * width - 2, 1.5 * width + 2, 0.25)
    offsets, widths = np.zeros(len(line)), np.full(len(line), np.nan)
    for i, (p, n) in enumerate(zip(line, normal)):
        q = p + steps[:, None] * n
        x, y = np.floor(q[:, 0]).astype(int), np.floor(q[:, 1]).astype(int)
        ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        inside = np.zeros(len(steps), bool)
        inside[ok] = region[y[ok], x[ok]]
        c = len(steps) // 2
        if not inside[c]:
            continue
        a = c
        while a > 0 and inside[a - 1]:
            a -= 1
        b = c
        while b < len(steps) - 1 and inside[b + 1]:
            b += 1
        lo, hi = steps[a] - 0.125, steps[b] + 0.125
        offsets[i] = (lo + hi) / 2
        widths[i] = hi - lo
    k = 7
    kern = np.ones(2 * k + 1) / (2 * k + 1)
    if closed:
        sm = np.convolve(np.r_[offsets[-k:], offsets, offsets[:k]], kern, "same")[k:-k]
    else:
        sm = np.convolve(np.r_[np.repeat(offsets[:1], k), offsets, np.repeat(offsets[-1:], k)], kern, "same")[k:-k]
    refined = line + sm[:, None] * normal
    good = widths[~np.isnan(widths)]
    return refined, (float(np.median(good)) if good.size else width)


def _length(line: np.ndarray, closed: bool) -> float:
    pts = np.vstack([line, line[:1]]) if closed else line
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


# Visible interval per frame -------------------------------------------------------------

def _visible_intervals(line: np.ndarray, closed: bool, masks: np.ndarray, width: float) -> np.ndarray | None:
    """(start, end) fractions of the longest visible run along the line, per frame."""
    T, H, W = masks.shape
    M = len(line)
    xi = np.clip(np.floor(line[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.floor(line[:, 1]).astype(int), 0, H - 1)
    out = np.full((T, 2), np.nan)
    kernel = np.ones((3, 3), np.uint8)
    for t in range(T):
        if not masks[t].any():
            continue
        m = cv2.dilate(masks[t].astype(np.uint8), kernel) > 0
        v = m[yi, xi]
        v = _close_gaps(v, 3, closed)
        if not v.any():
            continue
        if v.all():
            out[t] = (0.0, 1.0)
            continue
        a, b = _longest_run(v, closed)
        out[t] = (a / M, ((b + 1) / M) % 1.0 if closed else (b + 1) / M)
    return out if not np.isnan(out[:, 0]).all() else None


def _close_gaps(v: np.ndarray, gap: int, closed: bool) -> np.ndarray:
    v = v.copy()
    idx = np.flatnonzero(v)
    if len(idx) < 2:
        return v
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= gap + 1:
            v[a:b] = True
    if closed and len(v) - idx[-1] + idx[0] <= gap + 1:
        v[idx[-1]:] = True
        v[:idx[0]] = True
    return v


def _longest_run(v: np.ndarray, closed: bool) -> tuple[int, int]:
    M = len(v)
    if closed:
        shift = int(np.flatnonzero(~v)[0])  # start scanning just after a hidden sample
        vv = np.roll(v, -shift)
    else:
        shift, vv = 0, v
    best, cur_start, best_run = 0, None, (0, 0)
    for i, x in enumerate(np.r_[vv, False]):
        if x and cur_start is None:
            cur_start = i
        elif not x and cur_start is not None:
            if i - cur_start > best:
                best, best_run = i - cur_start, (cur_start, i - 1)
            cur_start = None
    a, b = best_run
    return (a + shift) % M, (b + shift) % M


def _best_cap(line, closed, width, raw, colour, masks, size) -> Shape:
    """Pick cap and width together: the pair whose rendering best matches the observed pixels.

    Round caps make the visible run longer by half a width at each free end, so the trim is
    corrected accordingly for that cap.
    """
    total = _length(line, closed)
    track = np.full((len(raw), 4), np.nan)
    track[~np.isnan(raw[:, 0])] = [0.0, 0.0, 0.0, 1.0]
    frames = np.flatnonzero(~np.isnan(raw[:, 0]))
    sample = frames[:: max(1, len(frames) // 8)]
    best, best_iou = None, -1.0
    for cap in ("round", "butt"):
        for w in (width - 1.0, width - 0.5, width, width + 0.5):
            if w < 1.5:
                continue
            trim = _cap_trim(raw, frames, closed, (w / 2) / total if cap == "round" else 0.0)
            probe = Shape("stroke", (255, 255, 255), line, None, track, width=w, closed=closed, cap=cap, trim=trim)
            iou = _mean_iou(probe, masks, sample, size)
            if iou > best_iou:
                best_iou = iou
                best = Shape("stroke", colour, line, None, track.copy(), width=w, closed=closed, cap=cap, trim=trim)
    return best


def _mean_iou(shape: Shape, masks: np.ndarray, frames, size) -> float:
    """Mean IoU, over the given frames, between the stroke's rendering and the observed pixels."""
    white = Shape("stroke", (255, 255, 255), shape.outer, None, shape.track, width=shape.width,
                  closed=shape.closed, cap=shape.cap, trim=shape.trim)
    ious = []
    for t in frames:
        drawn = render_frame([white], int(t), size, (0, 0, 0)).max(axis=2) > 127
        u = np.count_nonzero(drawn | masks[t])
        if u:
            ious.append(np.count_nonzero(drawn & masks[t]) / u)
    return float(np.mean(ious)) if ious else 0.0


def _cap_trim(raw, frames, closed, d):
    trim = raw.copy()
    if d == 0:
        return trim
    for t in frames:
        s, e = raw[t]
        if s == 0 and e == 1:
            continue
        if closed or s > 0:
            s = (s + d) % 1.0 if closed else min(s + d, 1.0)
        if closed or e < 1:
            e = (e - d) % 1.0 if closed else max(e - d, 0.0)
        trim[t] = (s, e)
    return trim
