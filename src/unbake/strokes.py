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
            line = _smooth_resample(path, closed)
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
    """Round caps make the visible run longer by half a width at each free end: try both caps."""
    T = len(raw)
    total = _length(line, closed)
    track = np.where(np.isnan(raw[:, :1]), np.nan, 0.0) * np.ones((1, 4))
    track[~np.isnan(raw[:, 0])] = [0.0, 0.0, 0.0, 1.0]
    best, best_iou = None, -1.0
    frames = np.flatnonzero(~np.isnan(raw[:, 0]))
    sample = frames[:: max(1, len(frames) // 8)]
    for cap in ("round", "butt"):
        trim = raw.copy()
        if cap == "round":
            d = (width / 2) / total
            full = (raw[:, 0] == 0) & (raw[:, 1] == 1)
            for t in frames:
                if full[t]:
                    continue
                s, e = raw[t]
                if closed or s > 0:
                    s = (s + d) % 1.0 if closed else min(s + d, 1.0)
                if closed or e < 1:
                    e = (e - d) % 1.0 if closed else max(e - d, 0.0)
                trim[t] = (s, e)
        shape = Shape("stroke", colour, line, None, track.copy(), width=width, closed=closed, cap=cap, trim=trim)
        probe = Shape("stroke", (255, 255, 255), line, None, shape.track, width=width, closed=closed, cap=cap, trim=trim)
        ious = []
        for t in sample:
            img = render_frame([probe], int(t), size, (0, 0, 0))
            drawn = img.max(axis=2) > 127
            obs = masks[t]
            u = np.count_nonzero(drawn | obs)
            if u:
                ious.append(np.count_nonzero(drawn & obs) / u)
        iou = float(np.mean(ious)) if ious else 0.0
        if iou > best_iou:
            best, best_iou = shape, iou
    return best
