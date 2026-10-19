"""Morphing shapes: outlines that change over time (Lottie animated paths, SVG animated d)."""
from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d


def _iou(outline, p, mask: np.ndarray, labels_t: np.ndarray) -> float:
    """IoU between the outline placed with pose p and the mask; pixels over another colour are neutral."""
    from .scene import transform
    outer, hole_list = outline
    h, w = mask.shape
    cover = np.zeros((h, w), np.uint8)
    # Pixel i spans [i, i + 1): shift by half a pixel, 4 fractional bits for sub-pixel edges.
    poly = lambda pts: np.rint((transform(pts, *p) - 0.5) * 16).astype(np.int32)  # noqa: E731
    cv2.fillPoly(cover, [poly(outer)], 1, shift=4)
    for hp in hole_list or []:
        cv2.fillPoly(cover, [poly(hp)], 0, shift=4)
    cover = cover > 0
    hits = np.count_nonzero(cover & mask)
    wrong = np.count_nonzero(cover & (labels_t == 0) & ~mask) + np.count_nonzero(mask & ~cover)
    return hits / max(1, hits + wrong)


def _best_pose(outline, p, mask, labels_t) -> float:
    """Best IoU of the rigid outline near pose p (the tracker's pose can be off when scale changes a lot)."""
    p = np.array(p, float)
    best = _iou(outline, p, mask, labels_t)
    steps = np.array([1.0, 1.0, 4.0, 0.04])
    for _ in range(30):
        improved = False
        for i in range(4):
            for sign in (-1, 1):
                q = p.copy()
                q[i] += sign * steps[i]
                v = _iou(outline, q, mask, labels_t)
                if v > best:
                    best, p, improved = v, q, True
        if not improved:
            steps /= 2
            if steps[0] < 0.125:
                break
    return best


def rigid_fit(obj, masks: dict[int, np.ndarray], labels: np.ndarray, outline, frames=None, refine=False) -> dict[int, float]:
    """Per frame, IoU between the placed rigid outline and the object's pixels."""
    out = {}
    for t, p in obj.params.items():
        if t not in masks or (frames is not None and t not in frames):
            continue
        out[t] = (_best_pose if refine else _iou)(outline, p, masks[t], labels[t])
    return out


def shape_descriptors(masks: dict[int, np.ndarray], labels: np.ndarray) -> dict[int, tuple[float, float]]:
    """(compactness, solidity) on the frames where the object is seen whole.

    Both are unchanged by moving, turning or scaling, so a rigid object keeps them while a
    morphing one does not. Frames where the object touches another colour (it may be partly
    hidden) or the image border are left out.
    """
    out = {}
    kernel = np.ones((5, 5), np.uint8)
    h, w = labels.shape[1:]
    for t, m in masks.items():
        ys, xs = np.nonzero(m)
        if len(xs) < 40 or xs.min() == 0 or ys.min() == 0 or xs.max() == w - 1 or ys.max() == h - 1:
            continue
        ring = (cv2.dilate(m.astype(np.uint8), kernel) > 0) & ~m
        if np.count_nonzero(labels[t][ring] > 0) > 0.02 * ring.sum():
            continue
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(contours, key=cv2.contourArea)
        area, per = cv2.contourArea(c), cv2.arcLength(c, True)
        hull = cv2.contourArea(cv2.convexHull(c))
        if area <= 0 or per <= 0 or hull <= 0:
            continue
        out[t] = (4 * math.pi * area / per ** 2, area / hull)
    return out


MORPH_PX = 1.2   # mean outline distance (px) that a rigid template must exceed to count as a morph
N_POINTS = 64
MIN_CLEAN = 5
SMOOTH_FRAMES = 0.7


def outline_error(fit: dict[int, float], masks: dict[int, np.ndarray], frames) -> np.ndarray:
    """(1 - IoU) x area / perimeter: the rigid outline's mean distance to the true edge, in pixels."""
    out = []
    for t in frames:
        m = masks[t].astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        per = sum(cv2.arcLength(cv2.approxPolyDP(c, 1.0, True), True) for c in contours)
        out.append((1 - fit[t]) * m.sum() / max(per, 1.0))
    return np.array(out)


def is_morph(obj, masks, labels, outline) -> tuple[bool, dict]:
    """A morph is an object that no rigid pose of its outline explains on the frames it is seen whole."""
    clean = shape_descriptors(masks, labels)
    frames = [t for t in clean if t in obj.params]
    if len(frames) < MIN_CLEAN:
        return False, clean
    fit = rigid_fit(obj, masks, labels, outline, frames)
    if np.percentile(outline_error(fit, masks, frames), 90) <= MORPH_PX:
        return False, clean
    fit = rigid_fit(obj, masks, labels, outline, frames, refine=True)
    return bool(np.percentile(outline_error(fit, masks, frames), 90) > MORPH_PX), clean


def _contour(mask: np.ndarray) -> np.ndarray | None:
    """Outer contour of a mask, smoothed like the rigid outlines, in continuous pixel coordinates."""
    up = 4
    ys, xs = np.nonzero(mask)
    lo = np.array([xs.min(), ys.min()]) - 2
    size = np.array([xs.max(), ys.max()]) - lo + 3
    big = cv2.resize(mask[lo[1]:lo[1] + size[1], lo[0]:lo[0] + size[0]].astype(np.uint8) * 255,
                     (int(size[0]) * up, int(size[1]) * up), interpolation=cv2.INTER_NEAREST)
    big = (cv2.GaussianBlur(big, (0, 0), up * 0.9) > 127).astype(np.uint8)
    contours, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
    return (c + 0.5) / up + lo


def _resample(poly: np.ndarray, n: int) -> np.ndarray:
    """n points evenly spaced along a closed polygon, counter-clockwise (in image axes)."""
    x, y = poly[:, 0], poly[:, 1]
    if 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0:
        poly = poly[::-1]
    pts = np.vstack([poly, poly[:1]])
    cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    s = np.linspace(0, cum[-1], n, endpoint=False)
    return np.c_[np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])]


def _centroid(poly: np.ndarray) -> np.ndarray:
    x, y = poly[:, 0], poly[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = cross.sum() / 2
    if abs(a) < 1e-9:
        return poly.mean(0)
    return np.array([((x + np.roll(x, -1)) * cross).sum(), ((y + np.roll(y, -1)) * cross).sum()]) / (6 * a)


def _align(prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
    """Cyclic shift of cur that best matches prev point by point."""
    costs = [np.sum((np.roll(cur, -k, axis=0) - prev) ** 2) for k in range(len(cur))]
    return np.roll(cur, -int(np.argmin(costs)), axis=0)


def fit_morph(obj, masks: dict[int, np.ndarray], clean: dict, n_frames: int):
    """Per-frame position and local outline of a morphing object, or None.

    Outlines are read on the frames where the object is seen whole; on the others (partly
    hidden, cut by the border) they are interpolated in time, and the position follows the
    rigid fit, offset as on the nearest clean frames.
    """
    frames = sorted(t for t in masks if t in obj.params)
    good = sorted(t for t in clean if t in obj.params)
    if len(good) < MIN_CLEAN:
        return None
    local, centre = {}, {}
    for t in good:
        c = _contour(masks[t])
        if c is None:
            continue
        pts = _resample(c, N_POINTS)
        centre[t] = _centroid(pts)
        local[t] = pts - centre[t]
    good = sorted(local)
    if len(good) < MIN_CLEAN:
        return None
    # Point correspondence: walk away from the middle clean frame, aligning each to its neighbour.
    mid = good[len(good) // 2]
    for order in (good[good.index(mid) + 1:], good[:good.index(mid)][::-1]):
        prev = local[mid]
        for t in order:
            local[t] = _align(prev, local[t])
            prev = local[t]
    g = np.array(good)
    L = np.stack([local[t] for t in good])
    offset = np.stack([centre[t] - obj.params[t][:2] for t in good])
    track = np.full((n_frames, 4), np.nan)
    morph = np.full((n_frames, N_POINTS, 2), np.nan)
    span = np.arange(frames[0], frames[-1] + 1)
    rigid_xy = np.stack([obj.params[t][:2] for t in frames])
    for t in span:
        xy = np.array([np.interp(t, frames, rigid_xy[:, i]) for i in range(2)])
        if t in local:
            morph[t], track[t] = local[t], [*centre[t], 0.0, 1.0]
            continue
        j = np.searchsorted(g, t)
        if j == 0 or j == len(g):
            k = 0 if j == 0 else len(g) - 1
            morph[t], off = L[k], offset[k]
        else:
            w = (t - g[j - 1]) / (g[j] - g[j - 1])
            morph[t] = L[j - 1] * (1 - w) + L[j] * w
            off = offset[j - 1] * (1 - w) + offset[j] * w
        track[t] = [*(xy + off), 0.0, 1.0]
    # Outlines traced from pixels jitter by a fraction of a pixel from frame to frame; smoothing
    # them in time keeps the motion and lets a few keys describe it.
    morph[span] = gaussian_filter1d(morph[span], SMOOTH_FRAMES, axis=0, mode="nearest")
    return track, morph
