"""Classical reconstruction: GIF frames -> objects with outlines and per-frame transforms.

Pipeline
1. palette: find the background and the dominant colours, label every pixel.
2. observe: connected components per colour and frame.
3. track: link components into objects across frames (Hungarian matching on centroids).
4. fit: for each object, take its largest view as a template and fit (x, y, rotation,
   scale) on every frame, ignoring template pixels hidden by another colour.
5. vectorise: polygon outline (and hole) from the template.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageSequence
from scipy.optimize import linear_sum_assignment

from .scene import Shape

MIN_AREA = 25
GATE = 45.0


def load_frames(path: str) -> np.ndarray:
    gif = Image.open(path)
    return np.stack([np.asarray(f.convert("RGB")) for f in ImageSequence.Iterator(gif)])


# 1. palette --------------------------------------------------------------------------

def find_palette(frames: np.ndarray, merge: float = 34.0, min_share: float = 0.0003):
    border = np.concatenate([frames[0, 0], frames[0, -1], frames[0, :, 0], frames[0, :, -1]])
    vals, counts = np.unique(border, axis=0, return_counts=True)
    background = vals[counts.argmax()].astype(float)

    sample = frames[:: max(1, len(frames) // 12)].reshape(-1, 3).astype(float)
    bins = (sample // 8).astype(int)
    keys, inv, counts = np.unique(bins, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(-counts)
    centres: list[np.ndarray] = []
    weights: list[float] = []
    total = len(sample)
    for idx in order:
        if counts[idx] < min_share * total:
            break
        colour = sample[inv.ravel() == idx].mean(axis=0)
        for j, c in enumerate(centres):
            if np.linalg.norm(c - colour) < merge:
                w = weights[j]
                centres[j] = (c * w + colour * counts[idx]) / (w + counts[idx])
                weights[j] += counts[idx]
                break
        else:
            centres.append(colour)
            weights.append(float(counts[idx]))
    # Candidate colours (with their pixel counts), excluding the background.
    cands = [(c, w) for c, w in zip(centres, weights) if np.linalg.norm(c - background) > merge]
    refs = [(background, float("inf"))] + cands
    keep = []
    for c, w in cands:
        # A rare colour lying on the segment between two much more frequent colours is an
        # antialiasing / dithering blend, not a real object colour.
        blend = False
        for i in range(len(refs)):
            for j in range(i + 1, len(refs)):
                (ca, wa), (cb, wb) = refs[i], refs[j]
                if ca is c or cb is c or w > 0.15 * min(wa, wb):
                    continue
                ab = cb - ca
                t = np.clip(np.dot(c - ca, ab) / (np.dot(ab, ab) + 1e-9), 0.15, 0.85)
                if np.linalg.norm(ca + t * ab - c) < 12:
                    blend = True
        if not blend:
            keep.append(c)
    return background, np.array(keep)


def label_frames(frames: np.ndarray, background: np.ndarray, colours: np.ndarray) -> np.ndarray:
    """0 = background, k+1 = colour k. Dither is removed with a small mode filter."""
    refs = np.vstack([background, colours])
    flat = frames.reshape(-1, 3).astype(np.float32)
    d = ((flat[:, None, :] - refs[None].astype(np.float32)) ** 2).sum(-1)
    labels = d.argmin(1).astype(np.uint8).reshape(frames.shape[:3])
    out = np.empty_like(labels)
    for t, lab in enumerate(labels):
        out[t] = _mode3(lab, len(refs))
    return out


def _mode3(lab: np.ndarray, n: int) -> np.ndarray:
    votes = np.stack([cv2.boxFilter((lab == k).astype(np.float32), -1, (3, 3), normalize=False) for k in range(n)])
    return votes.argmax(0).astype(np.uint8)


# 2-3. observe and track ---------------------------------------------------------------

@dataclass
class Track:
    colour: int
    frames: dict[int, np.ndarray] = field(default_factory=dict)  # frame -> boolean mask

    def centroid(self, t: int) -> np.ndarray:
        ys, xs = np.nonzero(self.frames[t])
        return np.array([xs.mean(), ys.mean()])

    def last(self) -> int:
        return max(self.frames)


def track_objects(labels: np.ndarray) -> list[Track]:
    tracks: list[Track] = []
    n_colours = int(labels.max())
    for k in range(1, n_colours + 1):
        active: list[Track] = []
        for t, lab in enumerate(labels):
            n, comp, stats, cents = cv2.connectedComponentsWithStats((lab == k).astype(np.uint8), connectivity=8)
            obs = [(comp == i, cents[i], stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)
                   if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA]
            live = [tr for tr in active if t - tr.last() <= 6]
            used = set()
            if live and obs:
                cost = np.array([[np.linalg.norm(tr.centroid(tr.last()) - o[1]) for o in obs] for tr in live])
                rows, cols = linear_sum_assignment(cost)
                for r, c in zip(rows, cols):
                    if cost[r, c] < GATE:
                        live[r].frames[t] = obs[c][0]
                        used.add(c)
            for c, o in enumerate(obs):
                if c in used:
                    continue
                # A small extra piece near an existing object is a fragment cut by an occluder.
                near = [tr for tr in live if t in tr.frames and
                        np.linalg.norm(tr.centroid(t) - o[1]) < GATE and o[2] < 0.5 * tr.frames[t].sum()]
                if near:
                    near[0].frames[t] = near[0].frames[t] | o[0]
                else:
                    tr = Track(k, {t: o[0]})
                    active.append(tr)
                    tracks.append(tr)
    return [tr for tr in merge_tracks(tracks) if len(tr.frames) >= 3]


def merge_tracks(tracks: list[Track]) -> list[Track]:
    """Join pieces of the same object that tracking split apart.

    - Re-appearance: an object hidden for a few frames comes back as a new track.
    - Fragment: an occluder cuts an object in two and one piece gets its own track.
    """
    tracks = sorted(tracks, key=lambda tr: min(tr.frames))
    changed = True
    while changed:
        changed = False
        for i, a in enumerate(tracks):
            for j, b in enumerate(tracks):
                if i == j or a.colour != b.colour:
                    continue
                if _is_continuation(a, b) or _is_fragment(b, a):
                    for t, m in b.frames.items():
                        a.frames[t] = a.frames[t] | m if t in a.frames else m
                    tracks.pop(j)
                    changed = True
                    break
            if changed:
                break
    return tracks


def _size(tr: Track) -> float:
    return max(m.sum() for m in tr.frames.values())


def _is_continuation(a: Track, b: Track) -> bool:
    a_end, b_start = max(a.frames), min(b.frames)
    gap = b_start - a_end
    if not -2 <= gap <= 15:
        return False
    if not 0.25 < _size(b) / _size(a) < 4.0:
        return False
    return np.linalg.norm(a.centroid(a_end) - b.centroid(b_start)) < GATE + 6 * gap


def _is_fragment(small: Track, big: Track) -> bool:
    common = set(small.frames) & set(big.frames)
    if not common or _size(small) > 0.8 * _size(big) or len(common) < 0.6 * len(small.frames):
        return False
    radius = math.sqrt(_size(big) / math.pi) * 1.6
    return all(np.linalg.norm(small.centroid(t) - big.centroid(t)) < radius for t in common)


# 4. fit ---------------------------------------------------------------------------------

@dataclass
class FittedObject:
    colour_index: int
    colour: tuple[int, int, int]
    ref_frame: int
    points: np.ndarray            # template pixel centres, relative to the reference centroid
    params: dict[int, np.ndarray]  # frame -> (x, y, rot_deg, scale)


def _score(points, p, labels_t, k):
    x, y, rot, s = p
    c, si = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    q = points @ np.array([[c, si], [-si, c]]) * s + [x, y]
    # Continuous coordinates: pixel i covers [i, i+1).
    u = np.floor(q[:, 0]).astype(int)
    v = np.floor(q[:, 1]).astype(int)
    h, w = labels_t.shape
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    lab = np.full(len(points), 0, np.uint8)
    lab[inside] = labels_t[v[inside], u[inside]]
    hits = np.count_nonzero(lab == k)
    misses = np.count_nonzero(lab == 0)  # template over background: wrong; over another colour: maybe hidden
    observed = np.count_nonzero(labels_t == k)
    a = s * s
    return hits * a / (observed + misses * a + 1e-6)


def fit_track(track: Track, labels: np.ndarray, colours: np.ndarray) -> FittedObject:
    k = track.colour
    ref = max(track.frames, key=lambda t: track.frames[t].sum())
    ys, xs = np.nonzero(track.frames[ref])
    # Pixel centres in continuous coordinates (pixel i is centred on i + 0.5).
    ref_c = np.array([xs.mean(), ys.mean()]) + 0.5
    points = np.c_[xs, ys] + 0.5 - ref_c
    ref_area = len(points)

    params: dict[int, np.ndarray] = {ref: np.array([*ref_c, 0.0, 1.0])}
    order_fwd = sorted(t for t in track.frames if t > ref)
    order_bwd = sorted((t for t in track.frames if t < ref), reverse=True)
    for order in (order_fwd, order_bwd):
        prev = params[ref].copy()
        for t in order:
            mask = track.frames[t]
            ys, xs = np.nonzero(mask)
            centroid = np.array([xs.mean(), ys.mean()]) + 0.5
            s0 = math.sqrt(max(mask.sum(), 1) / ref_area)
            candidates = [prev.copy(), np.array([*centroid, prev[2], prev[3]]),
                          np.array([*centroid, prev[2], np.clip(s0, 0.5, 2.0)])]
            best = max(candidates, key=lambda p: _score(points, p, labels[t], k))
            best = _refine(points, best, labels[t], k)
            params[t] = best
            prev = best
    colour = tuple(int(c) for c in colours[k - 1])
    return FittedObject(k, colour, ref, points, params)


def _refine(points, p, lab, k):
    p = p.copy()
    best = _score(points, p, lab, k)
    steps = np.array([2.0, 2.0, 6.0, 0.06])
    for _ in range(6):
        improved = False
        for i in range(4):
            for sign in (-1, 1):
                q = p.copy()
                q[i] += sign * steps[i]
                sc = _score(points, q, lab, k)
                if sc > best:
                    best, p, improved = sc, q, True
        if not improved:
            steps /= 2
            if steps[0] < 0.25:
                break
    return p


# 5. vectorise ---------------------------------------------------------------------------

def vectorise(obj: FittedObject) -> tuple[np.ndarray, np.ndarray | None]:
    """Outline (and hole) polygons of the template, in the object's local frame."""
    up = 4
    pts = obj.points
    lo = np.floor(pts.min(0)) - 2
    size = (np.ceil(pts.max(0) - lo) + 3).astype(int)
    mask = np.zeros((size[1] * up, size[0] * up), np.uint8)
    # Each template point is a pixel centre: paint the square [p - 0.5, p + 0.5).
    for x, y in (pts - lo - 0.5):
        mask[int(round(y * up)):int(round(y * up)) + up, int(round(x * up)):int(round(x * up)) + up] = 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((up + 1, up + 1), np.uint8))
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((3, 2)), None
    outer_i = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
    hole = None
    holes = [i for i in range(len(contours)) if hierarchy[0][i][3] == outer_i]
    if holes:
        hole_i = max(holes, key=lambda i: cv2.contourArea(contours[i]))
        if cv2.contourArea(contours[hole_i]) > 0.05 * cv2.contourArea(contours[outer_i]):
            hole = _simplify(contours[hole_i], up, lo)
    return _simplify(contours[outer_i], up, lo), hole


def _simplify(contour, up, lo, grow=0.0):
    approx = cv2.approxPolyDP(contour, 0.6 * up, True).reshape(-1, 2).astype(float)
    poly = approx / up + lo
    return _grow(poly, grow)


def _grow(poly, d):
    c = poly.mean(0)
    v = poly - c
    r = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return c + v * (1 + d / r)


# z-order and assembly ------------------------------------------------------------------

def to_shapes(objects: list[FittedObject], labels: np.ndarray, n_frames: int) -> list[Shape]:
    shapes = []
    for obj in objects:
        outer, hole = vectorise(obj)
        frames = sorted(obj.params)
        track = np.array([obj.params[t] for t in frames])
        full = np.empty((n_frames, 4))
        for i in range(4):
            full[:, i] = np.interp(np.arange(n_frames), frames, track[:, i])
        shapes.append(Shape("polygon", obj.colour, outer, hole, full))
    return _z_order(shapes, objects, labels)


def _z_order(shapes, objects, labels):
    """Shape A is above B when A's colour wins where their footprints overlap."""
    from .scene import transform
    n = len(shapes)
    wins = np.zeros(n)
    h, w = labels.shape[1:]
    for t in range(0, labels.shape[0], 3):
        masks = []
        for s in shapes:
            m = np.zeros((h, w), np.uint8)
            cv2.fillPoly(m, [np.rint(transform(s.outer, *s.track[t])).astype(np.int32)], 1)
            masks.append(m.astype(bool))
        for i in range(n):
            for j in range(i + 1, n):
                ov = masks[i] & masks[j]
                if ov.sum() < 10:
                    continue
                a = np.count_nonzero(labels[t][ov] == objects[i].colour_index)
                b = np.count_nonzero(labels[t][ov] == objects[j].colour_index)
                wins[i] += a > b
                wins[j] += b > a
    return [shapes[i] for i in np.argsort(wins, kind="stable")]


def reconstruct(path: str) -> dict:
    frames = load_frames(path)
    background, colours = find_palette(frames)
    labels = label_frames(frames, background, colours)
    tracks = track_objects(labels)
    objects = [fit_track(tr, labels, colours) for tr in tracks]
    shapes = to_shapes(objects, labels, len(frames))
    return {"frames": frames, "background": tuple(int(c) for c in background), "colours": colours,
            "labels": labels, "objects": objects, "shapes": shapes}
