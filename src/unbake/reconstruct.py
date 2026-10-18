"""Classical reconstruction: GIF frames -> objects with outlines and per-frame transforms.

Pipeline
1. palette: find the background and the dominant colours, label every pixel.
2. observe: connected components per colour and frame.
3. track: link components into objects across frames (Hungarian matching on centroids).
4. fit: for each object, take its largest view as a template and fit (x, y, rotation,
   scale) on every frame, ignoring template pixels hidden by another colour.
5. vectorise: polygon outline (and holes) from the template.
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
    """Decode a GIF at a constant frame rate.

    GIF frames can last different times (encoders merge identical consecutive frames and
    extend their duration), so each frame is repeated according to its duration.
    """
    gif = Image.open(path)
    frames, durations = [], []
    for f in ImageSequence.Iterator(gif):
        frames.append(np.asarray(f.convert("RGB")))
        durations.append(max(int(f.info.get("duration", 0) or 0), 1))
    base = min(durations)
    out = []
    for frame, d in zip(frames, durations):
        out.extend([frame] * max(1, round(d / base)))
    return np.stack(out)


# 1. palette --------------------------------------------------------------------------

def find_palette(frames: np.ndarray, merge: float = 34.0, min_share: float = 0.0003, exclude: np.ndarray | None = None):
    """Background and flat colours; pixels in `exclude` (gradient fills) are left out."""
    border = np.concatenate([frames[0, 0], frames[0, -1], frames[0, :, 0], frames[0, :, -1]])
    vals, counts = np.unique(border, axis=0, return_counts=True)
    background = vals[counts.argmax()].astype(float)

    step = max(1, len(frames) // 12)
    sample = frames[::step] if exclude is None else frames[::step][~exclude[::step]]
    sample = sample.reshape(-1, 3).astype(float)
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
    return background, np.array(keep).reshape(-1, 3)


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

    def radius(self) -> float:
        """Largest distance from the centroid to a pixel of the object, over its views."""
        r = 0.0
        for t, m in self.frames.items():
            ys, xs = np.nonzero(m)
            c = np.array([xs.mean(), ys.mean()])
            r = max(r, float(np.sqrt(((np.c_[xs, ys] - c) ** 2).sum(1)).max()))
        return r

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
                # A piece near an object joins it when an occluder cut it off, or when it is a
                # crumb (< 10 % of the object) split off by dithering.
                near = [tr for tr in live if t in tr.frames and o[2] < 0.9 * _size(tr) and
                        np.linalg.norm(tr.centroid(t) - o[1]) < max(12.0, 1.2 * tr.radius()) and
                        (o[2] < 0.1 * _size(tr) or _split_by_occluder(tr.frames[t], o[0], lab))]
                if near:
                    near[0].frames[t] = near[0].frames[t] | o[0]
                else:
                    tr = Track(k, {t: o[0]})
                    active.append(tr)
                    tracks.append(tr)
    return [tr for tr in merge_tracks(tracks, labels) if len(tr.frames) >= 3]


def _split_by_occluder(a: np.ndarray, b: np.ndarray, lab: np.ndarray) -> bool:
    """True when what separates masks a and b is another object, not background.

    Pieces of one object cut by an occluder have the occluder's colour between them;
    two distinct same-colour objects (a play triangle inside a ring) have background.
    """
    dist, labels_idx = cv2.distanceTransformWithLabels((~a).astype(np.uint8), cv2.DIST_L2, 5,
                                                       labelType=cv2.DIST_LABEL_PIXEL)
    by, bx = np.nonzero(b)
    i = int(np.argmin(dist[by, bx]))
    if dist[by[i], bx[i]] <= 2.0:
        return True
    # Nearest pixel of a to that point of b.
    ay, ax = np.nonzero(a)
    j = int(np.argmin((ay - by[i]) ** 2 + (ax - bx[i]) ** 2))
    n = int(max(abs(ay[j] - by[i]), abs(ax[j] - bx[i]))) + 1
    ys = np.rint(np.linspace(ay[j], by[i], n)).astype(int)
    xs = np.rint(np.linspace(ax[j], bx[i], n)).astype(int)
    between = lab[ys[1:-1], xs[1:-1]]
    return bool(between.size) and np.mean(between == 0) < 0.35


def merge_tracks(tracks: list[Track], labels: np.ndarray) -> list[Track]:
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
                if _is_continuation(a, b) or _is_duplicate(a, b) or _is_fragment(b, a, labels):
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


def _is_duplicate(a: Track, b: Track) -> bool:
    """Two tracks that sit on the same spot whenever both exist are one object split by noise."""
    common = sorted(set(a.frames) & set(b.frames))
    if not common or len(common) > 0.5 * min(len(a.frames), len(b.frames)):
        return False
    radius = 0.5 * max(a.radius(), b.radius())
    return all(np.linalg.norm(a.centroid(t) - b.centroid(t)) < radius for t in common)


def _is_fragment(small: Track, big: Track, labels: np.ndarray) -> bool:
    common = set(small.frames) & set(big.frames)
    if not common or _size(small) > 0.8 * _size(big) or len(common) < 0.6 * len(small.frames):
        return False
    radius = 1.2 * big.radius()
    if not all(np.linalg.norm(small.centroid(t) - big.centroid(t)) < radius for t in common):
        return False
    checks = sorted(common)[:: max(1, len(common) // 5)]
    return np.mean([_split_by_occluder(big.frames[t], small.frames[t], labels[t]) for t in checks]) >= 0.5


# 4. fit ---------------------------------------------------------------------------------

@dataclass
class FittedObject:
    colour_index: int
    colour: tuple[int, int, int]
    ref_frame: int
    points: np.ndarray            # template pixel centres, relative to the reference centroid
    params: dict[int, np.ndarray]  # frame -> (x, y, rot_deg, scale)
    gradient: tuple | None = None  # (p0, p1, colour0, colour1) in the local frame


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
    if _round(points):
        # A disc or ring has no visible orientation: keep its rotation constant.
        for t in params:
            params[t][2] = 0.0
    colour = tuple(int(c) for c in colours[k - 1])
    return FittedObject(k, colour, ref, points, params)


def _round(points: np.ndarray) -> bool:
    """True when the template looks the same after any rotation (disc, ring)."""
    lo = np.floor(points.min(0)).astype(int) - 2
    size = np.ceil(points.max(0)).astype(int) - lo + 3
    grid = np.zeros((size[1], size[0]), bool)
    ij = np.floor(points - lo).astype(int)
    grid[ij[:, 1], ij[:, 0]] = True
    for angle in (37.0, 61.0):
        c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        q = np.floor(points @ np.array([[c, s], [-s, c]]) - lo).astype(int)
        inside = (q[:, 0] >= 0) & (q[:, 0] < size[0]) & (q[:, 1] >= 0) & (q[:, 1] < size[1])
        overlap = grid[q[inside, 1], q[inside, 0]].mean() if inside.any() else 0.0
        if overlap < 0.93:
            return False
    return True


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

def vectorise(obj: FittedObject) -> tuple[np.ndarray, list | None]:
    """Outline polygon and hole polygons of the template, in the object's local frame."""
    up = 4
    pts = obj.points
    lo = np.floor(pts.min(0)) - 2
    size = (np.ceil(pts.max(0) - lo) + 3).astype(int)
    mask = np.zeros((size[1] * up, size[0] * up), np.uint8)
    # Each template point is a pixel centre: paint the square [p - 0.5, p + 0.5).
    for x, y in (pts - lo - 0.5):
        mask[int(round(y * up)):int(round(y * up)) + up, int(round(x * up)):int(round(x * up)) + up] = 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((up + 1, up + 1), np.uint8))
    # Smooth the pixel staircase before tracing, so curves come out as curves.
    mask = (cv2.GaussianBlur(mask, (0, 0), up * 0.9) > 127).astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((3, 2)), None
    outer_i = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
    outer_area = cv2.contourArea(contours[outer_i])
    hole_list = [_simplify(contours[i], up, lo) for i in range(len(contours))
                 if hierarchy[0][i][3] == outer_i and cv2.contourArea(contours[i]) > max(0.004 * outer_area, 2 * up * up)]
    return _simplify(contours[outer_i], up, lo), (hole_list or None)


def _simplify(contour, up, lo, grow=0.0):
    approx = cv2.approxPolyDP(contour, 0.3 * up, True).reshape(-1, 2).astype(float)
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
        full = np.full((n_frames, 4), np.nan)
        span = np.arange(frames[0], frames[-1] + 1)
        for i in range(4):
            full[span, i] = np.interp(span, frames, track[:, i])
        shapes.append(Shape("polygon", obj.colour, outer, hole, full, gradient=obj.gradient))
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
            if not np.isnan(s.track[t, 0]):
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



ANALYSIS_SIZE = 320  # thresholds are tuned in pixels at this scale


def _downscale(frames: np.ndarray) -> tuple[np.ndarray, float]:
    """Analyse large GIFs at ANALYSIS_SIZE (longest side): faster, and the pixel thresholds hold."""
    h, w = frames.shape[1:3]
    factor = min(1.0, ANALYSIS_SIZE / max(h, w))
    if factor >= 1.0:
        return frames, 1.0
    size = (max(1, round(w * factor)), max(1, round(h * factor)))
    small = np.stack([cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in frames])
    return small, factor


def _upscale(shapes: list[Shape], factor: float) -> list[Shape]:
    """Bring geometry measured on the reduced frames back to the input's pixel size."""
    if factor == 1.0:
        return shapes
    k = 1.0 / factor
    for s in shapes:
        s.outer = s.outer * k
        if s.hole is not None:
            s.hole = [h * k for h in s.hole] if isinstance(s.hole, list) else s.hole * k
        s.track = s.track.copy()
        s.track[:, :2] *= k
        if s.kind == "stroke":
            s.width *= k
        if s.gradient is not None:
            p0, p1, c0, c1 = s.gradient
            s.gradient = (p0 * k, p1 * k, c0, c1)
    return shapes


def reconstruct(path: str) -> dict:
    """Reconstruct a GIF. Shapes are in the input's pixel coordinates; `labels` are at analysis scale."""
    frames = load_frames(path)
    small, factor = _downscale(frames)
    from .gradients import detect_gradients, fit_gradient
    background, _ = find_palette(small)
    groups, group_colours, soft = detect_gradients(small, background)
    in_gradient = np.stack([cv2.dilate((g > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0 for g in groups])
    background, colours = find_palette(small, exclude=in_gradient if group_colours.size else None)
    labels = label_frames(small, background, colours)
    n_flat = len(colours)
    if group_colours.size:
        # Each gradient group is one more "colour" for the tracker.
        labels[groups > 0] = n_flat + groups[groups > 0]
        colours = np.vstack([colours.reshape(-1, 3), group_colours])
    from .strokes import detect_strokes
    strokes, consumed = detect_strokes(labels, colours)
    rigid_labels = labels.copy()
    rigid_labels[consumed] = 0
    tracks = track_objects(rigid_labels)
    objects = [fit_track(tr, rigid_labels, colours) for tr in tracks]
    for tr, obj in zip(tracks, objects):
        if obj.colour_index > n_flat:
            obj.gradient = fit_gradient(obj, tr.frames, soft, _round(obj.points))
    shapes = _upscale(to_shapes(objects, rigid_labels, len(frames)) + strokes, factor)
    return {"frames": frames, "background": tuple(int(c) for c in background), "colours": colours,
            "labels": labels, "objects": objects, "shapes": shapes, "analysis_scale": factor}
