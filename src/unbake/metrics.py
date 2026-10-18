"""Score a reconstruction against the ground truth JSON written by the corpus generator."""
from __future__ import annotations

import json
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from .reconstruct import label_frames
from .scene import SYMMETRY, render

ID_RADIUS = 6.0


def _wrap(a: np.ndarray, period: float) -> np.ndarray:
    return (a + period / 2) % period - period / 2


def score(rec: dict, truth_path: str) -> dict:
    truth = json.loads(open(truth_path).read())
    gts = truth["shapes"]
    n_frames = truth["n_frames"]
    shapes = rec["shapes"]

    gt_xy = np.array([np.array(g["track"])[:, :2] for g in gts])          # (G, T, 2)
    est_xy = np.array([s.track[:, :2] for s in shapes]) if shapes else np.zeros((0, n_frames, 2))

    # Match objects on mean centroid distance over the animation.
    matched: dict[int, int] = {}
    if len(shapes):
        cost = np.nanmean(np.linalg.norm(gt_xy[:, None] - est_xy[None], axis=-1), axis=-1)
        for g, e in zip(*linear_sum_assignment(cost)):
            if cost[g, e] < 25:
                matched[g] = e

    identity_ok = 0
    pos_err, rot_err, scale_err = [], [], []
    obj_by_shape = {id(s): o for s, o in zip(rec["shapes"], _objects_in_shape_order(rec))}
    for g, gt in enumerate(gts):
        track = np.array(gt["track"])
        e = matched.get(g)
        for t in range(n_frames):
            if e is None:
                continue
            d = np.nan_to_num(np.linalg.norm(est_xy[:, t] - track[t, :2], axis=1), nan=np.inf)
            if d.argmin() == e and d[e] < ID_RADIUS:
                identity_ok += 1
        if e is None:
            continue
        obj = obj_by_shape[id(shapes[e])]
        observed = sorted(obj.params)
        ref = obj.ref_frame
        sym = SYMMETRY[gt["kind"]]
        for t in observed:
            p = obj.params[t]
            pos_err.append(float(np.linalg.norm(p[:2] - track[t, :2])))
            if sym:
                true_rot = track[t, 2] - track[ref, 2]
                rot_err.append(abs(float(_wrap(np.array(p[2] - true_rot), 360 / sym))))
            scale_err.append(abs(p[3] - track[t, 3] / track[ref, 3]) / (track[t, 3] / track[ref, 3]))

    # Render IoU per colour region: input frames vs re-rendered reconstruction.
    rerender = np.stack(render(shapes, n_frames, tuple(truth["size"]), rec["background"]))
    bg = np.array(rec["background"], float)
    lab_in = rec["labels"]
    lab_out = label_frames(rerender, bg, rec["colours"])
    ious = []
    for t in range(n_frames):
        for k in range(1, len(rec["colours"]) + 1):
            a, b = lab_in[t] == k, lab_out[t] == k
            union = np.count_nonzero(a | b)
            if union > 20:
                ious.append(np.count_nonzero(a & b) / union)

    total = len(gts) * n_frames
    return {
        "n_gt": len(gts), "n_est": len(shapes), "matched": len(matched),
        "identity": identity_ok / total,
        "pos_median": float(np.median(pos_err)) if pos_err else math.inf,
        "rot_median": float(np.median(rot_err)) if rot_err else 0.0,
        "scale_median": float(np.median(scale_err)) if scale_err else math.inf,
        "iou": float(np.mean(ious)) if ious else 0.0,
        "rerender": rerender,
    }


def _objects_in_shape_order(rec):
    # to_shapes reorders shapes by depth; recover the object of each shape by colour + track.
    out = []
    for s in rec["shapes"]:
        best = min(rec["objects"], key=lambda o: (o.colour != s.color,
                                                 np.nan_to_num(np.linalg.norm(o.params[o.ref_frame][:2] - s.track[o.ref_frame, :2]), nan=1e9)))
        out.append(best)
    return out
