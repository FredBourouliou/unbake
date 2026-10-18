# Unbake (feasibility spike)

Turn a flattened motion-graphics animation (GIF/MP4) back into editable objects: shapes, colours and per-object motion that can be changed and re-exported.

This repository is currently a **one-week feasibility spike**, not a product.

## The question

Can objects and their motion be reconstructed from rendered frames cleanly enough that editing them is useful?

## Method

1. `unbake.corpus` generates synthetic animations whose source is known (shapes, colours, per-frame position, rotation, scale), exported as palette GIFs. Three levels:
   - **easy**: 1–3 shapes, distinct colours, no overlap, no dithering
   - **medium**: 3–5 shapes, overlaps, Floyd–Steinberg dithering
   - **hard**: 4–6 shapes, overlaps, dithering, near-identical colours
2. The reconstruction only sees the GIF. The ground truth is used for scoring only.
3. Dev set (seeds 0…) is used while developing. **Holdout set (seeds 10 000…) is only run for the final numbers.**

## Stop criterion (written before any reconstruction code)

Measured on the **medium holdout** set, per ground-truth object and frame:

| Metric | Go if |
|---|---|
| Identity: object-frames attributed to the right object | ≥ 90 % |
| Median position error | ≤ 2 px |
| Median rotation error (modulo the shape's symmetry) | ≤ 5° |
| Mean IoU between input frames and the re-rendered reconstruction (per colour region) | ≥ 0.90 |

- All four met on medium → build the editor.
- Met on easy but not on medium → one more iteration focused on occlusions, then decide.
- Not met on easy → stop: the classical approach is not enough.

The hard set is reported but not part of the decision.

**Correction logged on day 1:** the first corpus could give two shapes the same colour on the medium level (a generator bug: identical/near colours are meant to be a hard-level feature only). Found while inspecting the dev set; fixed and both sets regenerated before the holdout set had ever been scored.

## Comparison

OmniLottie (CVPR 2026, 4B parameters, video → Lottie) is the reference learned approach; it will be run on a subset of the same holdout GIFs when feasible on local hardware.
