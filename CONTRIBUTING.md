# Contributing to Unbake

Thanks for your interest! The most valuable contribution right now is a **GIF that Unbake gets wrong**: open an issue with the "A GIF Unbake gets wrong" template, or reply in the [welcome discussion](https://github.com/FredBourouliou/unbake/discussions/1).

## Setup

```bash
git clone https://github.com/FredBourouliou/unbake && cd unbake
uv sync --dev
uv run pytest                      # about 10 s
uv run unbake web/samples/bounce.gif /tmp/bounce
```

Web editor: `./scripts/build_web.sh && python3 -m http.server -d web 8000`, then open http://localhost:8000.

## Code map

| Path | Role |
|---|---|
| `src/unbake/reconstruct.py` | palette, tracking, template fitting, vectorisation |
| `src/unbake/strokes.py` | animated strokes (trim paths) |
| `src/unbake/export.py` | Lottie and animated SVG export |
| `src/unbake/corpus.py`, `metrics.py` | synthetic benchmark with ground truth |
| `web/` | browser editor (Pyodide worker + canvas UI) |

## Changing the reconstruction

Reconstruction changes are judged on numbers, not on a single example:

1. Run the benchmark before and after: `uv run python -m unbake.corpus data`, then `uv run python scripts/evaluate.py data/dev` and `uv run python scripts/evaluate_strokes.py data/dev`.
2. Only tune on `data/dev`. Holdout sets are for final checks, run once.
3. For a new capability, write the success criterion in the README **before** the code, as was done for v0.1 and v0.2.

Small, focused pull requests are easiest to review. New behaviour should come with a test in `tests/`.
