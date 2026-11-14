import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    mo.md(r"""
# 06 — Tips, tricks & pitfalls

A practical checklist for reliable PIV with OpenPIV.

## Experiment & images
- **Seeding density**: ~5–10 particle pairs per interrogation window is a common target.
- **Particle size**: ~2–3 pixels is often ideal for subpixel peak fitting.
- **Displacement**: aim for ~1/4 of window size (rule of thumb) between frames.
- **Avoid** saturation, motion blur, strong reflections, and large out-of-plane loss.

## Processing choices
- **Window size**: large = robust but low resolution; small = detailed but noisy.
- **Overlap**: 50% is a good default; 75% gives smoother-looking fields but correlated vectors.
- **Multi-pass**: use for gradients; validate between passes.
- **Search area**: must cover maximum expected displacement.

## Validation & post-processing
- Validate (s2n, median, range) before smoothing.
- Replace/interpolate outliers cautiously; report how many were invalid.
- Smooth for visualization if needed, but analyze carefully (smoothing can bias derivatives).

## Common pitfalls
| Symptom | Likely cause | Try |
|---|---|---|
| Random/noisy vectors | Window too small, low seeding, wrong $\Delta t$ | Larger window, better seeding, adjust $\Delta t$ |
| Systematic bias | Peak-locking, poor subpixel fit, gradients | Better particle size, deformation, multi-pass |
| Missing vectors near objects | No mask / bad mask | Static/dynamic masking |
| Over-smoothed field | Aggressive smoothing | Reduce smoothing, validate first |

## Reproducibility
- Save settings (e.g. `PIVSettings` to YAML), record versions, and keep raw outputs.
""")
    return (mo,)


@app.cell
def __(mo):
    mo.md("## Back to [index](index.html)")
    return ()


if __name__ == "__main__":
    app.run()
