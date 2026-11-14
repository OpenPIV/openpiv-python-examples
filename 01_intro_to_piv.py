import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    mo.md(r"""
# 01 — Introduction to PIV

**Particle Image Velocimetry (PIV)** is an optical, non-intrusive technique to measure velocity fields in fluids (and other moving media).

## Core idea
1. Seed the flow with small tracer particles that follow the motion.
2. Illuminate a plane/slice and take two images separated by a known time $\Delta t$.
3. Divide images into small **interrogation windows** and find the displacement that best matches each window between frames (usually by cross-correlation).
4. Velocity = displacement / $\Delta t$.

## What PIV assumes
- Particles follow the flow (good tracer behavior).
- Motion within a window is approximately uniform during $\Delta t$.
- Illumination and particle appearance are consistent between frames.
- Out-of-plane motion is limited (for planar PIV).

## Why it matters
PIV gives **spatial fields**, not just point measurements — you can see vortices, shear layers, wakes, jets, etc.
""")
    return (mo,)


@app.cell
def __(mo):
    mo.md("## Visual intuition with a synthetic example")
    return ()


@app.cell
def __():
    import numpy as np
    from openpiv.tools import synimage
    img_a, img_b = synimage.synimagegen(128, dt=0.1)
    return img_a, img_b, np


@app.cell
def __(img_a, img_b):
    import matplotlib.pyplot as plt1
    fig1, ax1 = plt1.subplots(1, 2, figsize=(8, 4))
    ax1[0].imshow(img_a, cmap="gray"); ax1[0].set_title("Frame A (t)")
    ax1[1].imshow(img_b, cmap="gray"); ax1[1].set_title("Frame B (t+Δt)")
    fig1
    return (fig1,)


@app.cell
def __(mo):
    mo.md(r"""
## Key parameters (preview)
- **Window size**: tradeoff between spatial resolution (small) and correlation robustness (large).
- **Overlap**: more vectors, but neighboring vectors share information.
- **$\Delta t$**: must produce measurable displacement (often ~5–10 pixels) without losing particles.
- **Search area**: must be large enough to capture the true displacement.

Next: [02 — Basic PIV workflow](02_basic_piv.html)
""")
    return ()


if __name__ == "__main__":
    app.run()
