import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    mo.md(r"""
# 02 — Basic PIV workflow

This lesson shows the minimal OpenPIV pipeline: load/generate images → cross-correlate → interpret vectors.

## Steps
1. Prepare image pair (A, B).
2. Choose **window size** (e.g. 32×32), **overlap** (e.g. 16 px = 50%), **search area**.
3. Run cross-correlation to estimate displacement $(u,v)$ in pixels.
4. Convert to physical units with $\Delta t$ and scaling if needed.
5. Validate and visualize.
""")
    return (mo,)


@app.cell
def __():
    import numpy as np
    from openpiv.tools import synimage
    img_a, img_b = synimage.synimagegen(128, dt=0.1)
    return img_a, img_b, np


@app.cell
def __(img_a, img_b):
    from openpiv import pyprocess
    u, v, s2n = pyprocess.extended_search_area_piv(
        img_a.astype(np.int32), img_b.astype(np.int32),
        window_size=32, overlap=16, dt=0.1, search_area_size=64,
        sig2noise_method='peak2peak'
    )
    print("u shape:", u.shape, "finite vectors:", np.isfinite(u).sum())
    return s2n, u, v


@app.cell
def __(u, v):
    import matplotlib.pyplot as plt1
    fig1, ax1 = plt1.subplots()
    ax1.quiver(u, v)
    ax1.invert_yaxis()
    ax1.set_title("Velocity vectors (image coordinates)")
    fig1
    return (fig1,)


@app.cell
def __(s2n):
    import matplotlib.pyplot as plt2
    fig2, ax2 = plt2.subplots()
    im = ax2.imshow(s2n, cmap='viridis')
    ax2.set_title('Signal-to-noise ratio (peak2peak)')
    fig2.colorbar(im, ax=ax2)
    fig2
    return (fig2,)


@app.cell
def __(mo):
    mo.md(r"""
## How to read this
- **u, v**: displacement/velocity components per interrogation window.
- **s2n**: confidence indicator — higher usually means a clearer correlation peak.
- **Inverted y-axis**: image origin is often top-left; plotting conventions matter.

## Practical tips
- If vectors look random: try larger window or check $\Delta t$/seeding.
- If field is too coarse: reduce window size or use multi-pass (next lesson).
- Always inspect s2n before trusting results.

Next: [03 — Multi-pass & window deformation](03_multipass_and_deformation.html)
""")
    return ()


if __name__ == "__main__":
    app.run()
