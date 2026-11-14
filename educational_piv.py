import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    mo.md("# Educational PIV with OpenPIV\n\nThis site teaches what PIV is, how to run it, and practical tips for good results.")
    return (mo,)


@app.cell
def __(mo):
    mo.md("## 1. What is PIV?\nParticle Image Velocimetry estimates 2D velocity fields by cross-correlating small interrogation windows between two images. Key ideas:\n- Window size sets spatial resolution and accuracy.\n- Overlap increases vector density but not independent information.\n- Multi-pass with deformation improves accuracy on strong gradients.")
    return ()


@app.cell
def __(mo):
    mo.md("## 2. Generate synthetic images\nWe use OpenPIV's synthetic image generator to create a known flow for validation.")
    return ()


@app.cell
def __():
    import numpy as np
    from openpiv.tools import synimage
    img_a, img_b = synimage.synimagegen(128)
    print("images", img_a.shape, img_b.shape)
    return (img_a, img_b)


@app.cell
def __(img_a, img_b):
    import matplotlib.pyplot as plt1
    fig1, ax1 = plt1.subplots(1, 2, figsize=(8, 4))
    ax1[0].imshow(img_a, cmap="gray")
    ax1[0].set_title("Image A")
    ax1[1].imshow(img_b, cmap="gray")
    ax1[1].set_title("Image B")
    return (fig1,)


@app.cell
def __(mo):
    mo.md("## 3. Run PIV\nTip: start with window_size=32, overlap=50%, 2-3 passes. Use dt to convert to physical units.")
    return ()


@app.cell
def __(img_a, img_b):
    from openpiv import pyprocess
    u, v, s2n = pyprocess.extended_search_area_piv(
        img_a.astype(int), img_b.astype(int), window_size=32, overlap=16, dt=0.1, search_area_size=64
    )
    print("vectors", u.shape, "finite:", (u == u).sum())
    return (u, v, s2n)


@app.cell
def __(u, v):
    import matplotlib.pyplot as plt2
    fig2, ax2 = plt2.subplots()
    ax2.quiver(u, v)
    ax2.set_title("Velocity field")
    return (fig2,)


@app.cell
def __(mo):
    mo.md("## 4. Tips & Tricks\n- Use dynamic masking for objects in the flow.\n- Apply validation (median, std) to remove outliers.\n- Smooth with smoothn for presentation, not for validation.\n- Check sig2noise; low values indicate poor correlation.\n- Use Rust backend for speed on large datasets.")
    return ()


@app.cell
def __(mo):
    mo.md("## 5. Common pitfalls\n- Too small windows → noise; too large → lose gradients.\n- Insufficient overlap → coarse field.\n- Not validating → false vectors.\n- Ignoring dt → wrong units.")
    return ()


if __name__ == "__main__":
    app.run()
