import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    mo.md("# Educational PIV with OpenPIV\n\nThis notebook teaches what PIV is and how to use OpenPIV properly.")
    return (mo,)


@app.cell
def __(mo):
    mo.md("## 1. What is PIV?\nParticle Image Velocimetry estimates velocity fields from pairs of images by correlating small windows.")
    return ()


@app.cell
def __(mo):
    mo.md("## 2. Generate synthetic images\nWe use OpenPIV's synthetic image generator to create a known flow.")
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
    mo.md("## 3. Run PIV\nWe use extended_search_area_piv with a 32x32 window.")
    return ()


@app.cell
def __(img_a, img_b):
    from openpiv import pyprocess
    u, v, s2n = pyprocess.extended_search_area_piv(
        img_a.astype(int), img_b.astype(int), window_size=32, overlap=16, dt=0.1, search_area_size=64
    )
    print("vectors", u.shape)
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
    mo.md("## 4. Why masking and smoothing?\nMasking removes objects; smoothing reduces noise. Use dynamic_masking and smoothn.")
    return ()


if __name__ == "__main__":
    app.run()
