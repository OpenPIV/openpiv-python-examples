# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "openpiv==0.26.3",
#     "numpy",
#     "matplotlib",
#     "scipy",
#     "scikit-image",
# ]
# ///

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    mo.md("# Introduction to PIV (Particle Image Velocimetry)\n\nThis notebook introduces PIV concepts and OpenPIV usage.")
    return (mo,)


@app.cell
def __(mo):
    mo.md("## What is PIV?\nPIV measures velocity fields by tracking particles in two images.")
    return ()


@app.cell
def __():
    import numpy as np
    from openpiv import tools, pyprocess
    from openpiv.tools import synimage
    img_a, img_b = synimage.synimagegen(128)
    u, v, s2n = pyprocess.extended_search_area_piv(img_a.astype(int), img_b.astype(int), window_size=32, overlap=16, dt=0.1, search_area_size=64)
    print("PIV done", u.shape)
    return ()


if __name__ == "__main__":
    app.run()
