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
    mo.md(r"""
# Test case: test8

This page demonstrates the **test8** dataset/case from the OpenPIV examples repository.

## What this case is for
- Explore the images and data included in `test8/`.
- Learn how to run PIV on this case and what to look for (seeding, displacement, noise, masks, etc.).
- Use it as a starting point: adjust window size, overlap, dt, and validation for your needs.

**[Open in molab (interactive)](https://molab.marimo.io/github/openpiv/openpiv-python-examples/blob/main/marimo_cases/test8.py)**

**Files (sample):** frame0001.tif, frame0002.tif, frame0003.tif, frame0004.tif
""")
    return (mo,)


@app.cell
def __():
    import os, glob
    files = sorted(glob.glob("test8/*"))
    print("Files in case:")
    for f in files[:30]:
        print(" -", os.path.basename(f))
    return (files,)


@app.cell
def __(mo):
    mo.md("## Load and display images")
    return ()


@app.cell
def __():
    import matplotlib.pyplot as plt1
    from openpiv import tools
    import os
    a = "frame0001.tif"
    b = "frame0002.tif"
    img_a = tools.imread(os.path.join("test8", a)) if a else None
    img_b = tools.imread(os.path.join("test8", b)) if b else None
    if img_a is not None and img_b is not None:
        fig1, ax1 = plt1.subplots(1,2,figsize=(8,4))
        ax1[0].imshow(img_a, cmap='gray'); ax1[0].set_title(a)
        ax1[1].imshow(img_b, cmap='gray'); ax1[1].set_title(b)
        out = fig1
    else:
        out = "No image pair found to display."
    out
    return (img_a, img_b)


@app.cell
def __(mo):
    mo.md(r"""
## Run PIV (basic example)
Adjust parameters based on this case: window size, overlap, search area, and `dt`.
- **Tip:** ensure particle displacement is ~5–10 px between frames.
- **Tip:** inspect signal-to-noise and validate outliers.
""")
    return ()


@app.cell
def __(img_a, img_b):
    import numpy as np
    from openpiv import pyprocess
    if img_a is not None and img_b is not None:
        try:
            u, v, s2n = pyprocess.extended_search_area_piv(
                img_a.astype(np.int32), img_b.astype(np.int32),
                window_size=32, overlap=16, dt=1.0, search_area_size=64,
                sig2noise_method='peak2peak'
            )
            print('PIV computed. shape:', u.shape, 'finite:', int(np.isfinite(u).sum()))
            res = (u,v,s2n)
        except Exception as e:
            print('PIV failed for default settings:', e)
            print('Try different window_size/overlap/search_area for this case.')
            res = (None,None,None)
    else:
        res = (None,None,None)
    res
    return res


@app.cell
def __(res):
    import matplotlib.pyplot as plt2
    u,v,s2n = res
    if u is not None:
        fig2, ax2 = plt2.subplots()
        ax2.quiver(u, v)
        ax2.invert_yaxis()
        ax2.set_title('Velocity field (test8)')
        out2 = fig2
    else:
        out2 = 'No vector field to plot.'
    out2
    return ()


@app.cell
def __(mo):
    mo.md(r"""
## Notes for this case
- Inspect images for seeding density, illumination, reflections, and masks.
- Choose window size so each window contains enough particle pairs.
- Validate vectors (s2n, median) and consider multi-pass/window deformation for gradients.
- Compare with provided reference outputs in this folder if available (e.g. `.txt` results).

Back to [all cases](all_cases.html) · [index](index.html)
""")
    return ()


if __name__ == "__main__":
    app.run()
