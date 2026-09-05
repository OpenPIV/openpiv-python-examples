# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "openpiv>=0.26.0",
#     "numpy",
#     "matplotlib",
#     "imageio",
# ]
# ///

import marimo

__generated_with = "0.23.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)

@app.cell
def _():
    import importlib.metadata
    print("openpiv", importlib.metadata.version("openpiv"))
    try:
        import openpiv_rust
        print("openpiv-rust available — Rust backend enabled")
    except ImportError:
        print("openpiv-rust not installed — pip install openpiv[rust] for faster Rust backend")
    return

@app.cell
def _():
    import marimo as mo
    mo.md(r"""*
Requires `openpiv>=0.26.0`. New in 0.26.0: `scipy.fft` default backend (2-3x faster) and optional `openpiv-rust` via `backend="rust"`/`"auto"`.
*""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # OpenPIV static masking example%load_ext watermark
    # magic command not supported in marimo; please file an issue to add support
    # %watermark -v -m -p numpy,openpiv -g -b
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %load_ext watermark
    # magic command not supported in marimo; please file an issue to add support
    # %watermark -v -m -p numpy,openpiv -g -b
    return


@app.cell
def _():
    import pathlib

    import matplotlib.pyplot as plt
    import numpy as np
    from openpiv import tools
    from openpiv.piv import simple_piv

    return np, pathlib, plt, simple_piv, tools


@app.cell
def _(pathlib):
    images = sorted(pathlib.Path("test9").glob('*.jpg'))
    print(images)
    return (images,)


@app.cell
def _(images, np, plt, tools):
    a,b = tools.imread(images[0]), tools.imread(images[1])
    plt.imshow(np.c_[a,b])
    return a, b


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We see the mask as a circle, but OpenPIV does not know about it, we see some vectors "inside the masked region". Since the arrows inside are erroneous, these are removed in the post-processing validation.
    """)
    return


@app.cell
def _(a, b, simple_piv):
    simple_piv(a,b)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Let's create a mask - it's an image size matrix of booleans
    """)
    return


@app.cell
def _(a, np, plt):
    from skimage.morphology import binary_dilation, disk, white_tophat


    # binary thresholding - all above 210 can be an object
    mask = a > 210

    # remove tracers:
    mask = np.logical_xor(mask, white_tophat(mask, disk(3))) # remove small objects

    # increase a bit the mask borders 
    mask = binary_dilation(mask, disk(7)) # dilate large object

    plt.imshow(mask)
    return (mask,)


@app.cell
def _(a, b, mask, np, plt):
    # long and descriptive way:
    masked_a = a.copy()
    masked_b = b.copy()
    masked_a[mask] = 0
    masked_b[mask] = 0
    plt.imshow(np.c_[masked_a, masked_b])
    return masked_a, masked_b


@app.cell
def _(masked_a, masked_b, simple_piv):
    simple_piv(masked_a, masked_b)
    return


@app.cell
def _(a, b, mask, simple_piv):
    # shorthand
    masked_a_1 = a | mask
    masked_b_1 = b | mask
    simple_piv(masked_a_1, masked_b_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use static masking with multi-pass window deformation processor
    """)
    return


@app.cell
def _(images, mask, pathlib):
    from openpiv import windef
    settings = windef.PIVSettings()
    settings.filepath_images = pathlib.Path(".")
    settings.frame_pattern_a = str(images[0])
    settings.frame_pattern_b = str(images[1])

    settings.static_mask  # was static_masking = True
    settings.static_mask = mask
    settings.show_all_plots = True
    settings.show_plot = True
    return settings, windef


@app.cell
def _(settings, windef):
    windef.piv(settings)
    return


if __name__ == "__main__":
    app.run()
