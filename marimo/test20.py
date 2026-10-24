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
    # OpenPIV tutorial copy for test20

    In this tutorial we read the pair of images using `imread`, compare them visually
    and process using OpenPIV. Here the import is using directly the basic functions and methods
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from openpiv import filters, pyprocess, scaling, tools, validation
    # '%matplotlib inline' command supported automatically in marimo


    return filters, np, plt, pyprocess, scaling, tools, validation


@app.cell
def _(tools):
    frame_a = tools.imread("data/test20/t_23.png")
    frame_b = tools.imread("data/test20/t_24.png")
    return frame_a, frame_b


@app.cell
def _(frame_a, frame_b, plt):
    _fig, _ax = plt.subplots(1, 2, figsize=(12, 10))
    _ax[0].imshow(frame_a, cmap=plt.cm.gray)
    _ax[1].imshow(frame_b, cmap=plt.cm.gray)
    return


@app.cell
def _(frame_a, frame_b, np, pyprocess):
    winsize = 32  # pixels, interrogation window size in frame A
    searchsize = 38  # pixels, search in image B
    overlap = 12  # pixels, 50% overlap
    dt = 0.02  # sec, time interval between pulses


    u0, v0, sig2noise = pyprocess.extended_search_area_piv(
        frame_a.astype(np.int32),
        frame_b.astype(np.int32),
        window_size=winsize,
        overlap=overlap,
        dt=dt,
        search_area_size=searchsize,
        sig2noise_method="peak2peak",
    )
    return overlap, searchsize, sig2noise, u0, v0


@app.cell
def _(frame_a, overlap, pyprocess, searchsize):
    x, y = pyprocess.get_coordinates(
        image_size=frame_a.shape, search_area_size=searchsize, overlap=overlap
    )
    return x, y


@app.cell
def _(sig2noise, validation):
    flags = validation.sig2noise_val(sig2noise, threshold=1.05)
    # if you need more detailed look, first create a histogram of sig2noise
    # plt.hist(sig2noise.flatten())
    # to see where is a reasonable limit
    return (flags,)


@app.cell
def _(filters, flags, u0, v0):
    # filter out outliers that are very different from the
    # neighbours

    u2, v2 = filters.replace_outliers(
        u0, v0, flags, method="localmean", max_iter=3, kernel_size=3
    )
    return u2, v2


@app.cell
def _(scaling, tools, u2, v2, x, y):
    # convert x,y to mm
    # convert u,v to mm/sec
    x_1, y_1, u3, v3 = scaling.uniform(x, y, u2, v2, scaling_factor=96.52)
    # 0,0 shall be bottom left, positive rotation rate is counterclockwise
    x_1, y_1, u3, v3 = tools.transform_coordinates(
        x_1, y_1, u3, v3
    )  # 96.52 microns/pixel
    return u3, v3, x_1, y_1


@app.cell
def _(flags, tools, u3, v3, x_1, y_1):
    # save in the simple ASCII table format
    tools.save("exp1_001.txt", x_1, y_1, u3, v3, flags)
    return


@app.cell
def _(plt, tools):
    _fig, _ax = plt.subplots(figsize=(8, 8))
    tools.display_vector_field(
        "exp1_001.txt",
        ax=_ax,
        scaling_factor=96.52,
        scale=50,
        width=0.0035,
        on_img=True,
        image_name="test1/exp1_001_a.bmp",
    )  # scale defines here the arrow length  # width is the thickness of the arrow  # overlay on the image
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## One could also use some shortcuts
    """)
    return


@app.cell
def _(frame_a, frame_b):
    from openpiv import piv

    piv.simple_piv(frame_a, frame_b)
    return


if __name__ == "__main__":
    app.run()
