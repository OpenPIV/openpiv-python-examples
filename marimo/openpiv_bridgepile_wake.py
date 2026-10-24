# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "openpiv>=0.26.0",
#     "numpy",
#     "matplotlib",
#     "imageio",
#     "scipy",
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
    # OpenPIV on the bridgepile_wake

    See the post on LinkedIn by Stefano Brizzolara
    https://www.linkedin.com/posts/stefano-brizzolara-6a8501198_rheinfall-flowvisualization-ugcPost-6672832128742408192-lRub
    """)
    return


@app.cell
def _():
    # '%matplotlib inline' command supported automatically in marimo
    import matplotlib.pyplot as plt
    import numpy as np
    from openpiv import filters, pyprocess, tools, validation

    return filters, np, plt, pyprocess, tools, validation


@app.cell
def _(tools):
    frame_a  = tools.imread("data/test8/frame0001.tif")
    frame_b  = tools.imread("data/test8/frame0002.tif")
    return frame_a, frame_b


@app.cell
def _(frame_a, frame_b, plt):
    _fig, _ax = plt.subplots(1, 2, figsize=(12, 10))
    _ax[0].imshow(frame_a, cmap=plt.cm.gray)
    _ax[1].imshow(frame_b, cmap=plt.cm.gray)
    return


@app.cell
def _(frame_a, frame_b, np, pyprocess):
    # %pdb
    # np.seterr(all="raise")
    winsize = 24 # pixels
    searchsize = 48  # pixels, search in image B
    overlap = 12 # pixels
    dt = 1./30 # sec, assume 30 fps

    frame_a[:600,:] = 0  # basically masking out the non-illuminated region
    frame_b[:600,:] = 0


    u0, v0, sig2noise = pyprocess.extended_search_area_piv(frame_a.astype(np.int32), 
                                                           frame_b.astype(np.int32), 
                                                           window_size=winsize, 
                                                           overlap=overlap, dt=dt, 
                                                           search_area_size=searchsize, 
                                                           sig2noise_method='peak2peak',
                                                          correlation_method='linear',
                                                          normalized_correlation=True)
    return overlap, searchsize, sig2noise, u0, v0


@app.cell
def _(frame_a, overlap, pyprocess, searchsize):
    x, y = pyprocess.get_coordinates(frame_a.shape,searchsize, overlap)
    return x, y


@app.cell
def _(sig2noise, validation):
    mask = validation.sig2noise_val(sig2noise, threshold = 1.2)
    return (mask,)


@app.cell
def _(filters, mask, u0, v0):
    u2, v2 = filters.replace_outliers( u0, v0, mask, method='localmean', max_iter=1, kernel_size=3)
    return u2, v2


@app.cell
def _():
    # x, y, u3, v3 = scaling.uniform(x, y, u2, v2, scaling_factor = 1. )
    return


@app.cell
def _(mask, tools, u2, v2, x, y):
    x_1, y_1, u2_1, v2_1 = tools.transform_coordinates(x, y, u2, v2)
    tools.save('exp1_001.txt', x_1, y_1, u2_1, v2_1, mask)
    return


@app.cell
def _():
    # tools.display_vector_field('exp1_001.txt', scaling_factor=100., width=0.0025)
    return


@app.cell
def _(plt, tools):
    # If you need a larger view:
    _fig, _ax = plt.subplots(figsize=(12, 12))
    tools.display_vector_field('exp1_001.txt', ax=_ax, scaling_factor=1.0, scale=1000, width=0.0045, on_img=True, image_name="data/test8/frame0001.tif")
    return


@app.cell
def _():
    from openpiv.windef import PIVSettings, piv

    return PIVSettings, piv


@app.cell
def _(PIVSettings, piv):
    settings = PIVSettings()
    settings.filepath_images = "data/test8"
    settings.frame_pattern_a = 'frame00*.tif'
    settings.frame_pattern_b = '(1+2),(2+3)'
    settings.show_plot=True
    piv(settings)
    return


if __name__ == "__main__":
    app.run()
