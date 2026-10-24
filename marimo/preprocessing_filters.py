# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "openpiv>=0.26.0",
#     "numpy",
#     "matplotlib",
#     "imageio",
#     "scipy",
#     "skimage",
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
    ### Here are some Particle Image Velocimetry (PIV) pre-processing functions that can be used to enhance image quality.
    #### Written by Erich Zimmer
    #### Created at 20210813, 0108 CTZ
    """)
    return


@app.cell
def _():
    import openpiv.tools as piv_tls
    from matplotlib import pyplot as plt
    from openpiv import preprocess as piv_pre
    from skimage import exposure

    return exposure, piv_pre, piv_tls, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Spatial filters
    """)
    return


@app.cell
def _():
    i1 = "test12/A001a.tif"
    _i2 = "test12/A001b.tif"
    _i3 = "test12/A002a.tif"
    _i4 = "test12/A002b.tif"
    img_list = [i1, _i2, _i3, _i4]
    return (i1,)


@app.cell
def _(exposure, i1, piv_pre, piv_tls, plt):
    # its a good idea to normalize arrays to [0, 1] float, but not needed
    _img = piv_pre.normalize_array(piv_tls.imread(i1))
    img_CLAHE = exposure.equalize_adapthist(
        _img.copy(), kernel_size=None, clip_limit=0.01, nbins=256
    )
    img_CLAHE = piv_pre.instensity_cap(img_CLAHE, 4)
    img_str = piv_pre.contrast_stretch(_img.copy(), 2, 99)
    img_str = piv_pre.instensity_cap(img_str, 2)
    _img_norm = piv_pre.local_variance_normalization(_img.copy(), 1.75, 1.5, clip=True)
    _fig, _ax = plt.subplots(2, 2, figsize=(13, 13))
    _ax[0, 0].imshow(_img * 255, cmap=plt.cm.gray, vmax=255)
    _ax[0, 1].imshow(img_CLAHE * 255, cmap=plt.cm.gray, vmax=255)
    _ax[1, 0].imshow(img_str * 255, cmap=plt.cm.gray, vmax=255)
    _ax[1, 1].imshow(_img_norm * 255, cmap=plt.cm.gray, vmax=255)
    _ax[0, 0].set_title("Original")
    _ax[0, 1].set_title("CLAHE and cap")
    _ax[1, 0].set_title("Contrast stretch and cap")
    _ax[1, 1].set_title("Local variance normalization")
    return


@app.cell
def _(i1, piv_pre, piv_tls, plt):
    # some additional filters
    _img = piv_pre.normalize_array(piv_tls.imread(i1))
    img_hp = piv_pre.high_pass(_img.copy(), 5, clip=True)
    img_cp = piv_pre.intensity_clip(_img.copy(), 0.1, 0.99, "clip")
    _img_norm = piv_pre.local_variance_normalization(_img.copy(), 2, 1.5, clip=True)
    img_bin = piv_pre.threshold_binarize(_img_norm, 0.1, 1)
    _fig, _ax = plt.subplots(
        2, 2, figsize=(13, 13)
    )  # this doesn't work properly anymore?
    _ax[0, 0].imshow(_img * 255, cmap=plt.cm.gray, vmax=255)
    _ax[0, 1].imshow(img_hp * 255, cmap=plt.cm.gray, vmax=255)
    _ax[1, 0].imshow(img_cp * 255, cmap=plt.cm.gray, vmax=255)
    _ax[1, 1].imshow(img_bin * 255, cmap=plt.cm.gray, vmax=255)
    _ax[0, 0].set_title("Original")
    _ax[0, 1].set_title("gaussian high pass")
    _ax[1, 0].set_title("intensity clip")
    _ax[1, 1].set_title("local variance norm with threshold binarizing")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Temporal filters for background subtraction
    Currently two filters are implemented: temporal averaging and temporal low pass. Apply by subtracting the image of interest with the generated background image.
    """)
    return


@app.cell
def _():
    i1_1 = "test13/00020359.bmp"
    _i2 = "test13/00020360.bmp"
    _i3 = "test13/00020361.bmp"
    _i4 = "test13/00020362.bmp"
    img_list_1 = [i1_1, _i2, _i3, _i4]
    return i1_1, img_list_1


@app.cell
def _(img_list_1, piv_pre):
    # temporal filtering
    background_min = piv_pre.gen_min_background(img_list_1, resize=255)
    # works better than min of images for low amount of images, produces less invalid vectors
    # normalizing can be disabled by setting »resize« to None
    background_low = piv_pre.gen_lowpass_background(img_list_1, resize=255)
    return background_low, background_min


@app.cell
def _(background_low, background_min, i1_1, piv_pre, piv_tls, plt):
    _fig, _ax = plt.subplots(2, 2, figsize=(13, 13))
    _img = piv_pre.normalize_array(piv_tls.imread(i1_1)) * 255
    new_img = _img - background_low
    new_img[new_img < 0] = 0
    _ax[0, 0].imshow(_img, cmap=plt.cm.gray, vmax=255)
    _ax[0, 1].imshow(background_min, cmap=plt.cm.gray, vmax=255)
    _ax[1, 1].imshow(background_low, cmap=plt.cm.gray, vmax=255)
    _ax[1, 0].imshow(new_img, cmap=plt.cm.gray, vmax=255)
    _ax[0, 0].set_title("Original")
    _ax[0, 1].set_title("Min background")
    _ax[1, 1].set_title("Lowpass background")
    _ax[1, 0].set_title("Temporal high pass")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Transformations
    Offsetting one image can be useful when frame straddling two offset cameras with a similar field of view.
    """)
    return


@app.cell
def _(i1_1, piv_pre, piv_tls, plt):
    _img = piv_pre.normalize_array(piv_tls.imread(i1_1)) * 255
    _fig, _ax = plt.subplots(2, 2, figsize=(13, 13))
    _ax[0, 0].imshow(_img, cmap=plt.cm.gray, vmax=255)
    _ax[0, 1].imshow(piv_pre.stretch_image(_img, 0, 1), cmap=plt.cm.gray, vmax=255)
    _ax[1, 0].imshow(
        piv_pre.offset_image(_img, 50, 0, pad="reflect"), cmap=plt.cm.gray, vmax=255
    )
    _ax[1, 1].imshow(piv_pre.offset_image(_img, -50, 0), cmap=plt.cm.gray, vmax=255)
    _ax[0, 0].set_title("Original")
    _ax[0, 1].set_title("stretched")
    _ax[1, 1].set_title("padded negative, pad = zeros")
    _ax[1, 0].set_title("padded positive, pad = reflection")
    return


if __name__ == "__main__":
    app.run()
