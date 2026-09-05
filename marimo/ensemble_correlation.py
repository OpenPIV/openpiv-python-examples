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
    # Ensemble correlation concept using OpenPIV
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Ensemble correlation should work in places where the flow is really steady and repeatable
    or could be phase averaged in the sense that the correlation map in a single interrogation
    window represents displacements from a statistically stationary distribution.

    In such case, the noisy position of the correlation peak is due to randomness that can
    be averaged out like the white noise and the avergaging of the correlation maps will
    yield a high quality peak that has great signal to noise ratio and close to Gaussian

    In this case the velocity estimate in the interrogation window will approach the mean
    velocity value at that location.
    """)
    return


@app.cell
def _():

    return


@app.cell
def _():
    from glob import glob

    return (glob,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.pyplot import (
        close,
        colorbar,
        contourf,
        figure,
        imshow,
        show,
        title,
        xlabel,
        ylabel,
    )

    return (
        np,
        plt,
        contourf,
        colorbar,
        figure,
        show,
        title,
        xlabel,
        ylabel,
        imshow,
        close,
    )


@app.cell
def _(glob):
    imlist = glob("test12/*.tif")
    imlist.sort()
    print(imlist)
    return (imlist,)


@app.cell
def _(imlist):
    # just a quick look at the data
    from openpiv.piv import simple_piv

    simple_piv(imlist[0], imlist[1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ensemble averaged correlation using FFT based correlation from OpenPIV
    """)
    return


@app.cell
def _(fft_correlate_images, imlist, imread, moving_window_array):
    corrs = []
    for i, j in zip(imlist[::2], imlist[1::2]):
        # print(i,j)
        corrs.append(
            fft_correlate_images(
                moving_window_array(imread(i), 64, 32),
                moving_window_array(imread(j), 64, 32),
                normalized_correlation=True,
            )
        )
    return (corrs,)


@app.cell
def _(corrs, np):
    corrs_1 = np.array(corrs)  # save also single image pair correlations
    mean_correlation = corrs_1.mean(axis=0)  # ensemble average
    return corrs_1, mean_correlation


@app.cell
def _(colorbar, contourf, mean_correlation):
    # Let's compare the result with instantaneous results
    contourf(mean_correlation[23, :, :])
    colorbar()
    return


@app.cell
def _(colorbar, contourf, corrs_1, figure):
    for i_1 in range(corrs_1.shape[0]):
        figure()
        contourf(corrs_1[i_1, 252, :, :])
        colorbar()
    return


@app.cell
def _(imlist, imread):
    im = imread(imlist[0])
    im.shape
    return (im,)


@app.cell
def _(get_field_shape, im):
    grid = get_field_shape(im.shape, search_area_size=64, overlap=32)
    nrows, ncols = grid[0], grid[1]
    return ncols, nrows


@app.cell
def _(correlation_to_displacement, mean_correlation, ncols, nrows):
    u, v = correlation_to_displacement(mean_correlation, nrows, ncols)
    return u, v


@app.cell
def _(get_coordinates, im):
    x, y = get_coordinates(im.shape, 64, 32)
    return x, y


@app.cell
def _(plot, subplots, u, v, x, y):
    fig, ax = subplots(figsize=(8, 8))
    ax.quiver(x, y, u, v, scale=80, width=0.003)
    ax.invert_yaxis()
    plot(u.mean(axis=1) * 80 + 400, y[:, 0])
    return


@app.cell
def _():

    return


@app.cell
def _(
    correlation_to_displacement,
    corrs_1,
    ncols,
    np,
    nrows,
    plot,
    subplots,
    x,
    y,
):
    U = []
    V = []
    for i_2 in range(corrs_1.shape[0]):
        tmpu, tmpv = correlation_to_displacement(corrs_1[i_2, :, :, :], nrows, ncols)
        U.append(tmpu)
        V.append(tmpv)
        fig_1, ax_1 = subplots(figsize=(6, 6))
        ax_1.quiver(x, y, tmpu, tmpv, scale=200)
        ax_1.invert_yaxis()
        plot(tmpu.mean(axis=1) * 80 + 400, y[:, 0])
    U = np.array(U)
    V = np.array(V)
    meanU = np.mean(U, axis=0)
    meanV = np.mean(V, axis=0)
    return meanU, meanV


@app.cell
def _(meanU, meanV, plot, subplots, x, y):
    fig_2, ax_2 = subplots(figsize=(8, 8))
    ax_2.quiver(x, y, meanU, meanV, scale=200)
    ax_2.invert_yaxis()
    plot(meanU.mean(axis=1) * 80 + 400, y[:, 0])
    return


if __name__ == "__main__":
    app.run()
