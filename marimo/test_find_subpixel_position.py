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


@app.cell
def _():
    # let's make a test for subpixel localization
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from openpiv.pyprocess import find_first_peak

    return find_first_peak, np, plt


@app.cell
def _(np):
    N = 64

    corr = np.zeros((N, N))

    corr[2:5, 2:5] = 1
    corr[3, 3] = 2
    corr[3, 4] = 3
    corr[3, 5] = 1
    corr
    return N, corr


@app.cell
def _(corr, find_first_peak):
    pos, height = find_first_peak(corr)
    return height, pos


@app.cell
def _(height, pos):
    pos, height
    return


@app.cell
def _():
    from openpiv.pyprocess import find_subpixel_peak_position

    return (find_subpixel_peak_position,)


@app.cell
def _(corr, find_subpixel_peak_position):
    find_subpixel_peak_position(corr)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## let's find some corner cases
    """)
    return


@app.cell
def _(N, np):
    # peak on the border
    corr_1 = np.zeros((N, N))
    corr_1[:3, :3] = 1
    corr_1[0, 0] = 2
    corr_1[0, 2] = 3
    corr_1[0, 3] = 1
    corr_1
    return (corr_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Corner case 1: peak on the border

    it is disregarded in our function because we cannot define well the subpixel
    position. Or do we?
    """)
    return


@app.cell
def _(corr_1, find_subpixel_peak_position):
    find_subpixel_peak_position(corr_1)
    return


@app.cell
def _(corr_1, np):
    # peak on the border
    corr_2 = np.flipud(corr_1)
    corr_2
    return (corr_2,)


@app.cell
def _(corr_2, find_subpixel_peak_position):
    find_subpixel_peak_position(corr_2)
    return


@app.cell
def _(corr_2, np):
    corr_3 = np.fliplr(corr_2)
    corr_3[-2, -1] = 5
    corr_3
    return (corr_3,)


@app.cell
def _(corr_3, find_subpixel_peak_position):
    find_subpixel_peak_position(corr_3)
    return


@app.cell
def _():
    ## Corner case 2: zero next to the peak - the log(0) fails
    return


@app.cell
def _(N, np):
    corr_4 = np.zeros((N, N))
    corr_4[2:5, 2:5] = 1
    corr_4[3, 3] = 2
    corr_4[3, 4] = 3
    # corr[3,5] = 1
    corr_4
    return (corr_4,)


@app.cell
def _(corr_4, find_subpixel_peak_position):
    find_subpixel_peak_position(corr_4)
    return


@app.cell
def _(corr_4, find_subpixel_peak_position, np, plt):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.pcolor(corr_4[:8, :8])
    for eps in np.logspace(-15, 5):
        i, j = find_subpixel_peak_position(corr_4 + eps)
        ax.plot(i, j, "rx")  # print(i,j)
    return eps, i, j


@app.cell
def _(corr_4, i, j, plt):
    plt.pcolor(corr_4[:8, :8])
    plt.plot(i, j, "ro")
    return


@app.cell
def _(corr_4, eps, find_subpixel_peak_position):
    for method in ["gaussian", "parabolic", "centroid"]:
        i_1, j_1 = find_subpixel_peak_position(corr_4, method)
        print(i_1, j_1)
        i_1, j_1 = find_subpixel_peak_position(corr_4 + eps, method)
        print(i_1, j_1)
    return


if __name__ == "__main__":
    app.run()
