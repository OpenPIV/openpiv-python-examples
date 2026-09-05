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
    # Breakdown the PIV into separate functions and details
    """)
    return


@app.cell
def _():
    # test the idea of vectorized cross correlation for 
    # strided images, rectangular windows and extended search area
    # in one function
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np


    return np, plt


@app.cell
def _():
    from openpiv.pyprocess import (
        correlation_to_displacement,
        fft_correlate_images,
        get_coordinates,
        get_field_shape,
        moving_window_array,
        normalize_intensity,
    )
    from openpiv.tools import imread, transform_coordinates

    return (
        correlation_to_displacement,
        fft_correlate_images,
        get_coordinates,
        get_field_shape,
        imread,
        moving_window_array,
        normalize_intensity,
        transform_coordinates,
    )


@app.cell
def _(imread):
    frame_a = imread("data/test1/exp1_001_a.bmp")
    frame_b = imread("data/test1/exp1_001_b.bmp")

    # frame_a = frame_a[:128,:128]
    # frame_b = frame_b[:128,:128]

    # frame_a = normalize_intensity(frame_a)
    # frame_b = normalize_intensity(frame_b)
    return frame_a, frame_b


@app.cell
def _():
    # for debugging purposes 
    # frame_a = frame_a[:64,:64]
    # frame_b = frame_b[:64,:64]
    return


@app.cell
def _():
    # parameters for the test
    window_size = 48
    overlap = 8
    search_size = window_size #not extended search for a while
    return overlap, search_size, window_size


@app.cell
def _(frame_a, frame_b, moving_window_array, overlap, window_size):
    # for the regular square windows case:
    aa = moving_window_array(frame_a, window_size, overlap)
    bb = moving_window_array(frame_b, window_size, overlap)
    return aa, bb


@app.cell
def _(
    aa,
    bb,
    correlation_to_displacement,
    fft_correlate_images,
    frame_a,
    get_coordinates,
    get_field_shape,
    overlap,
    search_size,
):
    c = fft_correlate_images(aa,bb)
    n_rows, n_cols = get_field_shape(frame_a.shape, search_size, overlap)
    u,v = correlation_to_displacement(c, n_rows,n_cols)
    x,y = get_coordinates(frame_a.shape,search_size,overlap)
    return u, v, x, y


@app.cell
def _():
    # let's assume we want the extended search type of PIV analysis
    # with search_area_size in image B > window_size in image A
    window_size_1 = 32
    overlap_1 = 8
    search_size_1 = 48
    return overlap_1, search_size_1, window_size_1


@app.cell
def _(
    frame_a,
    frame_b,
    moving_window_array,
    normalize_intensity,
    np,
    overlap_1,
    plt,
    search_size_1,
    window_size_1,
):
    # for the regular square windows case:
    aa_1 = moving_window_array(frame_a, search_size_1, overlap_1)
    bb_1 = moving_window_array(frame_b, search_size_1, overlap_1)
    aa_1 = normalize_intensity(aa_1)
    bb_1 = normalize_intensity(bb_1)
    plt.figure()
    plt.imshow(aa_1[-1, :, :], cmap=plt.cm.gray)
    # make it use only a small window inside a larger window
    mask = np.zeros((search_size_1, search_size_1))
    pad = int((search_size_1 - window_size_1) / 2)
    mask[slice(pad, search_size_1 - pad), slice(pad, search_size_1 - pad)] = 1
    mask = np.broadcast_to(mask, aa_1.shape)
    aa_1 = aa_1 * mask.astype(aa_1.dtype)
    plt.figure()
    plt.imshow(aa_1[0, :, :], cmap=plt.cm.gray)
    plt.figure()
    plt.imshow(bb_1[0, :, :], cmap=plt.cm.gray)
    return aa_1, bb_1


@app.cell
def _(aa_1, bb_1, fft_correlate_images):
    c1 = fft_correlate_images(aa_1, bb_1, correlation_method="linear")
    return (c1,)


@app.cell
def _(c1, plt):
    plt.contourf(c1[2,:,:])
    return


@app.cell
def _(frame_a, get_field_shape, overlap_1, search_size_1):
    n_rows_1, n_cols_1 = get_field_shape(frame_a.shape, search_size_1, overlap_1)
    return n_cols_1, n_rows_1


@app.cell
def _(
    c1,
    correlation_to_displacement,
    frame_a,
    get_coordinates,
    n_cols_1,
    n_rows_1,
    overlap_1,
    search_size_1,
):
    u1, v1 = correlation_to_displacement(c1, n_rows_1, n_cols_1)
    x1, y1 = get_coordinates(frame_a.shape, search_size_1, overlap_1)
    return u1, v1, x1, y1


@app.cell
def _(transform_coordinates, u, u1, v, v1, x, x1, y, y1):
    x_1, y_1, u_1, v_1 = transform_coordinates(x, y, u, v)
    x1_1, y1_1, u1_1, v1_1 = transform_coordinates(x1, y1, u1, v1)
    return u1_1, u_1, v1_1, v_1, x1_1, x_1, y1_1, y_1


@app.cell
def _(plt, u1_1, u_1, v1_1, v_1, x1_1, x_1, y1_1, y_1):
    plt.figure(figsize=(12, 12))
    plt.quiver(x_1, y_1, u_1, v_1, scale=100, color='b', alpha=0.2)
    plt.quiver(x1_1, y1_1, u1_1, v1_1, scale=100, color='r', alpha=0.2)
    return



@app.cell
def _(frame_a, frame_b, pyprocess):
    # OpenPIV 0.26.0: scipy.fft is now the default backend (2-3x faster than numpy.fft)
    # Optional Rust backend (openpiv-rust) provides multithreaded acceleration
    # Install with: pip install openpiv[rust]
    import time
    print("Rust available:", pyprocess.HAS_RUST)
    winsize, searchsize, overlap, dt = 32, 38, 12, 0.02
    for backend in (["scipy", "rust"] if pyprocess.HAS_RUST else ["scipy"]):
        t0 = time.time()
        try:
            u_b, v_b, s2n_b = pyprocess.extended_search_area_piv(frame_a, frame_b, window_size=winsize, overlap=overlap, dt=dt, search_area_size=searchsize, sig2noise_method="peak2peak", backend=backend)
            print(f"{backend:5s}: {time.time()-t0:.3f}s  mean u={u_b.mean():.3f} v={v_b.mean():.3f}")
        except Exception as e:
            print(backend, "failed:", e)
    return

if __name__ == "__main__":
    app.run()
