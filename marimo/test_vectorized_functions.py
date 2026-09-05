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
    ### Comparison with vectorized and original functions
    #### Edited by Erich Zimmer
    #### Created at 20210817, 2109 CTZ
    """)
    return


@app.cell
def _():
    from glob import glob

    import matplotlib.pyplot as plt
    import numpy as np
    from openpiv.pyprocess import (
        find_first_peak,
        vectorized_correlation_to_displacements,
    )
    from openpiv.tools import imread

    return (
        find_first_peak,
        glob,
        imread,
        np,
        plt,
        vectorized_correlation_to_displacements,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Vectorized solution for subpixel estimation
    """)
    return


@app.cell
def _(np):
    N = 64

    corr = np.zeros((N,N))

    corr[2:5,2:5] = 1
    corr[3,3] = 2
    corr[3,4] = 3
    corr[3,5] = 1
    corr
    return N, corr


@app.cell
def _(corr, find_first_peak):
    pos,height = find_first_peak(corr)
    return height, pos


@app.cell
def _(height, pos):
    pos,height
    return


@app.cell
def _():
    from openpiv.pyprocess import find_subpixel_peak_position

    return (find_subpixel_peak_position,)


@app.cell
def _(corr, find_subpixel_peak_position):
    find_subpixel_peak_position(corr)
    return


@app.cell
def _(corr, np, vectorized_correlation_to_displacements):
    np.flip(vectorized_correlation_to_displacements(corr[np.newaxis, :, :]) + np.floor(corr.shape[0] / 2))
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
def _(corr_1, np, vectorized_correlation_to_displacements):
    np.flip(vectorized_correlation_to_displacements(corr_1[np.newaxis, :, :]) + np.floor(corr_1.shape[0] / 2))
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
def _(corr_2, np, vectorized_correlation_to_displacements):
    np.flip(vectorized_correlation_to_displacements(corr_2[np.newaxis, :, :]) + np.floor(corr_2.shape[0] / 2))
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
def _(corr_3, np, vectorized_correlation_to_displacements):
    np.flip(vectorized_correlation_to_displacements(corr_3[np.newaxis, :, :]) + np.floor(corr_3.shape[0] / 2))
    return


@app.cell
def _():
    ## Corner case 2: negative value next to peak - the log(n<0) fails
    return


@app.cell
def _(N, np):
    corr_4 = np.zeros((N, N))
    corr_4[2:5, 2:5] = 1
    corr_4[3, 3] = 2
    corr_4[3, 4] = 3
    corr_4 = corr_4 - 0.5
    # corr[3,5] = 1
    corr_4
    return (corr_4,)


@app.cell
def _(corr_4, find_subpixel_peak_position):
    find_subpixel_peak_position(corr_4)  # automatically uses parabolic method
    return


@app.cell
def _(corr_4, np, vectorized_correlation_to_displacements):
    np.flip(vectorized_correlation_to_displacements(corr_4[np.newaxis, :, :]) + np.floor(corr_4.shape[0] / 2))
    return


@app.cell
def _():
    ## Corner case 3: zero next to the peak - the log(0) fails
    return


@app.cell
def _(N, np):
    corr_5 = np.zeros((N, N))
    corr_5[2:5, 2:5] = 1
    corr_5[3, 3] = 2
    corr_5[3, 4] = 3
    # corr[3,5] = 1
    corr_5
    return (corr_5,)


@app.cell
def _(corr_5, find_subpixel_peak_position):
    find_subpixel_peak_position(corr_5)
    return


@app.cell
def _(corr_5, np, vectorized_correlation_to_displacements):
    np.flip(vectorized_correlation_to_displacements(corr_5[np.newaxis, :, :]) + np.floor(corr_5.shape[0] / 2))
    return


@app.cell
def _(corr_5, find_subpixel_peak_position):
    eps = 1e-07
    for _method in ['gaussian', 'parabolic', 'centroid']:
        _i, _j = find_subpixel_peak_position(corr_5, _method)
        print(_i, _j)
        _i, _j = find_subpixel_peak_position(corr_5 + eps, _method)
        print(_i, _j)
    return


@app.cell
def _(corr_5, np, vectorized_correlation_to_displacements):
    for _method in ['gaussian', 'parabolic', 'centroid']:
        _j, _i = vectorized_correlation_to_displacements(corr_5[np.newaxis, :, :], subpixel_method=_method) + np.floor(corr_5.shape[0] / 2)
        print(_i, _j)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Speed increase demonstration
    """)
    return


@app.cell
def _():
    import pylab

    return (pylab,)


@app.cell
def _(imread, np, pylab):
    frame_a = imread("data/test11/A001_1.tif")
    frame_b = imread("data/test11/A001_2.tif")
    pylab.imshow(np.c_[frame_a,np.ones((frame_a.shape[0],20)),frame_b],
                 cmap=pylab.cm.gray)
    return frame_a, frame_b


@app.cell
def _(frame_a, frame_b):
    _window_size = 32
    _overlap = 16
    from openpiv.pyprocess import (
        correlation_to_displacement,
        fft_correlate_images,
        get_coordinates,
        get_field_shape,
        moving_window_array,
    )
    n_rows, n_cols = get_field_shape(frame_a.shape, _window_size, _overlap)
    x, y = get_coordinates(frame_a.shape, _window_size, _overlap)
    _aa = moving_window_array(frame_a, _window_size, _overlap)
    _bb = moving_window_array(frame_b, _window_size, _overlap)
    corr_6 = fft_correlate_images(_aa, _bb, correlation_method='circular', normalized_correlation=True)
    return (
        corr_6,
        correlation_to_displacement,
        fft_correlate_images,
        get_field_shape,
        moving_window_array,
        n_cols,
        n_rows,
    )


@app.cell
def _(corr_6):
    from openpiv.pyprocess import find_all_second_peaks, find_second_peak
    peaks_v = find_all_second_peaks(corr_6)[0]
    peaks_o = []
    for _i in range(len(corr_6)):
        (k, m), _ = find_second_peak(corr_6[_i, :, :])
        peaks_o.append([_i, k, m])
    print(['original', 'vectorized'])
    for _i in range(len(peaks_v)):
        if peaks_v[_i][1] != peaks_o[_i][1] or peaks_v[_i][2] != peaks_o[_i][2]:
            print(False)
    return


@app.cell
def _(corr_6, correlation_to_displacement, n_cols, n_rows):
    u_o, _v_o = correlation_to_displacement(corr_6, n_rows, n_cols, subpixel_method="gaussian")
    return (u_o,)


@app.cell
def _(corr_6, n_cols, n_rows, vectorized_correlation_to_displacements):
    u_v, _v_v = vectorized_correlation_to_displacements(corr_6, n_rows, n_cols, subpixel_method="gaussian")
    return (u_v,)


@app.cell
def _(np, u_o, u_v):
    # slight descrepancies possibly caused by setting eps to 1e-10
    print("[u original, u vectorized]")
    print(np.stack((u_o[0, 0:12], u_v[0, 0:12])).T)
    print((np.nanmean(u_o), np.nanmean(u_v)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Vectorized solution for signal-to-noise calculation
    """)
    return


@app.cell
def _():
    from openpiv.pyprocess import sig2noise_ratio, vectorized_sig2noise_ratio

    return sig2noise_ratio, vectorized_sig2noise_ratio


@app.cell
def _(corr_6, sig2noise_ratio):
    # magic command not supported in marimo; please file an issue to add support
    # %%time
    peak2peak_o = sig2noise_ratio(corr_6, "peak2peak")
    return (peak2peak_o,)


@app.cell
def _(corr_6, vectorized_sig2noise_ratio):
    # magic command not supported in marimo; please file an issue to add support
    # %%time
    peak2peak_v = vectorized_sig2noise_ratio(corr_6, "peak2peak")
    return (peak2peak_v,)


@app.cell
def _(corr_6, sig2noise_ratio):
    # magic command not supported in marimo; please file an issue to add support
    # %%time
    peak2mean_o = sig2noise_ratio(corr_6, "peak2mean")
    return (peak2mean_o,)


@app.cell
def _(corr_6, vectorized_sig2noise_ratio):
    # magic command not supported in marimo; please file an issue to add support
    # %%time
    peak2mean_v = vectorized_sig2noise_ratio(corr_6, "peak2mean")
    return (peak2mean_v,)


@app.cell
def _(np, peak2peak_o, peak2peak_v):
    print("[original, vectorized]")
    print(np.stack((peak2peak_o[0:10], peak2peak_v[0:10])).T)
    print((peak2peak_o.mean(), peak2peak_v.mean()))
    return


@app.cell
def _(np, peak2mean_o, peak2mean_v):
    print("[original, vectorized]")
    print(np.stack((peak2mean_o[0:10], peak2mean_v[0:10])).T)
    print((peak2mean_o.mean(), peak2mean_v.mean()))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Test for bias errors
    """)
    return


@app.cell
def _(glob):
    files = glob("test14/*.bmp")
    files_a = files[::2]
    files_b = files[1::2]
    return files_a, files_b


@app.cell
def _(
    correlation_to_displacement,
    fft_correlate_images,
    files_a,
    files_b,
    get_field_shape,
    imread,
    moving_window_array,
    np,
    vectorized_correlation_to_displacements,
):
    bias_error_original = []
    bias_error_vectorized = []
    _window_size = 32
    _overlap = 16
    real_disp = 3
    n = 1 / 32
    for _i in range(len(files_a)):
        frame_a_1 = imread(files_a[_i])
        frame_b_1 = imread(files_b[_i])
        n_rows_1, n_cols_1 = get_field_shape(frame_a_1.shape, _window_size, _overlap)
        _aa = moving_window_array(frame_a_1, _window_size, _overlap)
        _bb = moving_window_array(frame_b_1, _window_size, _overlap)
        corr_7 = fft_correlate_images(_aa, _bb, 'circular', False)
        u_o_1, _v_o = correlation_to_displacement(corr_7, n_rows_1, n_cols_1, "gaussian")
        u_v_1, _v_v = vectorized_correlation_to_displacements(corr_7, n_rows_1, n_cols_1, "gaussian")
        u_o_1 = u_o_1[2:-2, 2:-2]
        u_v_1 = u_v_1[2:-2, 2:-2]
        _v_o = _v_o[2:-2, 2:-2]
        _v_v = _v_v[2:-2, 2:-2]
        bias_error_original.append(np.hypot(real_disp, real_disp) - np.nanmean(np.hypot(u_o_1, _v_o)))
        bias_error_vectorized.append(np.hypot(real_disp, real_disp) - np.nanmean(np.hypot(u_v_1, _v_v)))
        real_disp = real_disp + n
    return bias_error_original, bias_error_vectorized, n


@app.cell
def _(bias_error_original, bias_error_vectorized, n, np, plt):
    fig, ax = plt.subplots()
    ax.set_ylabel("Bias error [px]")
    ax.set_xlabel("Real u and v displacements [px]")
    ax.plot(np.mgrid[3:4+n:n], bias_error_original)
    ax.plot(np.mgrid[3:4+n:n], bias_error_vectorized)
    ax.legend(
        ['orginal', 'vectorized'],
        loc = 'upper right'
    )
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
