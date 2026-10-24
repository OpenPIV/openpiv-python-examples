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
    ## OpenPIV tutorial of various correlation types

    In this notebook we compare the time to run the same analysis using Cython (precompiled) version
    with the Python process using FFT and/or direct cross-correlation method
    """)
    return


@app.cell
def _():
    import numpy as np
    import pylab
    from openpiv import filters, pyprocess, scaling, tools, validation
    # '%matplotlib inline' command supported automatically in marimo
    return filters, np, pylab, pyprocess, scaling, tools, validation


@app.cell
def _(np, pylab, tools):
    frame_a  = tools.imread("data/test1/exp1_001_a.bmp")
    frame_b  = tools.imread("data/test1/exp1_001_b.bmp")

    pylab.imshow(np.c_[frame_a,np.ones((frame_a.shape[0],20)),frame_b],
                 cmap=pylab.cm.gray)
    return frame_a, frame_b


@app.cell
def _():
    # Typical set of parameters 

    window_size = 32 # pixels, IW size in frame A
    overlap = 16 # 50% overlap
    search_area_size = 32 # pixels, IW in frame B, helps for large displacements
    dt = 0.02 # second, 50 Hz, just an example
    scaling_factor = 96.52 # micron/pixel
    return dt, overlap, scaling_factor, search_area_size, window_size


@app.cell
def _(
    dt,
    filters,
    frame_a,
    frame_b,
    np,
    overlap,
    pyprocess,
    scaling,
    scaling_factor,
    search_area_size,
    tools,
    validation,
    window_size,
):
    # magic command not supported in marimo; please file an issue to add support
    # %%time
    _u, _v, _sig2noise = pyprocess.extended_search_area_piv(frame_a, frame_b, window_size=window_size, overlap=overlap, dt=dt, search_area_size=search_area_size, sig2noise_method="peak2peak")
    # default correlation is FFT circular type (faster, less robust)
    # default type of correlation is not normalized, faster
    # we do not know the values of signal to noise ratio a priori
    # therefore we decide that we remove lowest 5%
    # the signal to noise ratio is defined here as 1st to 2nd peak ratio
    # All the parameters need to be checked. 
    _x, _y = pyprocess.get_coordinates(frame_a.shape, search_area_size=search_area_size, overlap=overlap)
    # get the values of displacements in pixel/sec units
    _mask = validation.sig2noise_val(_sig2noise, threshold=np.percentile(_sig2noise, 5))
    _u, _v = filters.replace_outliers(_u, _v, _mask, method='localmean', max_iter=10, kernel_size=2)
    _x, _y, _u, _v = scaling.uniform(_x, _y, _u, _v, scaling_factor=scaling_factor)
    _x, _y, _u, _v = tools.transform_coordinates(_x, _y, _u, _v)
    # prepare centers of the IWs to know where locate the vectors
    # removing and filling in the outlier vectors
    # rescale the results to millimeters and mm/sec
    # save the data
    tools.save('circular_default.txt', _x, _y, _u, _v, _mask)
    return


@app.cell
def _(
    dt,
    filters,
    frame_a,
    frame_b,
    np,
    overlap,
    pyprocess,
    scaling,
    scaling_factor,
    search_area_size,
    tools,
    validation,
    window_size,
):
    # magic command not supported in marimo; please file an issue to add support
    # %%time
    _u, _v, _sig2noise = pyprocess.extended_search_area_piv(frame_a, frame_b, window_size=window_size, overlap=overlap, dt=dt, search_area_size=search_area_size, sig2noise_method='peak2peak', normalized_correlation=True)
    # use normalized_correlation
    # both image intensity is normalized before correlation
    # and the correlation map has peaks between 0..1
    _x, _y = pyprocess.get_coordinates(frame_a.shape, search_area_size=search_area_size, overlap=overlap)
    # get the values of displacements in pixel/sec units
    _mask = validation.sig2noise_val(_sig2noise, threshold=np.percentile(_sig2noise, 5))
    _u, _v = filters.replace_outliers(_u, _v, _mask, method='localmean', max_iter=10, kernel_size=2)
    _x, _y, _u, _v = scaling.uniform(_x, _y, _u, _v, scaling_factor=scaling_factor)
    _x, _y, _u, _v = tools.transform_coordinates(_x, _y, _u, _v)
    # prepare centers of the IWs to know where locate the vectors
    # removing and filling in the outlier vectors
    # rescale the results to millimeters and mm/sec
    # save the data
    tools.save('circular_normalized.txt', _x, _y, _u, _v, _mask)
    return


@app.cell
def _(
    dt,
    filters,
    frame_a,
    frame_b,
    np,
    overlap,
    pyprocess,
    scaling,
    scaling_factor,
    search_area_size,
    tools,
    validation,
    window_size,
):
    # magic command not supported in marimo; please file an issue to add support
    # %%time
    _u, _v, _sig2noise = pyprocess.extended_search_area_piv(pyprocess.normalize_intensity(frame_a), pyprocess.normalize_intensity(frame_b), window_size=window_size, overlap=overlap, dt=dt, search_area_size=search_area_size, sig2noise_method='peak2peak', correlation_method="linear")
    # change to another type of correlation 'linear' - uses
    # zero padding prior to the correlation
    # it requires uniform background and therefore 
    # we need to normalize intensity of the images
    _x, _y = pyprocess.get_coordinates(frame_a.shape, search_area_size=search_area_size, overlap=overlap)
    _mask = validation.sig2noise_val(_sig2noise, threshold=np.percentile(_sig2noise, 5))
    _u, _v = filters.replace_outliers(_u, _v, _mask, method='localmean', max_iter=10, kernel_size=2)
    _x, _y, _u, _v = scaling.uniform(_x, _y, _u, _v, scaling_factor=scaling_factor)
    _x, _y, _u, _v = tools.transform_coordinates(_x, _y, _u, _v)
    # prepare centers of the IWs to know where locate the vectors
    # removing and filling in the outlier vectors
    # rescale the results to millimeters and mm/sec
    # save the data
    tools.save('linear_intensity.txt', _x, _y, _u, _v, _mask)
    return


@app.cell
def _(
    dt,
    filters,
    frame_a,
    frame_b,
    np,
    overlap,
    pyprocess,
    scaling,
    scaling_factor,
    search_area_size,
    tools,
    validation,
    window_size,
):
    # magic command not supported in marimo; please file an issue to add support
    # %%time
    _u, _v, _sig2noise = pyprocess.extended_search_area_piv(pyprocess.normalize_intensity(frame_a), pyprocess.normalize_intensity(frame_b), window_size=window_size, overlap=overlap, dt=dt, search_area_size=search_area_size, sig2noise_method='peak2peak', correlation_method='linear', normalized_correlation=True)
    # add normalized correlation to linear
    _x, _y = pyprocess.get_coordinates(frame_a.shape, search_area_size=search_area_size, overlap=overlap)
    _mask = validation.sig2noise_val(_sig2noise, threshold=np.percentile(_sig2noise, 5))
    _u, _v = filters.replace_outliers(_u, _v, _mask, method='localmean', max_iter=10, kernel_size=2)
    _x, _y, _u, _v = scaling.uniform(_x, _y, _u, _v, scaling_factor=scaling_factor)
    _x, _y, _u, _v = tools.transform_coordinates(_x, _y, _u, _v)
    # prepare centers of the IWs to know where locate the vectors
    # removing and filling in the outlier vectors
    # rescale the results to millimeters and mm/sec
    # save the data
    tools.save('linear_normalized.txt', _x, _y, _u, _v, None, _mask)
    return


@app.cell
def _(
    dt,
    filters,
    frame_a,
    frame_b,
    np,
    overlap,
    pyprocess,
    scaling,
    scaling_factor,
    tools,
    validation,
    window_size,
):
    search_area_size_1 = 40
    _u, _v, _sig2noise = pyprocess.extended_search_area_piv(frame_a, frame_b, window_size=window_size, overlap=overlap, dt=dt, search_area_size=search_area_size_1, sig2noise_method='peak2peak', correlation_method='linear', normalized_correlation=True)
    _x, _y = pyprocess.get_coordinates(frame_a.shape, search_area_size=search_area_size_1, overlap=overlap)
    _mask = validation.sig2noise_val(_sig2noise, threshold=np.percentile(_sig2noise, 5))
    _u, _v = filters.replace_outliers(_u, _v, _mask, method='localmean', max_iter=10, kernel_size=2)
    _x, _y, _u, _v = scaling.uniform(_x, _y, _u, _v, scaling_factor=scaling_factor)
    _x, _y, _u, _v = tools.transform_coordinates(_x, _y, _u, _v)
    tools.save('linear_normalized_extended.txt', _x, _y, _u, _v, _mask)
    return (search_area_size_1,)


@app.cell
def _(
    dt,
    filters,
    frame_a,
    frame_b,
    np,
    overlap,
    pyprocess,
    scaling,
    scaling_factor,
    search_area_size_1,
    tools,
    validation,
    window_size,
):
    _u, _v, _sig2noise = pyprocess.extended_search_area_piv(frame_a, frame_b, window_size=window_size, overlap=overlap, dt=dt, search_area_size=search_area_size_1, sig2noise_method="peak2peak")
    _x, _y = pyprocess.get_coordinates(frame_a.shape, search_area_size=search_area_size_1, overlap=overlap)
    _mask = validation.sig2noise_val(_sig2noise, threshold=np.percentile(_sig2noise, 5))
    _u, _v = filters.replace_outliers(_u, _v, _mask, method='localmean', max_iter=10, kernel_size=2)
    _x, _y, _u, _v = scaling.uniform(_x, _y, _u, _v, scaling_factor=scaling_factor)
    _x, _y, _u, _v = tools.transform_coordinates(_x, _y, _u, _v)
    tools.save('circular_extended.txt', _x, _y, _u, _v, _mask)
    return


@app.cell
def _(tools):
    tools.display_vector_field('linear_normalized_extended.txt', scale=30)
    tools.display_vector_field('linear_normalized.txt', scale=30)
    tools.display_vector_field('linear_intensity.txt', scale=30)
    tools.display_vector_field('circular_default.txt', scale=30)
    tools.display_vector_field('circular_normalized.txt', scale=30)
    tools.display_vector_field('circular_extended.txt', scale=30)
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
