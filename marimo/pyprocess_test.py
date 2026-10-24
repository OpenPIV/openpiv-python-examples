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
    # testing extended search area piv from pyprocess
    # this won't show any difference since 0.23 version
    # because we incorporate the extended_seach_piv
    #
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    from openpiv.pyprocess import extended_search_area_piv, normalize_intensity
    from openpiv.tools import imread

    return extended_search_area_piv, imread, normalize_intensity, plt


@app.cell
def _(imread):
    frame_a = imread("data/test1/exp1_001_a.bmp")
    frame_b = imread("data/test1/exp1_001_b.bmp")
    return frame_a, frame_b


@app.cell
def _():
    # frame_a = frame_a[:128,:64]
    # frame_b = frame_b[:128,:64]
    return


@app.cell
def _(plt):
    def show_pair(I,J):
        fig, ax = plt.subplots(1,2,figsize=(12,8))
        ax[0].imshow(I,cmap=plt.cm.gray)
        ax[1].imshow(J,cmap=plt.cm.gray)

    return (show_pair,)


@app.cell
def _(frame_a, frame_b, show_pair):
    show_pair(frame_a, frame_b)
    return


@app.cell
def _(frame_a, frame_b, normalize_intensity):
    frame_a_1 = normalize_intensity(frame_a)
    frame_b_1 = normalize_intensity(frame_b)
    return frame_a_1, frame_b_1


@app.cell
def _(frame_a_1, frame_b_1, show_pair):
    show_pair(frame_a_1, frame_b_1)
    return


@app.cell
def _():
    window_size = 32
    overlap = 16
    dt=1.0
    search_area_size = 32
    correlation_method="circular"
    subpixel_method="gaussian"
    sig2noise_method='peak2peak'
    return (
        dt,
        overlap,
        search_area_size,
        sig2noise_method,
        subpixel_method,
        window_size,
    )


@app.cell
def _(
    dt,
    extended_search_area_piv,
    frame_a_1,
    frame_b_1,
    overlap,
    search_area_size,
    sig2noise_method,
    subpixel_method,
    window_size,
):
    # magic command not supported in marimo; please file an issue to add support
    # %%time
    vel1 = extended_search_area_piv(frame_a_1, frame_b_1, window_size=window_size, search_area_size=search_area_size, overlap=overlap, dt=dt, correlation_method='circular', subpixel_method=subpixel_method, sig2noise_method=sig2noise_method)
    return (vel1,)


@app.cell
def _(
    dt,
    extended_search_area_piv,
    frame_a_1,
    frame_b_1,
    overlap,
    sig2noise_method,
    subpixel_method,
):
    window_size_1 = 24
    search_area_size_1 = 32
    _vel2 = extended_search_area_piv(frame_a_1, frame_b_1, window_size=window_size_1, search_area_size=search_area_size_1, overlap=overlap, dt=dt, correlation_method='circular', normalized_correlation=True, subpixel_method=subpixel_method, sig2noise_method=sig2noise_method)
    return


@app.cell
def _(
    dt,
    extended_search_area_piv,
    frame_a_1,
    frame_b_1,
    overlap,
    sig2noise_method,
    subpixel_method,
):
    window_size_2 = 24
    search_area_size_2 = 32
    _vel2 = extended_search_area_piv(frame_a_1, frame_b_1, window_size=window_size_2, search_area_size=search_area_size_2, overlap=overlap, dt=dt, correlation_method='linear', subpixel_method=subpixel_method, sig2noise_method=sig2noise_method)
    return


@app.cell
def _(
    dt,
    extended_search_area_piv,
    frame_a_1,
    frame_b_1,
    overlap,
    sig2noise_method,
    subpixel_method,
):
    window_size_3 = 24
    search_area_size_3 = 32
    _vel2 = extended_search_area_piv(frame_a_1, frame_b_1, window_size=window_size_3, search_area_size=search_area_size_3, overlap=overlap, dt=dt, correlation_method='linear', normalized_correlation=True, subpixel_method=subpixel_method, sig2noise_method=sig2noise_method)
    return


@app.cell
def _(plt, vel1):
    plt.figure(figsize=(20,20))
    plt.quiver(vel1[0],-vel1[1],scale=100,color='b',alpha=0.5)
    return


if __name__ == "__main__":
    app.run()
