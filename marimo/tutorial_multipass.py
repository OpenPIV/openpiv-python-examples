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
    import matplotlib.pyplot as plt
    import numpy as np
    from numpy import percentile
    from openpiv import filters, pyprocess, scaling, tools, validation

    return filters, np, plt, pyprocess, scaling, tools, validation, percentile


@app.cell
def _(tools):
    frame_a = tools.imread("data/test3/Y4-S3_Camera000398.tif")
    frame_b = tools.imread("data/test3/Y4-S3_Camera000399.tif")
    return frame_a, frame_b


@app.cell
def _(frame_a, frame_b, np, plt):
    plt.imshow(np.c_[frame_a[40:, :-40], frame_b[40:, :-40]], cmap=plt.cm.gray)
    return


@app.cell
def _(frame_a, frame_b, np):
    frame_a_1 = frame_a[40:, :-40].astype(
        np.int32
    )  # change of type for the Cython WiDIM
    frame_b_1 = frame_b[40:, :-40].astype(np.int32)
    return frame_a_1, frame_b_1


@app.cell
def _(frame_a_1, frame_b_1, pyprocess):
    # Use Python version, pyprocess:
    u, v, sig2noise = pyprocess.extended_search_area_piv(
        frame_a_1,
        frame_b_1,
        window_size=32,
        overlap=16,
        search_area_size=32,
        dt=0.1,
        sig2noise_method="peak2peak",
        normalized_correlation=True,
        correlation_method="circular",
    )
    x, y = pyprocess.get_coordinates(
        image_size=frame_a_1.shape, search_area_size=32, overlap=16
    )
    return sig2noise, u, v, x, y


@app.cell
def _(plt, sig2noise, u, v, x, y):
    plt.quiver(x, y, u, -v, sig2noise)
    plt.gca().invert_yaxis()
    plt.colorbar()
    return


@app.cell
def _(np, plt, sig2noise):
    plt.hist(sig2noise.flatten())
    p = np.percentile(sig2noise, 5)  # bottom 5%
    plt.plot([p, p], [0, 35], lw=2)
    return (p,)


@app.cell
def _(filters, p, scaling, sig2noise, tools, u, v, validation, x, y):
    mask = validation.sig2noise_val(sig2noise, threshold=p)
    u_1, v_1 = filters.replace_outliers(
        u, v, mask, method="localmean", max_iter=1, kernel_size=2
    )
    x_1, y_1, u_1, v_1 = scaling.uniform(x, y, u_1, v_1, scaling_factor=1.0)
    x_1, y_1, u_1, v_1 = tools.transform_coordinates(x_1, y_1, u_1, v_1)
    tools.save("Y4-S3_Camera000398_a.txt", x_1, y_1, u_1, v_1, mask)
    return u_1, v_1, x_1, y_1


@app.cell
def _(plt, u_1, v_1, x_1, y_1):
    # "natural" view without image
    fig, ax = plt.subplots(2, 1, figsize=(6, 12))
    # ax[0].invert_yaxis()
    ax[0].quiver(x_1, y_1, u_1, v_1)
    ax[0].set_title(" Sort of natural view ")
    ax[1].quiver(x_1, y_1, u_1, v_1)
    # plt.quiver(x,y,u,v)
    ax[1].set_title(
        "Quiver with 0,0 origin needs `negative` v for visualization purposes"
    )
    return


@app.cell
def _(plt, tools):
    fig_1, ax_1 = plt.subplots(figsize=(8, 8))
    tools.display_vector_field(
        "Y4-S3_Camera000398_a.txt",
        on_img=True,
        image_name="test3/Y4-S3_Camera000398.tif",
        scaling_factor=1.0,
        ax=ax_1,
    )
    return


@app.cell
def _(tools):
    tools.display_vector_field(
        "Y4-S3_Camera000398_a.txt", scale=300, width=0.005, scaling_factor=1.0
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
