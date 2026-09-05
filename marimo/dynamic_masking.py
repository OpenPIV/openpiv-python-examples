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
    from openpiv import filters, preprocess, pyprocess, scaling, tools, validation

    # '%matplotlib inline' command supported automatically in marimo
    return filters, np, plt, preprocess, pyprocess, scaling, tools, validation


@app.cell
def _(np, plt, tools):
    file_a = "test4/Camera1-0101.tif"
    file_b = "test4/Camera1-0102.tif"

    im_a = tools.imread(file_a)
    im_b = tools.imread(file_b)
    plt.imshow(np.c_[im_a, im_b], cmap="gray")
    return im_a, im_b


@app.cell
def _(im_a, im_b, np, plt):
    # let's crop the region of interest
    frame_a = im_a[380:1980, 0:1390]
    frame_b = im_b[380:1980, 0:1390]
    plt.imshow(np.c_[frame_a, frame_b], cmap="gray")
    return frame_a, frame_b


@app.cell
def _(filters, frame_a, frame_b, np, pyprocess, scaling, tools, validation):
    # Process the original cropped image and see the OpenPIV result:
    window_size = 32
    # typical parameters:
    overlap = 16  # pixels
    search_area_size = 64  # pixels
    frame_rate = 40  # pixels
    scaling_factor = 96.52  # fps
    _u, _v, _sig2noise = pyprocess.extended_search_area_piv(
        frame_a.astype(np.int32),
        frame_b.astype(np.int32),
        window_size=window_size,
        overlap=overlap,
        dt=1.0 / frame_rate,
        search_area_size=search_area_size,
        sig2noise_method="peak2peak",
    )  # micron/pixel
    _x, _y = pyprocess.get_coordinates(frame_a.shape, search_area_size, overlap)
    # process again with the masked images, for comparison# process once with the original images
    _mask_g = validation.global_val(_u, _v, (-300.0, 300.0), (-300.0, 300.0))
    _mask_s2n = validation.sig2noise_val(_sig2noise, threshold=1.1)
    _mask = _mask_g | _mask_s2n
    _u, _v = filters.replace_outliers(
        _u, _v, _mask, method="localmean", max_iter=3, kernel_size=3
    )
    _x, _y, _u, _v = scaling.uniform(_x, _y, _u, _v, scaling_factor=96.52)
    _x, _y, _u, _v = tools.transform_coordinates(_x, _y, _u, _v)
    tools.save("test.txt", _x, _y, _u, _v, _mask)
    # save to a file
    tools.display_vector_field("test.txt", scale=5, width=0.006)
    return frame_rate, overlap, scaling_factor, search_area_size, window_size


@app.cell
def _(frame_a, frame_b, np, plt, preprocess):
    # masking using not optimal choice of the methods or parameters:
    masked_a, _ = preprocess.dynamic_masking(
        frame_a, method="edges", filter_size=7, threshold=0.005
    )
    masked_b, _ = preprocess.dynamic_masking(
        frame_b, method="intensity", filter_size=3, threshold=0.0
    )
    plt.imshow(np.c_[masked_a, masked_b], cmap="gray")
    return


@app.cell
def _(frame_a, frame_b, np, plt, preprocess):
    # masking using optimal (manually tuned) set of parameters and the right method:
    masked_a_1, _ = preprocess.dynamic_masking(
        frame_a, method="edges", filter_size=15, threshold=0.005
    )
    masked_b_1, _ = preprocess.dynamic_masking(
        frame_b, method="edges", filter_size=15, threshold=0.005
    )
    plt.imshow(np.c_[masked_a_1, masked_b_1], cmap="gray")
    return masked_a_1, masked_b_1


@app.cell
def _(
    filters,
    frame_rate,
    masked_a_1,
    masked_b_1,
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
    _u, _v, _sig2noise = pyprocess.extended_search_area_piv(
        masked_a_1.astype(np.int32),
        masked_b_1.astype(np.int32),
        window_size=window_size,
        overlap=overlap,
        dt=1.0 / frame_rate,
        search_area_size=search_area_size,
        sig2noise_method="peak2peak",
    )
    _x, _y = pyprocess.get_coordinates(masked_a_1.shape, search_area_size, overlap)
    _mask_g = validation.global_val(_u, _v, (-300.0, 300.0), (-300.0, 300.0))
    _mask_s2n = validation.sig2noise_val(_sig2noise, threshold=1.1)
    _mask = _mask_g | _mask_s2n
    _u, _v = filters.replace_outliers(
        _u, _v, _mask, method="localmean", max_iter=3, kernel_size=3
    )
    _x, _y, _u, _v = scaling.uniform(_x, _y, _u, _v, scaling_factor=scaling_factor)
    _x, _y, _u, _v = tools.transform_coordinates(_x, _y, _u, _v)
    tools.save(
        "test_masked.txt", _x, _y, _u, _v, None, _mask, fmt="%9.6f", delimiter="\t"
    )
    tools.display_vector_field("test_masked.txt", scale=5, width=0.006)
    return


if __name__ == "__main__":
    app.run()
