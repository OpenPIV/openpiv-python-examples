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
    # Phase separation tutorial
    In the first section, we load an artificial two-phase PIV image and then use different phase separation methods to discriminate the two phases and save them as separate images. This section aims to introduce different methods that are available.

    In the second section, we demonstrate phase separation on real images followed by actual PIV process as a more realistic example.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Section 1 - Introducing different methods
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading image
    """)
    return


@app.cell
def _():
    from openpiv import tools, phase_separation
    import matplotlib.pyplot as plt
    plt.gray()
    return phase_separation, plt, tools


@app.cell
def _(plt, tools):
    two_phase_image = tools.imread("data/two_phase_piv/artificial_A.tif")
    plt.figure(figsize=(10,10))
    plt.imshow(two_phase_image)
    plt.show()
    return (two_phase_image,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Opening method
    """)
    return


@app.cell
def _(phase_separation, plt, two_phase_image):
    _big_particles_image, _small_particles_image = phase_separation.opening_method(two_phase_image, 11, thresh_factor=1.05)
    _fig, _ax = plt.subplots(1, 2, figsize=(16, 10))
    _ax[0].imshow(_big_particles_image, vmin=0, vmax=255)
    _ax[1].imshow(_small_particles_image, interpolation='bicubic', vmin=0, vmax=255)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Median filter method
    """)
    return


@app.cell
def _(phase_separation, plt, two_phase_image):
    _big_particles_image, _small_particles_image = phase_separation.median_filter_method(two_phase_image, 11)
    _fig, _ax = plt.subplots(1, 2, figsize=(16, 10))
    _ax[0].imshow(_big_particles_image)
    _ax[1].imshow(_small_particles_image, vmin=0, vmax=255)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Khalitov-Longmire method
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Using a simple size limit
    """)
    return


@app.cell
def _(phase_separation, plt, two_phase_image):
    # Plot size distribution
    plt.hist( phase_separation.get_particles_size_array(two_phase_image), 30)
    plt.yscale("log")
    plt.title("Particle size distribution")
    plt.show()
    return


@app.cell
def _(phase_separation, plt, two_phase_image):
    # Choose size limit = 100 
    _big_particles_criteria = {'min_size': 75}
    _small_particles_criteria = {'max_size': 75}
    _big_particles_image, _small_particles_image = phase_separation.khalitov_longmire(two_phase_image, _big_particles_criteria, _small_particles_criteria)
    _fig, _ax = plt.subplots(1, 2, figsize=(16, 10))
    _ax[0].imshow(_big_particles_image)
    _ax[1].imshow(_small_particles_image)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Using size and brightness limits
    """)
    return


@app.cell
def _(phase_separation, plt, two_phase_image):
    # First plot size-brightness map
    sb_map = phase_separation.get_size_brightness_map( two_phase_image ) # It is possible to sum over maps of multiple images
    plt.figure(figsize=(10,8))
    plt.imshow( sb_map, interpolation='nearest', aspect='auto', origin='lower', cmap="jet")
    plt.colorbar()
    plt.xlabel("Brightness")
    plt.ylabel("Size (px)")
    plt.title("Signal density")
    plt.show()
    return


@app.cell
def _(phase_separation, plt, two_phase_image):
    # size-brightness rectangle regions 
    _big_particles_criteria = {'min_size': 100, 'max_size': 350, 'min_brightness': 100, 'max_brightness': 180}
    _small_particles_criteria = {'min_size': 25, 'max_size': 100, 'min_brightness': 30, 'max_brightness': 100}
    _big_particles_image, _small_particles_image = phase_separation.khalitov_longmire(two_phase_image, _big_particles_criteria, _small_particles_criteria)
    _fig, _ax = plt.subplots(1, 2, figsize=(16, 10))
    _ax[0].imshow(_big_particles_image)
    _ax[1].imshow(_small_particles_image)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Section 2 - Practical example
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading image
    """)
    return


@app.cell
def _(plt, tools):
    raw_A = tools.imread("data/two_phase_piv/real_A.tif")
    raw_B = tools.imread("data/two_phase_piv/real_B.tif")
    _fig, _ax = plt.subplots(1, 2, figsize=(16, 10))
    _ax[0].imshow(raw_A)
    _ax[1].imshow(raw_B)
    plt.show()
    return raw_A, raw_B


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Separating phases
    """)
    return


@app.cell
def _(phase_separation, plt, raw_A, raw_B):
    solid_A, carrier_A = phase_separation.opening_method(raw_A, 7, thresh_factor=1.05)
    solid_B, carrier_B = phase_separation.opening_method(raw_B, 7, thresh_factor=1.05)
    _fig, _ax = plt.subplots(2, 2, figsize=(16, 10))
    _ax[0, 0].imshow(solid_A, vmin=0, vmax=255)
    _ax[0, 1].imshow(carrier_A, vmin=0, vmax=255)
    _ax[1, 0].imshow(solid_B, vmin=0, vmax=255)
    _ax[1, 1].imshow(carrier_A, vmin=0, vmax=255)
    plt.show()
    return carrier_A, carrier_B, solid_A, solid_B


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Carrier phase PIV
    """)
    return


@app.cell
def _():
    from openpiv import validation, filters, scaling, pyprocess
    import numpy as np

    return filters, np, pyprocess, scaling, validation


@app.cell
def _(carrier_A, carrier_B, np, pyprocess):
    winsize = 48 # pixels
    searchsize = 128  # pixels, search in image B
    overlap = 24 # pixels
    dt = 20e-6 # sec
    scaling_factor = 2500 # pixels/meter

    u0, v0, sig2noise = pyprocess.extended_search_area_piv( carrier_A.astype(np.int32), carrier_B.astype(np.int32),
                    subpixel_method ='parabolic', window_size=winsize, 
                    overlap=overlap, dt=dt, search_area_size=searchsize, 
                    sig2noise_method="peak2peak")
    x, y = pyprocess.get_coordinates(carrier_A.shape,searchsize,overlap)
    return dt, scaling_factor, sig2noise, u0, v0, x, y


@app.cell
def _(
    filters,
    scaling,
    scaling_factor,
    sig2noise,
    tools,
    u0,
    v0,
    validation,
    x,
    y,
):
    mask = validation.sig2noise_val(sig2noise, threshold=3.2)
    u2, v2 = filters.replace_outliers(u0, v0, mask, method='localmean', max_iter=5, kernel_size=2)
    x_1, y_1, u3, v3 = scaling.uniform(x, y, u2, v2, scaling_factor=scaling_factor)
    x_1, y_1, u3, v3 = tools.transform_coordinates(x_1, y_1, u3, v3)
    tools.save('data.txt', x_1, y_1, u3, v3, mask)
    tools.save('preview.txt', x_1[::5, ::5], y_1[::5, ::5], u3[::5, ::5], v3[::5, ::5], mask[::5, ::5])
    return


@app.cell
def _(plt, scaling_factor, tools):
    _fig, _ax = plt.subplots(figsize=(16, 10))
    tools.display_vector_field('preview.txt', on_img=True, image_name='data/two_phase_piv/real_A.tif', scaling_factor=scaling_factor, ax=_ax)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Solid phase PTV
    """)
    return


@app.cell
def _(dt, np, plt, scaling_factor, solid_A, solid_B, tools):
    # Note that this is an over-simplified PTV code, just for demonstration.
    # Please use OpenPTV or other PTV software for meaningful results.
    from skimage.measure import label, regionprops
    # '%matplotlib inline' command supported automatically in marimo
    # magic command not supported in marimo; please file an issue to add support
    # %config InlineBackend.close_figures=False
    from scipy.spatial.distance import cdist
    from IPython.display import clear_output
    centers_A = np.asarray([p.centroid for p in regionprops(label(solid_A > 2 * np.mean(solid_A)))])
    centers_B = np.asarray([p.centroid for p in regionprops(label(solid_B > 2 * np.mean(solid_B)))])
    cross_distance = cdist(centers_B, centers_A)
    PTV_X = []
    PTV_Y = []
    PTV_U = []
    PTV_V = []
    for i in range(1, len(centers_A)):
        if np.min(cross_distance, 0)[i] < 5:
            j = np.argmin(cross_distance, 0)[i]
            PTV_X.append(centers_A[i, 1] / scaling_factor)
            PTV_Y.append(centers_A[i, 0] / scaling_factor)
            PTV_U.append((centers_B[j, 1] - centers_A[i, 1]) / (scaling_factor * dt))
            PTV_V.append(-(centers_B[j, 0] - centers_A[i, 0]) / (scaling_factor * dt))
    PTV_X = np.asarray(PTV_X)
    PTV_Y = np.asarray(PTV_Y)
    PTV_U = np.asarray(PTV_U)
    PTV_V = np.asarray(PTV_V)
    width = solid_A.shape[1] / scaling_factor
    height = solid_A.shape[0] / scaling_factor
    fig2, ax2 = plt.subplots(figsize=(16, 10))
    ax2.imshow(solid_A, extent=[0, width, 0, height], origin="lower")
    tools.display_vector_field('preview.txt', scaling_factor=scaling_factor, ax=ax2)
    ax2.invert_yaxis()
    ax2.quiver(PTV_X, PTV_Y, PTV_U, PTV_V, color='yellow', width=0.003)
    ax2.invert_yaxis()
    clear_output(wait=True)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
