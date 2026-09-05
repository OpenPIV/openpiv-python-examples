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
    # Single pass on complex images, effect of normalized correlation
    """)
    return


@app.cell
def _():
    import glob

    import matplotlib.pyplot as plt
    import numpy as np
    from openpiv import filters, pyprocess, scaling, tools, validation
    # '%matplotlib inline' command supported automatically in marimo
    return filters, glob, np, plt, pyprocess, scaling, tools, validation


@app.cell
def _():
    # set of typical parameters
    window_size = 32  # pixels 32 x 32 pixels interrogation window, in frame A.
    overlap = 16  # overlap is 8 pixels, i.e. 25% of the window
    search_size = 32  # pixels 64 x 64 in frame B, avoids some peak locking for 
                      # large displacements
    return overlap, search_size, window_size


@app.cell
def _(
    filters,
    list_of_images,
    np,
    overlap,
    plt,
    pyprocess,
    scaling,
    search_size,
    tools,
    validation,
    window_size,
):
    def openpiv_default_run(im1,im2,normalized_correlation=False):
        """ default settings for OpenPIV analysis using
        extended_search_area_piv algorithm for two images
    
        Inputs:
            im1,im2 : str,str = path of two image
        """
        frame_a  = tools.imread(im1)
        frame_b  = tools.imread(im2)

        u, v, sig2noise = pyprocess.extended_search_area_piv(frame_a.astype(np.int32), 
                                                           frame_b.astype(np.int32), 
                                                           window_size=window_size, 
                                                           overlap=overlap, 
                                                           dt=1, 
                                                           search_area_size=search_size, 
                                                           sig2noise_method='peak2peak',
                                                            correlation_method='linear',
                                                            normalized_correlation=normalized_correlation)
        x, y = pyprocess.get_coordinates(frame_a.shape, 
                                         search_size, 
                                         overlap)
        mask = validation.sig2noise_val(sig2noise, threshold = 1.0 )
        plt.figure()
        plt.hist(sig2noise.flatten(),51)
        plt.ylabel("Signal to noise ratio")
        u, v = filters.replace_outliers( u, v, mask, method='localmean', 
                                        max_iter=1, kernel_size=2)
        x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor = 1. )
        tools.save(list_of_images[0]+'.txt', x, y, u, v, mask )
        fig, ax = plt.subplots(figsize=(8,8))
        fig,ax = tools.display_vector_field(list_of_images[0]+'.txt', 
                                            on_img=True,image_name=list_of_images[0],
                                            scaling_factor=1.,
                                            scale=20,
                                            ax=ax)

    return (openpiv_default_run,)


@app.cell
def _(glob, openpiv_default_run):
    alist_filter = ['jpg','bmp','png','tif','tiff']

    # all test cases in /openpiv/examples/
    list_of_files = glob.glob("data/test3/*.*")
    list_of_files.sort()
    list_of_images = [f for f in list_of_files if f[-3:] in alist_filter]
    list_of_images.sort()
    if len(list_of_images) > 1:
        print(list_of_images[0], list_of_images[1])
        openpiv_default_run(list_of_images[0],list_of_images[1],normalized_correlation=False)
    return (list_of_images,)


@app.cell
def _(list_of_images, openpiv_default_run):
    if len(list_of_images) > 1:
        print(list_of_images[0], list_of_images[1])
        openpiv_default_run(list_of_images[0],list_of_images[1],normalized_correlation=True)
    return


if __name__ == "__main__":
    app.run()
