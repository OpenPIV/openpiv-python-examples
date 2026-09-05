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
    ## OpenPIV tutorial of all test cases
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It also gives a clue on how to process batch series of images
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
    # magic command not supported in marimo; please file an issue to add support
    # %load_ext watermark
    # magic command not supported in marimo; please file an issue to add support
    # %watermark -v -m -p numpy,openpiv -g -b
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Square windows using integer inputs for window sizes, etc.
    """)
    return


@app.cell
def _():
    # set of typical parameters
    window_size = 32 # pixels 32 x 32 pixels interrogation window, in frame A.
    overlap = 16 # overlap is 8 pixels, i.e. 25% of the window
    search_size = 40  # pixels 64 x 64 in frame B, avoids some peak locking for 
                      # large displacements
    return overlap, search_size, window_size


@app.cell
def _(filters, list_of_images, np, plt, pyprocess, scaling, tools, validation):
    def openpiv_default_run(im1, im2, window_size, overlap, search_size):
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
                                                           sig2noise_method='peak2mean',
                                                           correlation_method='circular',
                                                           normalized_correlation=True)
        x, y = pyprocess.get_rect_coordinates(frame_a.shape, 
                                         search_size, 
                                         overlap)
        # 5% lowest range
        invalid_mask = validation.sig2noise_val(
                                            sig2noise, 
                                            threshold = np.percentile(sig2noise,2.5)
        )
    
        u, v = filters.replace_outliers( u, v, invalid_mask, method='localmean', 
                                        max_iter=3, kernel_size=3)
        x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor = 1. )

        x, y, u, v = tools.transform_coordinates(x, y, u, v)
    
        tools.save(list_of_images[0]+'.txt', x, y, u, v) 
        fig,ax = plt.subplots(figsize=(8,8))
        ax.set_title(im1+".txt")
        tools.display_vector_field(im1+'.txt', 
                                            on_img=True,image_name=list_of_images[0],
                                            scaling_factor=1.,
                                            ax=ax)
    
    #     tools.display_vector_field(list_of_images[0]+'.txt', 
    #                                         scaling_factor=1.,
    #                                         ax=ax[1])
    return (openpiv_default_run,)


@app.cell
def _(glob, openpiv_default_run, overlap, search_size, window_size):
    alist_filter = ['jpg', 'bmp', 'png', 'tif', 'tiff']
    list_of_tests = glob.glob("data/test*")
    list_of_tests.sort()
    # all test cases in /openpiv/examples/
    list_of_images = []
    for _test in list_of_tests:
        _list_of_files = glob.glob(_test + "/*.*")
        _list_of_files.sort()
        list_of_images = [f for f in _list_of_files if f[-3:] in alist_filter]
        if len(list_of_images) > 1:
            openpiv_default_run(list_of_images[0], list_of_images[1], window_size, overlap, search_size)
    return alist_filter, list_of_images, list_of_tests


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rectangular windows
    """)
    return


@app.cell
def _():
    # rectangular windows
    window_size_1 = [16, 32]  # pixels 24 x 32 pixels interrogation window, in frame A.
    overlap_1 = [8, 16]  # overlap is [12 x 16] , 50%
    search_size_1 = [40, 40]  # search size is larger than the window size to get also some large displacements
    return overlap_1, search_size_1, window_size_1


@app.cell
def _(
    alist_filter,
    glob,
    list_of_tests,
    openpiv_default_run,
    overlap_1,
    search_size_1,
    window_size_1,
):
    list_of_images_1 = []
    for _test in list_of_tests:
        _list_of_files = glob.glob(_test + "/*.*")
        _list_of_files.sort()
        list_of_images_1 = [f for f in _list_of_files if f[-3:] in alist_filter]
        if len(list_of_images_1) > 1:
            openpiv_default_run(list_of_images_1[0], list_of_images_1[1], window_size_1, overlap_1, search_size_1)
    return


if __name__ == "__main__":
    app.run()
