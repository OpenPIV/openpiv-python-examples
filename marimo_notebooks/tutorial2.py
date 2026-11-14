import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # OpenPIV tutorial 2

    Demonstrates the use of the multiprocessing run
    """)
    return


@app.cell
def _():
    from openpiv import tools, scaling, pyprocess, validation, filters
    import pathlib
    import numpy as np

    return filters, np, pathlib, pyprocess, tools, validation


@app.cell
def _(filters, np, path, pyprocess, tools, validation):
    def func( args ):
        """A function to process each image pair."""

        # this line is REQUIRED for multiprocessing to work
        # always use it in your custom function

        file_a, file_b, counter = args


        #####################
        # Here goes you code
        #####################

        # read images into numpy arrays
        frame_a  = tools.imread( path / file_a )
        frame_b  = tools.imread( path.joinpath(file_b) )

        frame_a = (frame_a*1024).astype(np.int32)
        frame_b = (frame_b*1024).astype(np.int32)


        # process image pair with extended search area piv algorithm.
        u, v, sig2noise = pyprocess.extended_search_area_piv( frame_a, frame_b, \
            window_size=64, overlap=32, dt=0.02, search_area_size=128, sig2noise_method='peak2peak')
        mask = validation.sig2noise_val( sig2noise, threshold = 1.5 )
        u, v = filters.replace_outliers( u, v, mask, method='localmean', max_iter=10, kernel_size=2)
        # get window centers coordinates
        x, y = pyprocess.get_coordinates( image_size=frame_a.shape, search_area_size=128, overlap=32 )
        # save to a file
        tools.save('test2_%03d.txt' % counter, x, y, u, v, mask)
        tools.display_vector_field('test2_%03d.txt' % counter)

    return (func,)


@app.cell
def _(func, pathlib, tools):
    path = pathlib.Path('../test2/')
    task = tools.Multiprocesser( data_dir = path, pattern_a='2image_*0.tif', pattern_b='2image_*1.tif' )
    task.run( func = func, n_cpus=1 )
    return (path,)


if __name__ == "__main__":
    app.run()
