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
    # Active nematics demo case

    The source from the openpiv-users group:

    https://groups.google.com/g/openpiv-users/c/Us_q7h3Uri8/m/1p8XAYkHCQAJ

    https://github.com/OpenPIV/openpiv-python-examples/blob/main/test18/active-nematics.gif
    """)
    return


@app.cell
def _():
    from IPython.display import Image
    Image("data/test18/active-nematics.gif")
    return


@app.cell
def _():
    import pathlib

    # '%matplotlib inline' command supported automatically in marimo
    import matplotlib
    from openpiv import windef
    matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
    return pathlib, windef


@app.cell
def _(pathlib, windef):
    settings = windef.PIVSettings()


    'Data related settings'
    # Folder with the images to process
    settings.filepath_images = pathlib.Path("test18")
    # Folder for the outputs
    settings.save_path = pathlib.Path("test18/")
    # Root name of the output Folder for Result Files
    settings.save_folder_suffix = 'Test_1'
    # Format and Image Sequence
    settings.frame_pattern_a = 'active-nematics0*.tif'
    settings.frame_pattern_b = '(1+2),(3+4)'
    # see documentation for more options. https://openpiv.readthedocs.io/en/latest/src/windef.html 

    'Region of interest'
    # (50,300,50,300) #Region of interest: (xmin,xmax,ymin,ymax) or 'full' for full image
    settings.roi = 'full'

    'Processing Parameters'
    settings.correlation_method='linear' #'circular' or 'linear'
    settings.normalized_correlation = True

    settings.num_iterations = 3  # select the number of PIV passes
    # add the interrogation window size for each pass. 
    # For the moment, it should be a power of 2 

    settings.windowsizes=(64, 32, 24, 8)
    settings.overlap=(32, 16, 12, 4)

    # settings.windowsizes = (128, 64, 32, 16, 8) # if longer than n iteration the rest is ignored
    # The overlap of the interroagtion window for each pass.
    # settings.overlap = (64, 32, 16, 8, 4) # This is 50% overlap

    settings.show_plot = True
    settings.scale_plot = 200  # select a value to scale the quiver plot of the vectorfield
    # run the script with the given settings

    settings.show_all_plots = False
    return (settings,)


@app.cell
def _(settings, windef):
    windef.piv(settings)
    return


if __name__ == "__main__":
    app.run()
