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
    import pathlib

    # '%matplotlib inline' command supported automatically in marimo
    import matplotlib
    from openpiv import windef
    matplotlib.rcParams['figure.figsize'] = (8.0, 8.0)
    return pathlib, windef


@app.cell
def _(pathlib, windef):
    settings = windef.PIVSettings()


    'Data related settings'
    # Folder with the images to process
    settings.filepath_images = pathlib.Path("test10/")
    # Folder for the outputs
    settings.save_path = pathlib.Path("test10/")
    # Root name of the output Folder for Result Files
    settings.save_folder_suffix = 'Test_1'
    # Format and Image Sequence
    settings.frame_pattern_a = 'B001_1.tif'
    settings.frame_pattern_b = 'B001_2.tif'




    settings.num_iterations = 3  # select the number of PIV passes
    settings.windowsizes=(128, 64, 32)
    settings.overlap=(64, 32, 16)

    settings.show_plot = True
    return (settings,)


@app.cell
def _(settings, windef):
    windef.piv(settings)
    return


@app.cell
def _(settings, windef):
    settings.num_iterations = 4  # select the number of PIV passes
    settings.windowsizes=(128, 64, 32, 16)
    settings.overlap=(64, 32, 16, 8)
    windef.piv(settings)
    return


@app.cell
def _(settings, windef):
    settings.num_iterations = 3  # select the number of PIV passes
    settings.windowsizes=(128, 64, 32, 16)
    settings.overlap=(64, 32, 16, 8)
    windef.piv(settings)
    return


if __name__ == "__main__":
    app.run()
