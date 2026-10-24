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
    # compare the OpenPIV Python with PIVLab
    """)
    return


app._unparsable_cell(
    r"""
    Analysis of the Karman images
    final int area 6 pixels and 50% overlap, 
    vector validation is allowed, but no smoothing after the last correlation. 
    Only the circle in the middle must be masked, not the shadows.

    Then we can compare the vorticity maps (color bar scale of uncalibrated data -0.3 1/frame until +0.3 1/frame, 
    color map preferably "parula", but "jet" is also ok). That might give an idea about the "quality"...?
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    FFT window deformation
    Pass1: 64x64 px with 50% overlap
    Pass2: 32x32 px with 50% overlap
    Pass3: 16x16 px with 50% overlap
    Pass4: 6x6 px with 50% overlap
    Gauss2x3-point subpixel estimator
    Correlation quality: Extreme
    """,
    name="_"
)


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %reload_ext watermark
    # magic command not supported in marimo; please file an issue to add support
    # %watermark -v -m -p numpy,openpiv
    return


@app.cell
def _():
    # '%matplotlib inline' command supported automatically in marimo
    import matplotlib
    from openpiv import windef
    matplotlib.rcParams['figure.figsize'] = (8.0, 6.0)
    return (windef,)


@app.cell
def _(windef):
    settings = windef.PIVSettings()

    # 'Data related settings'
    # Folder with the images to process
    settings.filepath_images = "test9/"
    # Folder for the outputs
    settings.save_path = "test9/results/"
    # Root name of the output Folder for Result Files
    settings.save_folder_suffix = 'Test_1'
    # Format and Image Sequence
    settings.frame_pattern_a = 'karman_16Hz_000_A.jpg'
    settings.frame_pattern_b = 'karman_16Hz_000_B.jpg'

    'Region of interest'
    # (50,300,50,300) #Region of interest: (xmin,xmax,ymin,ymax) or 'full' for full image
    settings.roi = 'full'
    # settings.roi = (200,400,500,900)

    # settings.deformation_method = 'symmetric' 
    settings.deformation_method = 'second image'


    settings.num_iterations = 4  # select the number of PIV passes

    # add the interrogation window size for each pass. 
    # For the moment, it should be a power of 2 
    settings.windowsizes=(64, 32, 16, 6)
    settings.overlap=(32, 16, 8, 3)

    # settings.windowsizes = (128, 64, 32, 16, 8) # if longer than n iteration the rest is ignored
    # The overlap of the interroagtion window for each pass.
    # settings.overlap = (64, 32, 16, 8, 4) # This is 50% overlap


    # Has to be a value with base two. In general window size/2 is a good choice.
    # methode used for subpixel interpolation: 'gaussian','centroid','parabolic'
    settings.subpixel_method = 'gaussian'

    # order of the image interpolation for the window deformation
    settings.interpolation_order = 3
    settings.scaling_factor = 1  # scaling factor pixel/meter
    settings.dt = 1  # time between to frames (in seconds)

    # 'Signal to noise ratio options (only for the last pass)'
    # It is possible to decide if the S/N should be computed (for the last pass) or not
    # settings.extract_sig2noise = True  # 'True' or 'False' (only for the last pass)
    settings.sig2noise_threshold = 1.25
    # method used to calculate the signal to noise ratio 'peak2peak' or 'peak2mean'
    settings.sig2noise_method = 'peak2peak'
    # select the width of the masked to masked out pixels next to the main peak
    settings.sig2noise_mask = 2
    settings.sig2noise_validate = False

    # If extract_sig2noise==False the values in the signal to noise ratio
    # output column are set to NaN

    # only effecting the first pass of the interrogation the following passes
    # in the multipass will be validated

    'Output options'
    # Select if you want to save the plotted vectorfield: True or False
    settings.save_plot = False
    # Choose wether you want to see the vectorfield or not :True or False
    settings.show_plot = True
    settings.scale_plot = 100  # select a value to scale the quiver plot of the vectorfield
    # run the script with the given settings



    # 'Processing Parameters'
    settings.correlation_method='circular'  # 'circular' or 'linear'
    settings.normalized_correlation = False

    # 'vector validation options'
    # choose if you want to do validation of the first pass: True or False
    settings.validation_first_pass = True


    settings.filter_method = 'localmean'


    settings.replace_vectors = True
    # maximum iterations performed to replace the outliers
    settings.max_filter_iteration = 4
    settings.filter_kernel_size = 2  # kernel size for the localmean method



    settings.min_max_u_disp = (-10, 10)
    settings.min_max_v_disp = (-10, 10)

    # The second filter is based on the global STD threshold
    settings.std_threshold = 5  # threshold of the std validation

    # The third filter is the median test (not normalized at the moment)
    settings.median_threshold = 3  # threshold of the median validation
    # On the last iteration, an additional validation can be done based on the S/N.
    settings.median_size = 2 # defines the size of the local median, it'll be 3 x 3

    # Image mask properties
    settings.dynamic_masking_method = 'intensity'
    settings.dynamic_masking_threshold = 0.1
    settings.dynamic_masking_filter_size = 21



    # Smoothing after the first pass
    settings.smoothn = False #Enables smoothing of the displacemenet field
    settings.smoothn_p = 0.5 # This is a smoothing parameter


    settings.show_all_plots = True
    return (settings,)


@app.cell
def _():
    import glob
    file_list = sorted(glob.glob("data/karman_16Hz_*.jpg"))
    file_list = file_list[-2:]
    file_list
    return


@app.cell
def _(settings, windef):
    # magic command not supported in marimo; please file an issue to add support
    # %time 
    windef.piv(settings)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %debug 
    return


@app.cell
def _():
    # !pip install git+https://github.com/alexlib/pivpy
    from pivpy import io

    return (io,)


@app.cell
def _(io):
    data = io.load_openpiv_txt("data/test9/results/OpenPIV_results_6_Test_1/field_A0000.txt")
    data.piv.vorticity()
    data.piv.quiver()
    return



@app.cell
def _():
    import marimo as mo
    mo.md(r"""### OpenPIV 0.26.0: `scipy.fft` and Rust backends in `windef`
`PIVSettings(backend="scipy"|"rust"|"auto")` controls the FFT engine. `auto` (default) prefers Rust when installed.
""")
    return

@app.cell
def _(frame_a, frame_b, windef):
    import importlib.metadata
    print("openpiv", importlib.metadata.version("openpiv"))
    from openpiv.pyprocess import HAS_RUST
    print("HAS_RUST =", HAS_RUST)
    settings = windef.PIVSettings()
    for backend in (["scipy", "rust"] if HAS_RUST else ["scipy"]):
        settings.backend = backend
        print(f"\\n--- backend={backend} ---")
        from openpiv.pyprocess import extended_search_area_piv
        u, v, s2n = extended_search_area_piv(frame_a, frame_b, window_size=settings.windowsizes[0], overlap=settings.overlap[0], sig2noise_method="peak2peak", backend=backend)
        print(f"single-pass mean u={u.mean():.2f}, v={v.mean():.2f}")
    return

if __name__ == "__main__":
    app.run()
