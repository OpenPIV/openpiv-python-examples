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
    # Run the window deformation algorithm
    """)
    return


@app.cell
def _():
    import pathlib

    import numpy as np
    import openpiv.pyprocess as process
    from openpiv import filters, scaling, tools, validation, windef


    # '%matplotlib inline' command supported automatically in marimo
    return filters, np, pathlib, process, scaling, tools, validation, windef


@app.cell
def _(pathlib, windef):
    settings = windef.PIVSettings()


    'Data related settings'
    # Folder with the images to process
    settings.filepath_images = pathlib.Path("test1/")
    # Folder for the outputs
    settings.save_path = settings.filepath_images
    # Root name of the output Folder for Result Files
    settings.save_folder_suffix = 'Test_1'
    # Format and Image Sequence
    settings.frame_pattern_a = 'exp1_001_a.bmp'
    settings.frame_pattern_b = 'exp1_001_b.bmp'

    'Region of interest'
    # (50,300,50,300) #Region of interest: (xmin,xmax,ymin,ymax) or 'full' for full image
    settings.roi = 'full'

    'Image preprocessing'
    # 'None' for no flaging, 'edges' for edges flaging, 'intensity' for intensity flaging
    # WARNING: This part is under development so better not to use flagS
    settings.dynamic_masking_method = 'None'
    settings.dynamic_masking_threshold = 0.005
    settings.dynamic_masking_filter_size = 7

    settings.deformation_method = 'symmetric'

    'Processing Parameters'
    settings.correlation_method='circular'  # 'circular' or 'linear'
    settings.normalized_correlation=False

    settings.num_iterations = 3  # select the number of PIV passes
    # add the interroagtion window size for each pass. 
    # For the moment, it should be a power of 2 
    settings.windowsizes = (64, 32, 16) # if longer than n iteration the rest is ignored, rectangular windows are arrange as [y,x]
    # The overlap of the interroagtion window for each pass.
    settings.overlap = (34, 16, 8) # This is 50% overlap
    # Has to be a value with base two. In general window size/2 is a good choice.
    # methode used for subpixel interpolation: 'gaussian','centroid','parabolic'
    settings.subpixel_method = 'gaussian'
    # use vectorized sig2noise and subpixel approximation functions for speed
    settings.use_vectorized = False
    # order of the image interpolation for the window deformation
    settings.interpolation_order = 3
    settings.scaling_factor = 1  # scaling factor pixel/meter
    settings.dt = 1  # time between to frames (in seconds)
    'Signal to noise ratio options (only for the last pass)'
    # It is possible to decide if the S/N should be computed (for the last pass) or not
    # settings.extract_sig2noise = True  # 'True' or 'False' (only for the last pass)
    # method used to calculate the signal to noise ratio 'peak2peak' or 'peak2mean'
    settings.sig2noise_method = 'peak2peak'
    # select the width of the flaged to flaged out pixels next to the main peak
    settings.sig2noise_mask = 2
    # If extract_sig2noise==False the values in the signal to noise ratio
    # output column are set to NaN
    'vector validation options'
    # choose if you want to do validation of the first pass: True or False
    settings.validation_first_pass = True
    # only effecting the first pass of the interrogation the following passes
    # in the multipass will be validated
    'Validation Parameters'
    # The validation is done at each iteration based on three filters.
    # The first filter is based on the min/max ranges. Observe that these values are defined in
    # terms of minimum and maximum displacement in pixel/frames.
    settings.min_max_u_disp = (-30, 30)
    settings.min_max_v_disp = (-30, 30)
    # The second filter is based on the global STD threshold
    settings.std_threshold = 7  # threshold of the std validation
    # The third filter is the median test (not normalized at the moment)
    settings.median_threshold = 3  # threshold of the median validation
    # On the last iteration, an additional validation can be done based on the S/N.
    settings.median_size=1 #defines the size of the local median
    'Validation based on the signal to noise ratio'
    # Note: only available when extract_sig2noise==True and only for the last
    # pass of the interrogation
    # Enable the signal to noise ratio validation. Options: True or False
    # settings.do_sig2noise_validation = False # This is time consuming
    # minmum signal to noise ratio that is need for a valid vector
    settings.sig2noise_threshold = 1.2
    'Outlier replacement or Smoothing options'
    # Replacment options for vectors which are flaged as invalid by the validation
    settings.replace_vectors = True # Enable the replacment. Chosse: True or False
    settings.smoothn=True #Enables smoothing of the displacemenet field
    settings.smoothn_p=0.5 # This is a smoothing parameter
    # select a method to replace the outliers: 'localmean', 'disk', 'distance'
    settings.filter_method = 'localmean'
    # maximum iterations performed to replace the outliers
    settings.max_filter_iteration = 4
    settings.filter_kernel_size = 2  # kernel size for the localmean method
    'Output options'
    # Select if you want to save the plotted vectorfield: True or False
    settings.save_plot = False
    # Choose wether you want to see the vectorfield or not :True or False
    #settings.show_plot = True
    settings.scale_plot = 200  # select a value to scale the quiver plot of the vectorfield
    # run the script with the given settings
    return (settings,)


@app.cell
def _(settings, windef):
    windef.piv(settings)
    return


@app.cell
def _():
    #Run the extended search area PIV
    return


@app.cell
def _(filters, np, process, scaling, settings, tools, validation):
    # we can run it from any folder
    path = settings.filepath_images


    frame_a  = tools.imread( path / settings.frame_pattern_a )
    frame_b  = tools.imread( path / settings.frame_pattern_b )

    frame_a = (frame_a).astype(np.int32)
    frame_b = (frame_b).astype(np.int32)

    u, v, sig2noise = process.extended_search_area_piv( frame_a, frame_b, \
        window_size=32, overlap=16, dt=1, search_area_size=64, sig2noise_method="peak2peak")
    x, y = process.get_coordinates( image_size=frame_a.shape, 
                                   search_area_size=64, overlap=16 )

    flag_s = validation.sig2noise_val( sig2noise, threshold = 1.3 )
    flag_g = validation.global_val( u, v, (-1000, 2000), (-1000, 1000) )
    flag = flag_s | flag_g

    u, v = filters.replace_outliers( u, v, flag, method='localmean', max_iter=10, kernel_size=2)
    x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor = 1)
    x, y, u, v = tools.transform_coordinates(x, y, u, v)

    tools.save('test1.vec', x, y, u, v, flag)
    tools.display_vector_field('test1.vec', scale=75, width=0.0035)
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
