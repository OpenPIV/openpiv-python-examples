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
    #Run the new window deformation
    return


@app.cell
def _():
    from time import time
    start = time()
    return start, time


@app.cell
def _():
    new_windef = True
    return (new_windef,)


@app.cell
def _():
    import pathlib

    # '%matplotlib inline' command supported automatically in marimo
    import matplotlib
    from openpiv import windef
    matplotlib.rcParams['figure.figsize'] = (8.0, 8.0)
    return pathlib, windef


@app.cell
def _(new_windef, pathlib, windef):
    settings = windef.PIVSettings()


    'Data related settings'
    # Folder with the images to process
    settings.filepath_images = pathlib.Path("test10/")
    # Folder for the outputs
    settings.save_path = pathlib.Path("test1/")
    # Root name of the output Folder for Result Files
    settings.save_folder_suffix = 'Test_1'
    # Format and Image Sequence
    settings.frame_pattern_a = 'B001_1.tif'
    settings.frame_pattern_b = 'B001_2.tif'

    'Region of interest'
    # (50,300,50,300) #Region of interest: (xmin,xmax,ymin,ymax) or 'full' for full image
    settings.roi = 'full'

    'Image preprocessing'
    # 'None' for no masking, 'edges' for edges masking, 'intensity' for intensity masking
    # WARNING: This part is under development so better not to use MASKS
    settings.dynamic_masking_method = 'edges'
    settings.dynamic_masking_threshold = 0.005
    settings.dynamic_masking_filter_size = 7

    # settings.deformation_method = 'symmetric' #'second image' #'symmetric' # or 'second image'

    'Processing Parameters'
    settings.correlation_method='linear' #'circular' or 'linear'
    settings.normalized_correlation = True


    settings.num_iterations = 4  # select the number of PIV passes
    # add the interrogation window size for each pass. 
    # For the moment, it should be a power of 2 

    settings.windowsizes=(64, 32, 24, 8)
    settings.overlap=(32, 16, 12, 4)

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
    'Signal to noise ratio options (only for the last pass)'
    # It is possible to decide if the S/N should be computed (for the last pass) or not
    if not new_windef:
        settings.extract_sig2noise = True  # 'True' or 'False' (only for the last pass)
    
    # method used to calculate the signal to noise ratio 'peak2peak' or 'peak2mean'
    settings.sig2noise_method = 'peak2mean'
    # select the width of the masked to masked out pixels next to the main peak
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
    settings.std_threshold = 4  # threshold of the std validation
    # The third filter is the median test (not normalized at the moment)
    settings.median_threshold = 3  # threshold of the median validation
    # On the last iteration, an additional validation can be done based on the S/N.
    settings.median_size = 3 #defines the size of the local median
    'Validation based on the signal to noise ratio'
    # Note: only available when extract_sig2noise==True and only for the last
    # pass of the interrogation
    # Enable the signal to noise ratio validation. Options: True or False
    # settings.do_sig2noise_validation = False # This is time consuming
    # minmum signal to noise ratio that is need for a valid vector
    settings.sig2noise_threshold = 1.5
    'Outlier replacement or Smoothing options'
    # Replacment options for vectors which are masked as invalid by the validation
    settings.replace_vectors = True # Enable the replacment. Chosse: True or False
    settings.smoothn = False #Enables smoothing of the displacemenet field
    settings.smoothn_p = 0.5 # This is a smoothing parameter
    # select a method to replace the outliers: 'localmean', 'disk', 'distance'
    settings.filter_method = 'localmean'
    # maximum iterations performed to replace the outliers
    settings.max_filter_iteration = 4
    settings.filter_kernel_size = 2  # kernel size for the localmean method
    'Output options'
    # Select if you want to save the plotted vectorfield: True or False
    settings.save_plot = False
    # Choose wether you want to see the vectorfield or not :True or False
    settings.show_plot = True
    settings.scale_plot = 200  # select a value to scale the quiver plot of the vectorfield
    # run the script with the given settings

    settings.show_all_plots = False
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
def _(settings, windef):
    settings.frame_pattern_a = 'B002_1.tif'
    settings.frame_pattern_b = 'B002_2.tif'
    windef.piv(settings)
    return


@app.cell
def _(settings, windef):
    settings.frame_pattern_a = 'B003_1.tif'
    settings.frame_pattern_b = 'B003_2.tif'
    windef.piv(settings)
    return


@app.cell
def _(settings, windef):
    settings.frame_pattern_a = 'B004_1.tif'
    settings.frame_pattern_b = 'B004_2.tif'
    windef.piv(settings)
    return


@app.cell
def _(settings, windef):
    settings.frame_pattern_a = 'B005_1.tif'
    settings.frame_pattern_b = 'B005_2.tif'
    windef.piv(settings)
    return


@app.cell
def _(settings, windef):
    settings.frame_pattern_a = 'B006_1.tif'
    settings.frame_pattern_b = 'B006_2.tif'
    windef.piv(settings)
    return


@app.cell
def _():
    from openpiv.piv import simple_piv

    return (simple_piv,)


@app.cell
def _(settings, simple_piv):
    files = sorted(settings.filepath_images.glob("*.tif"))
    from openpiv.tools import imread
    simple_piv(imread(files[0]), imread(files[1]))
    return


@app.cell
def _(start, time):
    end = time()
    print(end - start)
    return


if __name__ == "__main__":
    app.run()
