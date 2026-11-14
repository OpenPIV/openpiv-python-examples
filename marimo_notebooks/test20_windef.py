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
    # Run the window deformation algorithm
    """)
    return


@app.cell
def _():
    from openpiv import windef
    from openpiv import tools, scaling, validation, filters, preprocess
    import openpiv.pyprocess as process
    from openpiv import pyprocess
    import numpy as np
    import pathlib
    from time import time
    import warnings


    import matplotlib.pyplot as plt
    # '%matplotlib inline' command supported automatically in marimo
    return pathlib, windef


@app.cell
def _(pathlib, windef):
    settings = windef.PIVSettings()


    # 'Data related settings'
    # Folder with the images to process
    settings.filepath_images = pathlib.Path('../test20/')
    # Folder for the outputs
    settings.save_path = settings.filepath_images
    # Root name of the output Folder for Result Files
    settings.save_folder_suffix = 'test_20'
    # Format and Image Sequence
    settings.frame_pattern_a = 't_23.png'
    settings.frame_pattern_b = 't_24.png'

    # 'Region of interest'
    # (50,300,50,300) #Region of interest: (xmin,xmax,ymin,ymax) or 'full' for full image
    settings.ROI = 'full'

    # 'Image preprocessing'
    # 'None' for no flaging, 'edges' for edges flaging, 'intensity' for intensity flaging
    # WARNING: This part is under development so better not to use flagS
    settings.dynamic_flaging_method = 'None'
    settings.dynamic_flaging_threshold = 0.005
    settings.dynamic_flaging_filter_size = 7

    settings.deformation_method = 'symmetric'

    # 'Processing Parameters'
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
    settings.sig2noise_flag = 2
    # If extract_sig2noise==False the values in the signal to noise ratio
    # output column are set to NaN

    # 'vector validation options'
    # choose if you want to do validation of the first pass: True or False
    settings.validation_first_pass = True
    # only effecting the first pass of the interrogation the following passes
    # in the multipass will be validated

    # 'Validation Parameters'

    # The validation is done at each iteration based on three filters.
    # The first filter is based on the min/max ranges. Observe that these values are defined in
    # terms of minimum and maximum displacement in pixel/frames.
    settings.MinMax_U_disp = (-30, 30)
    settings.MinMax_V_disp = (-30, 30)
    # The second filter is based on the global STD threshold
    settings.std_threshold = 7  # threshold of the std validation
    # The third filter is the median test (not normalized at the moment)
    settings.median_threshold = 3  # threshold of the median validation
    # On the last iteration, an additional validation can be done based on the S/N.
    settings.median_size=1 #defines the size of the local median

    #  'Validation based on the signal to noise ratio'
    # Note: only available when extract_sig2noise==True and only for the last
    # pass of the interrogation
    # Enable the signal to noise ratio validation. Options: True or False
    # settings.do_sig2noise_validation = False # This is time consuming
    # minmum signal to noise ratio that is need for a valid vector
    settings.sig2noise_threshold = 1.2

    # 'Outlier replacement or Smoothing options'
    # Replacment options for vectors which are flaged as invalid by the validation
    settings.replace_vectors = True # Enable the replacment. Chosse: True or False
    settings.smoothn=True #Enables smoothing of the displacemenet field
    settings.smoothn_p=0.5 # This is a smoothing parameter
    # select a method to replace the outliers: 'localmean', 'disk', 'distance'
    settings.filter_method = 'localmean'
    # maximum iterations performed to replace the outliers
    settings.max_filter_iteration = 4
    settings.filter_kernel_size = 2  # kernel size for the localmean method

    # 'Output options'
    # Select if you want to save the plotted vectorfield: True or False
    settings.save_plot = False
    settings.show_all_plots = True
    # Choose wether you want to see the vectorfield or not :True or False
    settings.show_plot = True
    settings.scale_plot = 20  # select a value to scale the quiver plot of the vectorfield
    # run the script with the given settings
    return (settings,)


@app.cell
def _(settings, windef):
    windef.piv(settings)
    return


if __name__ == "__main__":
    app.run()
