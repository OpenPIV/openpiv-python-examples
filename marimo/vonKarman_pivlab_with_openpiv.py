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


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %reload_ext watermark
    # magic command not supported in marimo; please file an issue to add support
    # %watermark -v -m -p numpy,openpiv
    return


@app.cell
def _():
    import os

    # '%matplotlib inline' command supported automatically in marimo
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    from openpiv import filters, preprocess, scaling, smoothn, tools, validation, windef
    from openpiv.preprocess import mask_coordinates
    matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)
    return (
        filters,
        mask_coordinates,
        np,
        os,
        plt,
        preprocess,
        scaling,
        smoothn,
        tools,
        validation,
        windef,
    )


@app.cell
def _(windef):
    settings = windef.PIVSettings()

    # 'Data related settings'
    # Folder with the images to process
    settings.filepath_images = "test9/"
    # Folder for the outputs
    settings.save_path = '.'
    # Root name of the output Folder for Result Files
    settings.save_folder_suffix = ''
    # Format and Image Sequence
    settings.frame_pattern_a = 'karman_16Hz_000_A.jpg'
    settings.frame_pattern_b = 'karman_16Hz_000_B.jpg'

    'Region of interest'
    # (50,300,50,300) #Region of interest: (xmin,xmax,ymin,ymax) or 'full' for full image
    settings.roi = 'full'
    # settings.roi = (200,400,600,850)



    settings.deformation_method = 'symmetric' # or 'second image'


    settings.num_iterations = 4  # select the number of PIV passes

    # add the interrogation window size for each pass. 
    # For the moment, it should be a power of 2 
    settings.windowsizes=(32, 16, 8, 6)
    settings.overlap=(16, 8, 4, 3)

    # settings.windowsizes = (128, 64, 32, 16, 8) # if longer than n iteration the rest is ignored
    # The overlap of the interroagtion window for each pass.
    # settings.overlap = (64, 32, 16, 8, 4) # This is 50% overlap


    # Has to be a value with base two. In general window size/2 is a good choice.
    # methode used for subpixel interpolation: 'gaussian','centroid','parabolic'
    settings.subpixel_method = 'gaussian'

    # order of the image interpolation for the window deformation
    settings.interpolation_order = 1
    settings.scaling_factor = 1  # scaling factor pixel/meter
    settings.dt = 1  # time between to frames (in seconds)
    'Signal to noise ratio options (only for the last pass)'
    # It is possible to decide if the S/N should be computed (for the last pass) or not
    # settings.extract_sig2noise = True  # 'True' or 'False' (only for the last pass)
    # method used to calculate the signal to noise ratio 'peak2peak' or 'peak2mean'
    settings.sig2noise_method = 'peak2peak'
    # select the width of the masked to masked out pixels next to the main peak
    settings.sig2noise_mask = 2
    # If extract_sig2noise==False the values in the signal to noise ratio
    # output column are set to NaN

    # only effecting the first pass of the interrogation the following passes
    # in the multipass will be validated

    'Output options'
    # Select if you want to save the plotted vectorfield: True or False
    settings.save_plot = False
    # Choose wether you want to see the vectorfield or not :True or False
    settings.show_plot = True
    settings.scale_plot = 200  # select a value to scale the quiver plot of the vectorfield
    # run the script with the given settings



    # 'Processing Parameters'
    settings.correlation_method='linear'  # 'circular' or 'linear'
    settings.normalized_correlation = True

    # 'vector validation options'
    # choose if you want to do validation of the first pass: True or False
    settings.validation_first_pass = True


    settings.filter_method = 'localmean'
    # maximum iterations performed to replace the outliers
    settings.max_filter_iteration = 10
    settings.filter_kernel_size = 3  # kernel size for the localmean method

    settings.replace_vectors = True

    settings.min_max_u_disp = (-5, 5)
    settings.min_max_v_disp = (-5, 5)

    # The second filter is based on the global STD threshold
    settings.std_threshold = 3  # threshold of the std validation

    # The third filter is the median test (not normalized at the moment)
    settings.median_threshold = 3  # threshold of the median validation
    # On the last iteration, an additional validation can be done based on the S/N.
    settings.median_size=1 #defines the size of the local median, it'll be 3 x 3


    settings.dynamic_masking_method = 'intensity'
    settings.dynamic_masking_threshold = 0.1
    settings.dynamic_masking_filter_size = 21
    return (settings,)


@app.cell
def _(settings):
    vars(settings)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Read and crop the images
    """)
    return


@app.cell
def _(os, settings, tools):
    file_a = settings.frame_pattern_a
    file_b = settings.frame_pattern_b

    # " read images into numpy arrays"
    frame_a = tools.imread(os.path.join(settings.filepath_images, file_a))
    frame_b = tools.imread(os.path.join(settings.filepath_images, file_b))

    # " crop to ROI"
    if settings.roi == "full":
        pass
    else:
        frame_a = frame_a[
            settings.roi[0]:settings.roi[1],
            settings.roi[2]:settings.roi[3]
        ]
        frame_b = frame_b[
            settings.roi[0]:settings.roi[1],
            settings.roi[2]:settings.roi[3]
        ]
    return frame_a, frame_b


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Show the images
    """)
    return


@app.cell
def _(frame_a, plt):
    plt.imshow(frame_a,cmap=plt.cm.gray)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Image masking
    """)
    return


@app.cell
def _(frame_a, frame_b, plt, preprocess, settings):
    # 'Image preprocessing'
    # 'None' for no masking, 'edges' for edges masking, 'intensity' for intensity masking
    # WARNING: This part is under development so better not to use MASKS
    if settings.dynamic_masking_method == 'edge' or 'intensity':
        frame_a_1, image_mask_a = preprocess.dynamic_masking(frame_a, method=settings.dynamic_masking_method, filter_size=settings.dynamic_masking_filter_size, threshold=settings.dynamic_masking_threshold)
        frame_b_1, image_mask_b = preprocess.dynamic_masking(frame_b, method=settings.dynamic_masking_method, filter_size=settings.dynamic_masking_filter_size, threshold=settings.dynamic_masking_threshold)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(frame_a_1)
    ax[1].imshow(frame_b_1)
    return frame_a_1, frame_b_1, image_mask_a, image_mask_b


@app.cell
def _(image_mask_a, image_mask_b, np, plt):
    # let's combine the two masks if the body is slightly moving
    image_mask = np.logical_and(image_mask_a, image_mask_b)
    plt.imshow(image_mask)
    return (image_mask,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exract coordinates of the mask as a list of coordinates of a polygon
    """)
    return


@app.cell
def _(image_mask, mask_coordinates):
    mask_coords = mask_coordinates(image_mask)
    return (mask_coords,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run the first pass

    We use typically the most robust approach: linear correlation (with zero padding)
        and normalized correlation function (0..1)
    """)
    return


@app.cell
def _(frame_a_1, frame_b_1, np, settings, windef):
    # In order to convert the image mask to the data mask in x,y 
    # coordinates, we have to either run first pass or 
    # use get_coordinates
    # Since we do not know how to use the image_mask in the 
    # first pass with the vectorized correlations, i.e. how to 
    # save some computational time by skipping the interrogation
    # windows within the image mask, we just run the first pass
    x, y, u, v, sig2noise_ratio = windef.first_pass(frame_a_1, frame_b_1, settings)
    u0 = u.copy()
    # "first pass"
    v0 = v.copy()

    def status_message(u):
        print(f"{np.isnan(u).sum() / u.size * 100:.2f}% invalid vectors out of {u.size} vectors")
    # store for the comparison of the following steps
    status_message(u)
    return sig2noise_ratio, status_message, u, u0, v, v0, x, y


@app.cell
def _(image_mask, mask_coords, np, plt, x, y):
    # Now we can convert the image mask to the data mask in x,y coordinates

    from skimage.measure import points_in_poly

    # mark those points on the grid of PIV inside the mask
    xymask = points_in_poly(np.c_[y.flatten(),x.flatten()],mask_coords)

    plt.imshow(~image_mask,cmap=plt.cm.gray)
    plt.plot(x.flat[xymask],y.flat[xymask],"x")
    return (xymask,)


@app.cell
def _(np, u, v, x, xymask):
    # mask the velocity maps for the future use in validation
    tmp = np.zeros_like(x, dtype=bool)
    tmp.flat[xymask] = True
    u_1 = np.ma.masked_array(u, mask=tmp)
    v_1 = np.ma.masked_array(v, mask=tmp)
    return u_1, v_1


@app.cell
def _(plt, sig2noise_ratio, u_1, v_1, x, xymask, y):
    # we need to remove those values for the display
    def quick_quiver():
        """ u,v expected to have a mask """
        plt.quiver(x, y, u_1, -v_1, sig2noise_ratio, scale=50, color="b")
        plt.gca().invert_yaxis()
        plt.gca().set_aspect(1)
        plt.plot(x.flat[xymask], y.flat[xymask], "rx")
        plt.colorbar(orientation="horizontal")

    return (quick_quiver,)


@app.cell
def _(quick_quiver):
    quick_quiver()
    return


@app.cell
def _(plt, sig2noise_ratio):
    # see the distribution of the signal to noise ratio
    tmp_1 = sig2noise_ratio.copy()
    tmp_1[tmp_1 > 10] = 10  # there are some extra high values 1e7 ...
    plt.imshow(tmp_1)
    plt.colorbar(orientation="horizontal")
    return (tmp_1,)


@app.cell
def _(plt, tmp_1):
    plt.hist(tmp_1.flatten())
    return


@app.cell
def _(np, settings, sig2noise_ratio):
    # let's consider 5% of signoise ratio problems. 
    sig2noise_threshold = np.percentile(sig2noise_ratio,(2.5))
    print(f"S2N threshold is estimated as {sig2noise_threshold:.3f}")

    settings.sig2noise_threshold = 1.2
    return


@app.cell
def _(settings, sig2noise_ratio, status_message, u_1, validation):
    mask_s2n = validation.sig2noise_val(sig2noise_ratio, threshold=settings.sig2noise_threshold)
    status_message(u_1)
    return (mask_s2n,)


@app.cell
def _(mask_s2n, plt, sig2noise_ratio, u0, u_1, v0, v_1, x, y):
    plt.figure()
    plt.quiver(x, y, u_1, -v_1, sig2noise_ratio)
    plt.quiver(x[mask_s2n], y[mask_s2n], u0[mask_s2n], -v0[mask_s2n], color="r")
    plt.gca().invert_yaxis()
    plt.gca().set_aspect(1.0)
    plt.colorbar(orientation="horizontal")
    return


@app.cell
def _(np, x):
    # False everywhere, all passes
    outliers_mask = np.zeros_like(x, dtype=bool)
    return


@app.cell
def _(plt, v_1):
    plt.hist(v_1.flatten())
    return


@app.cell
def _(settings, status_message, u_1, v_1, validation):
    # 'Validation Parameters'
    # The validation is done at each iteration based on three filters.
    # The first filter is based on the min/max ranges. Observe that these values are defined in
    # terms of minimum and maximum displacement in pixel/frames.
    mask_g = validation.global_val(u_1, v_1, settings.min_max_u_disp, settings.min_max_v_disp)
    status_message(u_1)
    return (mask_g,)


@app.cell
def _(mask_g, plt, sig2noise_ratio, u0, u_1, v0, v_1, x, y):
    plt.figure()
    plt.quiver(x, y, u_1, -v_1, sig2noise_ratio)
    plt.quiver(x[mask_g], y[mask_g], u0[mask_g], -v0[mask_g], color="r")
    plt.gca().invert_yaxis()
    plt.gca().set_aspect(1.0)
    plt.colorbar(orientation="horizontal")
    return


@app.cell
def _():
    ## also global std should take masked array
    return


@app.cell
def _(settings, status_message, u_1, v_1, validation):
    # The second filter is based on the global STD threshold
    settings.std_threshold = 5  # threshold of the std validation
    mask_s = validation.global_std(u_1, v_1, std_threshold=settings.std_threshold)
    status_message(u_1)
    return (mask_s,)


@app.cell
def _(mask_s, plt, sig2noise_ratio, u0, u_1, v0, v_1, x, y):
    plt.figure()
    plt.quiver(x, y, u_1, -v_1, sig2noise_ratio)
    plt.quiver(x[mask_s], y[mask_s], u0[mask_s], -v0[mask_s], color="r")
    plt.gca().invert_yaxis()
    plt.gca().set_aspect(1.0)
    plt.colorbar(orientation="horizontal")
    return


@app.cell
def _():
    ## validation.local_median_val should also take masked array
    return


@app.cell
def _(settings, status_message, u_1, v_1, validation):
    # The third filter is the median test (not normalized at the moment)
    settings.median_threshold = 3  # threshold of the median validation
    # On the last iteration, an additional validation can be done based on the S/N.
    settings.median_size = 1  #defines the size of the local median
    mask_m = validation.local_median_val(u_1, v_1, u_threshold=settings.median_threshold, v_threshold=settings.median_threshold, size=settings.median_size)
    status_message(u_1)
    return (mask_m,)


@app.cell
def _(mask_m, plt, sig2noise_ratio, u0, u_1, v0, v_1, x, y):
    plt.figure()
    plt.quiver(x, y, u_1, -v_1, sig2noise_ratio)
    plt.quiver(x[mask_m], y[mask_m], u0[mask_m], -v0[mask_m], color="r")
    plt.gca().invert_yaxis()
    plt.gca().set_aspect(1.0)
    plt.colorbar(orientation="horizontal")
    return


@app.cell
def _(mask_g, mask_m, mask_s, mask_s2n):
    # Combining masks
    outliers_mask_1 = mask_g + mask_m + mask_s + mask_s2n
    return (outliers_mask_1,)


@app.cell
def _(outliers_mask_1, plt, sig2noise_ratio, u0, u_1, v0, v_1, x, y):
    plt.figure()
    plt.quiver(x, y, u_1, -v_1, sig2noise_ratio)
    plt.quiver(x[outliers_mask_1], y[outliers_mask_1], u0[outliers_mask_1], -v0[outliers_mask_1], color="r")
    plt.gca().invert_yaxis()
    plt.gca().set_aspect(1.0)
    plt.colorbar(orientation="horizontal")
    return


@app.cell
def _(status_message, u_1):
    status_message(u_1)
    return


@app.cell
def _(filters, outliers_mask_1, settings, u_1, v_1):
    # "filter to replace the values that where marked by the validation"
    # if settings.num_iterations > 1:
    u_2, v_2 = filters.replace_outliers(u_1, v_1, outliers_mask_1, method=settings.filter_method, max_iter=settings.max_filter_iteration, kernel_size=settings.filter_kernel_size)
    return u_2, v_2


@app.cell
def _(np, u_2, v_2, x, xymask):
    # mask the velocity maps
    tmp_2 = np.zeros_like(x, dtype=bool)
    tmp_2.flat[xymask] = 1
    u_3 = np.ma.masked_array(u_2, mask=tmp_2)
    v_3 = np.ma.masked_array(v_2, mask=tmp_2)
    return u_3, v_3


@app.cell
def _(quick_quiver):
    quick_quiver()
    return


@app.cell
def _(np, settings, smoothn, u_3, v_3, x, xymask):
    # Smoothing after the first pass
    settings.smoothn = True  #Enables smoothing of the displacemenet field
    settings.smoothn_p = 0.5  # This is a smoothing parameter
    u_4, dummy_u1, dummy_u2, dummy_u3 = smoothn.smoothn(u_3, s=settings.smoothn_p)
    v_4, dummy_v1, dummy_v2, dummy_v3 = smoothn.smoothn(v_3, s=settings.smoothn_p)
    tmp_3 = np.zeros_like(x, dtype=bool)
    tmp_3.flat[xymask] = 1
    u_4 = np.ma.masked_array(u_4, mask=tmp_3)
    # mask the velocity maps
    v_4 = np.ma.masked_array(v_4, mask=tmp_3)
    return u_4, v_4


@app.cell
def _(plt, sig2noise_ratio, u0, u_4, v0, v_4, x, xymask, y):
    # x, y, u, v = tools.transform_coordinates(x, y, u, v)
    plt.figure()
    plt.quiver(x, y, u0, -v0, color='r', scale=30, alpha=0.5)
    plt.quiver(x, y, u_4, -v_4, sig2noise_ratio, scale=30)
    plt.plot(x.flat[xymask], y.flat[xymask], "ro")
    plt.gca().invert_yaxis()
    plt.colorbar(orientation="horizontal")
    plt.gca().set_aspect(1.0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multi-pass loop with window deformation, validation and smoothing

    **Note**: no smoothing on the last step
    """)
    return


@app.cell
def _(frame_a_1, frame_b_1, settings, u_4, v_4, windef, x, y):
    # TODO: study the sig2noise validation in multipass
    settings.sig2noise_validate = False
    for i in range(1, settings.num_iterations):
        x_1, y_1, u_5, v_5, sig2noise_ratio_1, mask = windef.multipass_img_deform(frame_a_1, frame_b_1, i, x, y, u_4, v_4, settings)  ## all other passes
    return mask, u_5, v_5, x_1, y_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Save the outcome
    """)
    return


@app.cell
def _(mask, os, plt, scaling, settings, tools, u_5, v_5, x_1, y_1):
    save_path = '.'
    counter = 0
    u_6 = u_5 / settings.dt
    # "pixel/frame->pixel/sec"
    v_6 = v_5 / settings.dt
    x_2, y_2, u_6, v_6 = scaling.uniform(x_1, y_1, u_6, v_6, scaling_factor=settings.scaling_factor)
    x_2, y_2, u_6, v_6 = tools.transform_coordinates(x_2, y_2, u_6, v_6)
    # "scales the results pixel-> meter"
    tools.save(os.path.join(save_path, 'field_A%03d.txt' % counter), x_2, y_2, u_6, v_6, mask, delimiter="\t")
    settings.show_plot = True
    settings.save_plot = True
    if settings.show_plot is True or settings.save_plot is True:
        plt.close("all")
    # "save to a file"
        plt.ioff()
        filename = os.path.join(save_path, 'Image_A%03d.png' % counter)
        tools.display_vector_field(os.path.join(save_path, 'field_A%03d.txt' % counter), scale=settings.scale_plot)
        if settings.save_plot is True:
            plt.savefig(filename)
        if settings.show_plot is True:
            plt.show()
    # "some other stuff that one might want to use"
    print('Image Pair ' + str(counter + 1))
    return (counter,)


@app.cell
def _(counter, plt):
    import glob

    import xarray as xr
    from pivpy import io
    file_list = sorted(glob.glob('field_A%03d.txt' % counter))
    print(file_list)
    data = []
    frame = 0
    for f in file_list:
        data.append(io.load_openpiv_txt(f, frame=frame))
        frame = frame + 1
    data = xr.concat(data, dim="t")
    data.attrs['units'] = ['pix', 'pix', 'pix/dt', 'pix/dt']
    data.piv.vorticity()

    def plot_data(data):
        fig, ax = plt.subplots(1, 1, figsize=(20, 12))
        ax.quiver(data.x.data, data.y.data, data.u.isel(t=0).data.T, data.v.isel(t=0).data.T, color='r', scale=120)
        s = ax.pcolor(data.x, data.y, data.w.T.isel(t=0), shading='interp', vmin=-0.3, vmax=0.3, alpha=0.7)
        ax.set_aspect(1)
        fig.colorbar(s, ax=ax)
        plt.show()
    plot_data(data)  # for ax in axs:
    return


if __name__ == "__main__":
    app.run()
