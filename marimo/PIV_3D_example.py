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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    by Andreas Bauer and David Boehringer, 19.06.2020
    This script contains two examples for 3D-PIV: The shift of a bar of binary pixels in one direction, the expansion
    and a real data set where we recorded two stacks of collagen fibres at the same field of view with confocal microscopy
     in reflection mode.  One stack contains a NK cell that deforms the matrix and the other doe not.
    Please download the data at https://github.com/fabrylab/3D_piv_example_data.git (180 MB, unpacked) and provide the
    folder in the code below.
    We tested this on ubuntu 16 and 18, with Anaconda Python installation. The whole script
    takes about 5 minutes on my 4 core-intel i5 @2.5 GHz Laptop. You should have !!! 8 Gb ob Memory !!!! or take care not
    to open  all matplotlib plots as interactive windows at once.
    For questions contact andreas.b.bauer@fau.de
    """)
    return


app._unparsable_cell(
    r"""
    from openpiv.pyprocess3D import *
    from openpiv.PIV_3D_plotting import *
    from openpiv.validation import sig2noise_val
    from openpiv.filters import replace_outliers
    from openpiv.lib import replace_nans
    import glob as glob
    import os
    from natsort import natsorted
    import matplotlib.animation as animation
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Make save_plots = True if you want to compare the
    visual results
    """)
    return


@app.cell
def _(os):
    save_plots = False
    out_put_folder = "output_3D_test"
    if save_plots:
        if not os.path.exists(out_put_folder):
            try:
                os.mkdir(out_put_folder)
            except:
                print("could not generate output folder")
                save_plots = False
    return out_put_folder, save_plots


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ############ a group of bars shifted by 1 pixel to the each dimesion the second frame #############
    takes ~4 seconds
    """)
    return


@app.cell
def _(np):
    # constructing frame 1 and frame 2
    size = (32, 32, 32)
    shape1 = np.zeros(size)
    shape2 = np.zeros(size)
    return shape1, shape2


@app.cell
def _(shape1):
    shape1[16, 16, 25:27] = 1
    shape1[16, 16, 7:9] = 1
    shape1[16, 25:27, 16] = 1
    shape1[16, 7:9, 16] = 1
    shape1[25:27, 16, 16] = 1
    shape1[7:9, 16, 16] = 1
    return


@app.cell
def _(shape2):
    shape2[16, 16, 24:26] = 1
    shape2[16, 16, 8:10] = 1
    shape2[16, 24:26, 16] = 1
    shape2[16, 8:10, 16] = 1
    shape2[24:26, 16, 16] = 1
    shape2[8:10, 16, 16] = 1
    return


@app.cell
def _():
    window_size = (4, 4, 4)
    overlap = (3, 3, 3)
    search_area = (5, 5, 5)
    return overlap, search_area, window_size


@app.cell
def _(
    extended_search_area_piv3D,
    overlap,
    search_area,
    shape1,
    shape2,
    window_size,
):
    u, v, w, sig2noise = extended_search_area_piv3D(shape1, shape2, window_size=window_size, overlap=overlap,
                                                    search_area_size=search_area, subpixel_method='gaussian',
                                                    sig2noise_method='peak2peak',
                                                    width=2)
    return sig2noise, u, v, w


@app.cell
def _(quiver_3D, scatter_3D, shape1, shape2, sig2noise, u, v, w):
    # magic command not supported in marimo; please file an issue to add support
    # %pdb 
    # displaying the shapes with 3D scatter plot
    fig1 = scatter_3D(shape1, control="size")
    fig2 = scatter_3D(shape2, control="size")
    # 3d plot of the signal-to-noise rations
    fig3 = scatter_3D(sig2noise, control="size")
    # 3d quiver plot of the displacement field
    fig4 = quiver_3D(-u, v, w, cmap="coolwarm", quiv_args={"arrow_length_ratio":0.6})
    return fig1, fig2, fig3, fig4


@app.cell
def _(fig1, fig2, fig3, fig4, os, out_put_folder, save_plots):
    # saving the plots
    if save_plots:
        fig1.savefig(os.path.join(out_put_folder, "displaced_bar_frame1.png"))
        fig2.savefig(os.path.join(out_put_folder, "displaced_bar_frame2.png"))
        fig3.savefig(os.path.join(out_put_folder, "displaced_bar_sig2noise.png"))
        fig4.savefig(os.path.join(out_put_folder, "displaced_bar_deformation_field.png"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ################### test to check the replace_nans_function ######################
    takes ~4 seconds
    """)
    return


@app.cell
def _(np):
    # ball shape with a gap of nans in the middle
    center = (5, 5, 5)
    size_1 = (10, 10, 10)
    distance = np.linalg.norm(np.subtract(np.indices(size_1).T, np.asarray(center)), axis=len(center))
    arr = np.ones(size_1) * (distance <= 5)
    hide = arr == 0
    arr[5:7] = np.nan
    return arr, hide


@app.cell
def _(arr, hide, np, replace_nans, scatter_3D):
    # displaying in 3d plots. Values outside of the original ball are hidden by setting to nan
    arr_show = arr.copy()
    arr_show[hide] = np.nan
    fig9 = scatter_3D(arr_show, size=50, sca_args={'alpha': 0.6})
    # replacing outliers
    arr_1 = replace_nans(arr, max_iter=2, tol=2, kernel_size=2, method="disk")
    return arr_1, fig9


@app.cell
def _(arr_1, hide, np, scatter_3D):
    # displaying in 3d plots. Values outside of the original ball are hidden by setting to nan
    arr_show_1 = arr_1.copy()
    arr_show_1[hide] = np.nan
    fig10 = scatter_3D(arr_show_1, size=50, sca_args={'alpha': 0.6})
    return (fig10,)


@app.cell
def _(fig10, fig9, os, out_put_folder, save_plots):
    # saving the plots
    if save_plots:
        fig9.savefig(os.path.join(out_put_folder, "replace_nan_gap.png"))
        fig10.savefig(os.path.join(out_put_folder, "replace_nan_filled.png"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #################### real data example ############################
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    we recorded stacks of collagen fibres with confocal microscopy in reflection mode
    "alive" stack contains a force generating NK-cell, marked by the red circle in the animation
    "relaxed" stack is the same field of view with out the cell
    download the data at https://github.com/fabrylab/3D_piv_example_data.git
    this calculation takes ~ 3-4 minutes on my 4-core Intel i5@2.5 GHz Laptop
    """)
    return


@app.cell
def _():
    # please enter the path to the dataset provided at
    folder = r"test_3d"
    return (folder,)


@app.cell
def _(folder, os):
    if not os.path.exists(folder): 
        import git 
        repo = git.Repo.clone_from("https://github.com/fabrylab/3D_piv_example_data.git", './test_3d', branch="master")
    return


@app.cell
def _(folder, os):
    if not os.path.exists(folder):
        raise FileNotFoundError("path to 3d piv data '%s' does not exists\n"
                            ". Please download the data from https://github.com/fabrylab/3D_piv_example_data.git" % folder)
    # stack properties
    # factors for voxel size
    du = 0.2407
    dv = 0.2407
    dw = 1.0071
    # total image dimension for x y z
    image_dim = (123.02, 123.02, 122.86)
    return du, dv, dw


@app.cell
def _():
    # keep these values for our nk cells stacks
    win_um = 12  # window size in µm
    fac_overlap = 0.3  # overlap in percent of the window size
    signoise_filter = 1.3
    return fac_overlap, signoise_filter, win_um


@app.cell
def _(du, dv, dw, fac_overlap, win_um):
    # window size for stacks in pixel
    window_size_1 = (int(win_um / du), int(win_um / dv), int(win_um / dw))
    overlap_1 = (int(fac_overlap * win_um / du), int(fac_overlap * win_um / dv), int(fac_overlap * win_um / dw))
    search_area_1 = (int(win_um / du), int(win_um / dv), int(win_um / dw))
    return overlap_1, search_area_1, window_size_1


@app.cell
def _(folder, glob, natsorted, np, os, plt):
    # load tense stacks
    images = natsorted(glob.glob(os.path.join(folder, "Series001_t22_z*_ch00.tif")))
    im_shape = plt.imread(images[0]).shape
    alive = np.zeros((im_shape[0], im_shape[1], len(images)))
    for i, im in enumerate(images):
        alive[:, :, i] = plt.imread(im)
    return (alive,)


@app.cell
def _(folder, glob, natsorted, np, os, plt):
    # load relaxed stack
    images_1 = natsorted(glob.glob(os.path.join(folder, 'Series003_t05_z*_ch00.tif')))
    im_shape_1 = plt.imread(images_1[0]).shape
    relax = np.zeros((im_shape_1[0], im_shape_1[1], len(images_1)))
    for i_1, im_1 in enumerate(images_1):
        relax[:, :, i_1] = plt.imread(im_1)
    return (relax,)


@app.cell
def _(
    alive,
    du,
    dv,
    dw,
    extended_search_area_piv3D,
    overlap_1,
    relax,
    search_area_1,
    window_size_1,
):
    # 3D PIV
    u_1, v_1, w_1, sig2noise_1 = extended_search_area_piv3D(relax, alive, window_size=window_size_1, overlap=overlap_1, search_area_size=search_area_1, dt=(1 / du, 1 / dv, 1 / dw), subpixel_method='gaussian', sig2noise_method='peak2peak', width=2)
    return sig2noise_1, u_1, v_1, w_1


@app.cell
def _(np, u_1, v_1, w_1):
    # correcting stage drift between the field of views
    u_2 = u_1 - np.nanmean(u_1)
    v_2 = v_1 - np.nanmean(v_1)
    w_2 = w_1 - np.nanmean(w_1)
    return u_2, v_2, w_2


@app.cell
def _(
    replace_outliers,
    sig2noise_1,
    sig2noise_val,
    signoise_filter,
    u_2,
    v_2,
    w_2,
):
    # filtering
    mask = sig2noise_val(sig2noise_1, threshold=signoise_filter)
    uf, vf, wf = replace_outliers(u_2, v_2, mask, w=w_2, max_iter=1, tol=100, kernel_size=2, method="disk")
    return uf, vf, wf


@app.cell
def _(plt):
    # plotting
    # representation of the image stacks by maximums projections. The red circle marks the position of the cell
    def update_plot(i, ims, ax):
        a1 = ax.imshow(ims[i])
        a2 = ax.add_patch(plt.Circle((330, 140), 100, color="red", fill=False))
        return [a1, a2]

    return (update_plot,)


@app.cell
def _(alive, animation, np, plt, relax, update_plot):
    ims = [np.max(relax[:, :, 60:], axis=2), np.max(alive[:, :, 60:], axis=2)]
    fig = plt.figure()
    ax = plt.gca()
    ani = animation.FuncAnimation(fig, update_plot, 2, interval=200, blit=True, repeat_delay=0, fargs=(ims, ax))
    return (ims,)


@app.cell
def _(quiver_3D, u_2, v_2, w_2):
    # unfiltered 3d deformation field
    fig11 = quiver_3D(-u_2, v_2, w_2, quiv_args={'length': 2, 'alpha': 0.8, 'linewidth': 1}, filter_def=0.1)
    return (fig11,)


@app.cell
def _(quiver_3D, uf, vf, wf):
    # filtered 3d deformation field
    fig12 = quiver_3D(-uf, vf, wf, quiv_args={"length": 2, "alpha": 0.8, "linewidth": 1},
                      filter_def=0.1)
    return (fig12,)


@app.cell
def _(fig11, fig12, ims, os, out_put_folder, plt, save_plots):
    # saving the plots
    if save_plots:
        fig11.savefig(os.path.join(out_put_folder, "real_data_unfiltered.png"))
        fig12.savefig(os.path.join(out_put_folder, "real_data_filtered.png"))

        # This needs a working ImageMagick installation, and probably works only on linux
        try:
            import imageio
            plt.ioff()
            f1 = plt.figure()
            plt.imshow(ims[0])
            plt.gca().add_artist(plt.Circle((330, 140), 100, color="red", fill=False))
            f1.savefig(os.path.join(out_put_folder,"tem1.png"))

            f2 = plt.figure()
            plt.imshow(ims[1])
            plt.gca().add_artist(plt.Circle((330, 140), 100, color="red", fill=False))
            f2.savefig(os.path.join(out_put_folder,"tem2.png"))

            i1 = plt.imread(os.path.join(out_put_folder,"tem1.png"))
            i2 = plt.imread(os.path.join(out_put_folder, "tem2.png"))
            imageio.mimsave(os.path.join(out_put_folder, "reaL_data_max_proj.gif"),[i1,i2], fps=1)
            os.remove(os.path.join(out_put_folder,"tem1.png"))
            os.remove(os.path.join(out_put_folder,"tem2.png"))
            plt.ion()
        except Exception as e:
            print ("failed to write gif of collagen embedded cell:")
            print(e)
    return


if __name__ == "__main__":
    app.run()
