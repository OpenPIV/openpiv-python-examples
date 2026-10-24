# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "openpiv>=0.26.0",
#     "numpy",
#     "matplotlib",
#     "imageio",
#     "scipy",
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


@app.cell
def _():
    # we need to read frames from the movie
    # so we install opencv-python - change the next cell type to "Code"
    return


@app.cell
def _():
    # !pip install opencv-python
    return


@app.cell
def _():
    import cv2

    return (cv2,)


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams["figure.figsize"] = (12, 12)

    from skimage import exposure
    from skimage import img_as_ubyte

    return exposure, img_as_ubyte, np, plt


@app.cell
def _():
    from openpiv import pyprocess, piv, validation, tools, filters, scaling

    return filters, pyprocess, scaling, tools, validation


@app.cell
def _(exposure, img_as_ubyte, np):
    def as_grey(frame):
        red = frame[:, :, 0]
        green = frame[:, :, 1]
        blue = frame[:, :, 2]
        im = np.ceil(0.2125 * red + 0.7154 * green + 0.0721 * blue).astype(np.uint8)
        im = exposure.equalize_adapthist(im,clip_limit=1.2)
    
        return img_as_ubyte(im)

    return (as_grey,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Video source:

    https://www.youtube.com/watch?v=EeS1rYMZUxI&ab_channel=USUExperimentalFluidDynamicsLab
    """)
    return


@app.cell
def _(cv2):
    # the video is the jet PIV from Youtube
    # https://www.youtube.com/watch?v=EeS1rYMZUxI&ab_channel=USUExperimentalFluidDynamicsLab
    # all the rights reserved to the authors

    vidcap = cv2.VideoCapture("data/test_movie/videoplayback.mp4")
    success, image1 = vidcap.read()
    skip = 3
    for i in range(skip):
        success, image2 = vidcap.read()
    # count = 0
    # U = []
    # V = []

    # plt.figure(figsize=(12,12))

    # while success and count < 1:
    #     # cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
    #     success, image2 = vidcap.read()
    #     # print('Read a new frame: ', success)
    #     if success:
    #         x,y,u,v = piv.simple_piv(image1.sum(axis=2), image2.sum(axis=2),plot=True);
    #         # image1 = image2.copy()
    #         count += 1
    #         U.append(u)
    #         V.append(v)
    return image1, image2


@app.cell
def _(as_grey, image1, image2, np):
    frame_a = as_grey(image1).astype(np.int32)
    frame_b = as_grey(image2).astype(np.int32)
    return frame_a, frame_b


@app.cell
def _(
    as_grey,
    filters,
    frame_a,
    frame_b,
    image1,
    pyprocess,
    scaling,
    tools,
    validation,
):
    winsize = 64 # pixels, interrogation window size in frame A
    searchsize = 128  # pixels, search in image B
    overlap = 32 # pixels, 50% overlap
    dt = 1 # sec, time interval between pulses


    u0, v0, sig2noise = pyprocess.extended_search_area_piv(frame_a, 
                                                           frame_b, 
                                                           window_size=winsize, 
                                                           overlap=overlap, 
                                                           dt=dt, 
                                                           search_area_size=searchsize, 
                                                           sig2noise_method="peak2peak")

    x, y = pyprocess.get_coordinates( image_size=frame_a.shape, 
                                     search_area_size=searchsize, 
                                     overlap=overlap )

    mask = validation.sig2noise_val( sig2noise, 
                                     threshold = 1.0)
    # if you need more detailed look, first create a histogram of sig2noise
    # plt.hist(sig2noise.flatten(),bins=100)
    # to see where is a reasonable limit

    # filter out outliers that are very different from the
    # neighbours

    u2, v2 = filters.replace_outliers( u0, v0, mask,
                                      method='localmean', 
                                      max_iter=3, 
                                      kernel_size=3)
    # convert x,y to mm
    # convert u,v to mm/sec

    x, y, u3, v3 = scaling.uniform(x, y, u2, v2, 
                                   scaling_factor = 1.0 ) 

    # 0,0 shall be bottom left, positive rotation rate is counterclockwise
    x, y, u3, v3 = tools.transform_coordinates(x, y, u3, v3)

    tools.save("data/test16/night_lava_test.txt" , x, y, u3, v3, mask )

    tools.imsave("data/test16/bkg.png",as_grey(image1))
    return x, y


@app.cell
def _(plt, tools):
    _fig, _ax = plt.subplots()
    tools.display_vector_field("data/test16/night_lava_test.txt", ax=_ax, scaling_factor=1, scale=1000, width=0.0035, on_img=True, image_name="data/test16/bkg.png")  # scale defines here the arrow length  # width is the thickness of the arrow  # overlay on the image
    return


@app.cell
def _(np):
    U = np.stack(U)
    Umean = np.mean(U, axis=0)
    V = np.stack(V)
    Vmean = np.mean(V,axis=0)
    return U, Umean, V, Vmean


@app.cell
def _(Umean, Vmean, image1, np, plt, x, y):
    _fig, _ax = plt.subplots(figsize=(8, 8))
    _ax.imshow(image1, alpha=0.7)
    _ax.quiver(x, y, Umean, Vmean, scale=200, color='r', width=0.008)
    # plt.show()
    plt.plot(np.mean(Umean, axis=1) * 20, y[:, 0], color='y', lw=3)
    return


@app.cell
def _(frame_a, frame_b, np, plt):
    from skimage.registration import optical_flow_ilk
    v, u = optical_flow_ilk(frame_a, frame_b, radius=15)
    norm = np.sqrt(u ** 2 + v ** 2)
    _fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))
    ax0.imshow(frame_a, cmap="gray")
    ax0.set_title("Sequence image sample")
    ax0.set_axis_off()
    nvec = 20
    nl, nc = frame_a.shape
    step = max(nl // nvec, nc // nvec)
    y_1, x_1 = np.mgrid[:nl:step, :nc:step]
    u_ = u[::step, ::step]
    v_ = v[::step, ::step]
    ax1.imshow(norm)
    ax1.quiver(x_1, y_1, u_, v_, color='r', units='dots', angles='xy', scale_units='xy', lw=3)
    ax1.set_title("Optical flow magnitude and vector field")
    ax1.set_axis_off()
    _fig.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
