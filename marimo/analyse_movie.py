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

# /// script
# dependencies = ["opencv-python-headless"]
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
    # Test OpenPIV on a MP4 video clip
    """)
    return


@app.cell
def _():
    # we need to read frames from the movie
    # so we install opencv-python - change the next cell type to "Code"
    return


@app.cell
def _():
    # packages added via marimo's package management: opencv-python-headless !pip install opencv-python-headless
    return


@app.cell
def _():
    import cv2

    return (cv2,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    return np, plt


@app.cell
def _():
    from openpiv import pyprocess

    return (pyprocess,)


@app.cell
def _(cv2, pyprocess):
    # the video is the jet PIV from Youtube
    # https://www.youtube.com/watch?v=EeS1rYMZUxI&ab_channel=USUExperimentalFluidDynamicsLab
    # all the rights reserved to the authors
    vidcap = cv2.VideoCapture("data/test_movie/videoplayback.mp4")
    success, image1 = vidcap.read()
    count = 0
    U = []
    V = []
    while success and count < 10:
        success, image2 = vidcap.read()
        if success:
            u, v, s2n = pyprocess.extended_search_area_piv(image1.sum(axis=2), image2.sum(axis=2), window_size=64, overlap=32)
            x, y = pyprocess.get_coordinates(image1.shape[:2], search_area_size=64, overlap=32)
            image1 = image2.copy()  # cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
            count = count + 1
            U.append(u)  # print('Read a new frame: ', success)
            V.append(v)
    return U, V, image1, x, y


@app.cell
def _():
    from IPython.display import HTML
    HTML("""
        <video alt="test" controls>
            <source src="../test_movie/videoplayback.mp4" type="video/mp4">
        </video>
    """)
    return


@app.cell
def _(U, V, np):
    U_1 = np.stack(U)
    Umean = np.mean(U_1, axis=0)
    V_1 = np.stack(V)
    Vmean = np.mean(V_1, axis=0)
    return Umean, Vmean


@app.cell
def _(Umean, Vmean, image1, np, plt, x, y):
    fig,ax = plt.subplots(figsize=(8,16))
    ax.imshow(image1,alpha=0.7)
    Q= ax.quiver(x,y,Umean,Vmean,Umean**2+Vmean**2,scale=50, width=.007)
    # plt.show()
    plt.plot(np.mean(Umean,axis=1)*30,y[:,0],color='r',lw=3)
    plt.colorbar(Q, orientation="horizontal")
    return


if __name__ == "__main__":
    app.run()
