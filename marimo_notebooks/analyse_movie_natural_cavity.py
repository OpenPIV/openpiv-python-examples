# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "openpiv==0.26.3",
#     "numpy",
#     "matplotlib",
#     "scipy",
#     "scikit-image",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    # we need to read frames from the movie
    # so we install opencv-python - change the next cell type to "Code"
    return


@app.cell
def _():
    # !pip install opencv-python-headless
    return


@app.cell
def _():
    import cv2

    return (cv2,)


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    return np, plt


@app.cell
def _():
    from openpiv import pyprocess, piv

    return (piv,)


@app.cell
def _(cv2, piv):
    vidcap = cv2.VideoCapture('/home/user/Downloads/cavity.mov')
    # vidcap = cv2.VideoCapture('../test_movie/Manikin_Thermal_Plume.MOV')
    success, image1 = vidcap.read()
    count = 0
    U = []
    V = []
    while success:
        success, image2 = vidcap.read()
        if success:  # cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
            x, y, u, v, s2n = piv.simple_piv(image1.sum(axis=2), image2.sum(axis=2), plot=False)
            image1 = image2.copy()  # print('Read a new frame: ', success)
            count = count + 1
            U.append(u)
            V.append(v)
    return U, V, image1, x, y


@app.cell
def _(U, V, np):
    U_1 = np.stack(U)
    Umean = np.nanmean(U_1, axis=0)
    V_1 = np.stack(V)
    Vmean = np.nanmean(V_1, axis=0)
    return Umean, Vmean


@app.cell
def _(Umean, Vmean, image1, np, plt, x, y):
    fig,ax = plt.subplots(figsize=(12,24))
    ax.imshow(image1,cmap='gray')
    cm = ax.quiver(x,y.max()-y,Umean,Vmean,np.abs(Vmean),scale=90,width=.008)
    # plt.show()
    # plt.plot(x[10,:], np.nanmean(Vmean[:10],axis=0)*100+200,color='r',lw=3)
    # plt.plot(np.nanmean(Umean,axis=1)*2+50,y[:,5],lw=3,color='k')
    plt.title('Quiver and U(y) profile')
    plt.colorbar(cm, orientation='horizontal');
    return


if __name__ == "__main__":
    app.run()
