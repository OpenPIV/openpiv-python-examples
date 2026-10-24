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
    # Linear vs Circular, Normalized vs non-normalized correlations
    """)
    return


@app.cell
def _():
    # we should test linear vs circular to understand the differences
    # the main difference in the new version was a bug as 
    # the zero-padding for linear case does not work as long as you leave values on the 
    # borders not zero. 
    # see new normalize_intensity function that should take care of uneven illumination
    # both between the frames and at different regions of the image A or B
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    # from skimage.feature.phase_cross_correlation import _upsampled_dft
    from scipy.ndimage import fourier_shift
    from skimage.exposure import match_histograms
    from skimage.registration import phase_cross_correlation

    return fourier_shift, match_histograms, np, phase_cross_correlation, plt


app._unparsable_cell(
    r"""
    from openpiv.tools import imread
    from pylab import *
    """,
    name="_"
)


@app.cell
def _():

    return


@app.cell
def _():
    from openpiv.pyprocess import fft_correlate_images, find_subpixel_peak_position

    return fft_correlate_images, find_subpixel_peak_position


@app.cell
def _(fourier_shift, imread, match_histograms, np):
    a = imread("data/test11/A001_1.tif")
    # b = imread("data/PIVChallenge2001_A/A001_2.tif")
    a = a[:32,:32].copy()
    a[16:18,16:18] = 255
    # b = b[:32,:32]

    # should be in the order of y,x:
    # so it's about 5 pixels upwards and about 3 pixels to the right
    shift = (-12.035, -10.92)
    # The shift corresponds to the pixel offset relative to the reference image
    b = fourier_shift(np.fft.fftn(a), shift)
    b = np.fft.ifftn(b).real
    b = match_histograms(b,a).astype("uint8")
    # b = b + np.linspace(10,85,32)
    return a, b, shift


@app.cell
def _(a, colorbar, imshow):
    imshow(a)
    colorbar()
    return


@app.cell
def _(b, colorbar, imshow):
    imshow(b)
    colorbar()
    return


@app.cell
def _(a, b, np):
    # like moving window with the 0th dimension the IW no.
    a_1 = a[np.newaxis, :, :]
    b_1 = b[np.newaxis, :, :]
    return a_1, b_1


@app.cell
def _(a_1, b_1, fft_correlate_images):
    # %%timeit 
    c1 = fft_correlate_images(a_1, b_1, 'circular', normalized_correlation=False)
    return (c1,)


@app.cell
def _(a_1, b_1, fft_correlate_images):
    # %%timeit
    c2 = fft_correlate_images(a_1, b_1, 'linear', normalized_correlation=False)
    return (c2,)


@app.cell
def _(a_1, b_1, fft_correlate_images):
    # %%timeit
    c3 = fft_correlate_images(a_1, b_1, 'circular', normalized_correlation=True)
    return (c3,)


@app.cell
def _(a_1, b_1, fft_correlate_images):
    # %%timeit
    c4 = fft_correlate_images(a_1, b_1, 'linear', normalized_correlation=True)
    return (c4,)


@app.cell
def _(c1, c2, c3, c4, colorbar, find_subpixel_peak_position, np, plt, shift):
    fig, ax = plt.subplots(1, 4, figsize=(14, 2.5))
    counter = 0
    for c in [c1, c2, c3, c4]:
        s = ax[counter].contourf(c[0, :, :])
        ax[counter].invert_yaxis()
        colorbar(s, ax=ax[counter])
        default_peak_position = np.floor(np.array(c[0, :, :].shape) / 2)
        i = np.array(find_subpixel_peak_position(c[0, :, :]))
        ax[counter].plot(i[1], i[0], "rx")
        print(np.array(i - default_peak_position), np.sum(np.abs(np.array(i - default_peak_position) - np.array(shift))))
        counter = counter + 1
    return


@app.cell
def _(a_1, b_1, np, phase_cross_correlation, plt):
    image = a_1[0, :, :]
    offset_image = b_1[0, :, :]
    # pixel precision first
    shift_1, error, diffphase = phase_cross_correlation(image, offset_image)
    fig_1 = plt.figure(figsize=(8, 3))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
    ax3 = plt.subplot(1, 3, 3)
    ax1.imshow(image, cmap="gray")
    ax1.set_axis_off()
    ax1.set_title("Reference image")
    ax2.imshow(offset_image.real, cmap="gray")
    ax2.set_axis_off()
    ax2.set_title("Offset image")
    image_product = np.fft.fft2(image).conj() * np.fft.fft2(offset_image)
    cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
    ax3.imshow(cc_image.real)
    ax3.set_axis_off()
    # Show the output of a cross-correlation to show what the algorithm is
    # doing behind the scenes
    ax3.set_title("Cross-correlation")
    plt.show()
    print('Detected pixel offset (y, x): {}'.format(shift_1))
    shift_1, error, diffphase = phase_cross_correlation(offset_image, image, upsample_factor=1000)
    fig_1 = plt.figure(figsize=(8, 3))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
    ax1.imshow(image, cmap="gray")
    ax1.set_axis_off()
    ax1.set_title("Reference image")
    # subpixel precision
    ax2.imshow(offset_image.real, cmap="gray")
    ax2.set_axis_off()
    ax2.set_title("Offset image")
    plt.show()
    # ax3 = plt.subplot(1, 3, 3)
    # Calculate the upsampled DFT, again to show what the algorithm is doing
    # behind the scenes.  Constants correspond to calculated values in routine.
    # See source code for details.
    # cc_image = _upsampled_dft(image_product, 150, 100, (shift*100)+75).conj()
    # ax3.imshow(cc_image.real)
    # ax3.set_axis_off()
    # ax3.set_title("Supersampled XC sub-area")
    print('Detected subpixel offset (y, x): {}'.format(shift_1))
    return


if __name__ == "__main__":
    app.run()
