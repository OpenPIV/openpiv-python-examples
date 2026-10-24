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
    import imageio
    from openpiv.piv import simple_piv

    return imageio, simple_piv


@app.cell
def _(imageio, simple_piv):
    # gifurl = 'https://64.media.tumblr.com/15d6395f97f2d12e32a764c4a17be406/699471e89e1d5634-11/s500x750/eabb9c7c1ea719d4b9889d8e0217a878ed3f7a3f.gifv'

    gifurl = 'https://64.media.tumblr.com/3decdb9824c82cc625396d5162b9c72c/tumblr_ohqkj1wMvh1qckzoqo2_500.gifv'

    im = imageio.get_reader(gifurl)
    # print(im)

    from openpiv.tools import rgb2gray

    images = []
    for frame in im:
        images.append(rgb2gray(frame))

    for I,J in zip(images[25:-1],images[26:]):
        simple_piv(I,J)
    return


if __name__ == "__main__":
    app.run()
