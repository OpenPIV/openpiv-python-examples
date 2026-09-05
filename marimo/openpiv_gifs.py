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
    list_of_gifs = ['01','02','03','04','05','06','07','08','21','22','23','24','41']
    base_path = 'http://www.vsj.jp/~pivstd/gif/'
    base_name = 'image'
    base_ext = '.gif'

    for gif in list_of_gifs: 
        print("Reading ..")
        print(f"{base_path}{base_name}{gif}{base_ext}")
        im = imageio.get_reader(f"{base_path}{base_name}{gif}{base_ext}")
        # print(im)

        images = []
        for frame in im:
            if frame.ndim > 2:
                frame = frame[:,:,0]
            
            images.append(frame)
    
        # images = np.array(images)

        # plt.figure()
        # plt.imshow(np.c_[images[0],images[1]])

        # for I,J in zip(images[:-1],images[1:]):
        #     simple_piv(I,J)

        # let's do only one pair 
        simple_piv(images[0], images[1])
    return


if __name__ == "__main__":
    app.run()
