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

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    mo.md(r"""
# Test case: test14

This page demonstrates the **test14** dataset/case from the OpenPIV examples repository.

## What this case is for
- Explore the images and data included in `test14/`.
- Learn how to run PIV on this case and what to look for (seeding, displacement, noise, masks, etc.).
- Use it as a starting point: adjust window size, overlap, dt, and validation for your needs.

**[Open in molab (interactive)](https://molab.marimo.io/github/OpenPIV/openpiv-python-examples/blob/main/marimo_cases/test14.py)**

**Files (sample):** disp_096a.bmp, disp_096a.bmp.txt, disp_096b.bmp, disp_097a.bmp, disp_097b.bmp, disp_098a.bmp

*Note: when running in molab/cloud, this notebook downloads the case files from GitHub automatically.*
""")
    return (mo,)


@app.cell
def __():
    import os, glob, urllib.request
    case_name = 'test14'
    base_url = 'https://raw.githubusercontent.com/OpenPIV/openpiv-python-examples/main' + "/" + case_name
    case_files = ['disp_096a.bmp', 'disp_096a.bmp.txt', 'disp_096b.bmp', 'disp_097a.bmp', 'disp_097b.bmp', 'disp_098a.bmp', 'disp_098b.bmp', 'disp_099a.bmp', 'disp_099b.bmp', 'disp_100a.bmp', 'disp_100b.bmp', 'disp_101a.bmp', 'disp_101b.bmp', 'disp_102a.bmp', 'disp_102b.bmp', 'disp_103a.bmp', 'disp_103b.bmp', 'disp_104a.bmp', 'disp_104b.bmp', 'disp_105a.bmp', 'disp_105b.bmp', 'disp_106a.bmp', 'disp_106b.bmp', 'disp_107a.bmp', 'disp_107b.bmp', 'disp_108a.bmp', 'disp_108b.bmp', 'disp_109a.bmp', 'disp_109b.bmp', 'disp_110a.bmp', 'disp_110b.bmp', 'disp_111a.bmp', 'disp_111b.bmp', 'disp_112a.bmp', 'disp_112b.bmp', 'disp_113a.bmp', 'disp_113b.bmp', 'disp_114a.bmp', 'disp_114b.bmp', 'disp_115a.bmp', 'disp_115b.bmp', 'disp_116a.bmp', 'disp_116b.bmp', 'disp_117a.bmp', 'disp_117b.bmp', 'disp_118a.bmp', 'disp_118b.bmp', 'disp_119a.bmp', 'disp_119b.bmp', 'disp_120a.bmp']
    os.makedirs(case_name, exist_ok=True)
    for name in case_files:
        dest = os.path.join(case_name, name)
        if not os.path.exists(dest):
            try:
                urllib.request.urlretrieve(base_url + "/" + name, dest)
            except Exception as e:
                print("skip", name, e)
    file_list = sorted(glob.glob(case_name + "/*"))
    print("Files in case:")
    for fpath in file_list[:30]:
        print(" -", os.path.basename(fpath))
    return case_name


@app.cell
def __(mo):
    mo.md("## Load and display images")
    return


@app.cell
def __(case_name):
    import matplotlib.pyplot as plt1
    from openpiv import tools
    from os.path import join, exists
    img_a_path = 'disp_096a.bmp'
    img_b_path = 'disp_096b.bmp'
    img_a = tools.imread(join(case_name, img_a_path)) if img_a_path and exists(join(case_name, img_a_path)) else None
    img_b = tools.imread(join(case_name, img_b_path)) if img_b_path and exists(join(case_name, img_b_path)) else None
    if img_a is not None and img_b is not None:
        fig1, ax1 = plt1.subplots(1, 2, figsize=(8, 4))
        ax1[0].imshow(img_a, cmap="gray"); ax1[0].set_title(img_a_path)
        ax1[1].imshow(img_b, cmap="gray"); ax1[1].set_title(img_b_path)
        display_fig = fig1
    else:
        display_fig = "No image pair found to display."
    display_fig
    return img_a, img_b


@app.cell
def __(mo):
    mo.md(r"""
## Run PIV (basic example)
Adjust parameters based on this case: window size, overlap, search area, and `dt`.
- **Tip:** ensure particle displacement is ~5–10 px between frames.
- **Tip:** inspect signal-to-noise and validate outliers.
""")
    return


@app.cell
def __(img_a, img_b):
    import numpy as np
    from openpiv import pyprocess
    if img_a is not None and img_b is not None:
        try:
            u_res, v_res, s2n_res = pyprocess.extended_search_area_piv(
                img_a.astype(np.int32), img_b.astype(np.int32),
                window_size=32, overlap=16, dt=1.0, search_area_size=64,
                sig2noise_method="peak2peak"
            )
            print("PIV computed. shape:", u_res.shape, "finite:", int(np.isfinite(u_res).sum()))
            piv_res = (u_res, v_res, s2n_res)
        except Exception as e:
            print("PIV failed for default settings:", e)
            print("Try different window_size/overlap/search_area for this case.")
            piv_res = (None, None, None)
    else:
        piv_res = (None, None, None)
    piv_res
    return (piv_res,)


@app.cell
def __(piv_res):
    import matplotlib.pyplot as plt2
    u_r, v_r, s2n_r = piv_res
    if u_r is not None:
        fig2, ax2 = plt2.subplots()
        ax2.quiver(u_r, v_r)
        ax2.invert_yaxis()
        ax2.set_title("Velocity field (test14)")
        out2 = fig2
    else:
        out2 = "No vector field to plot."
    out2
    return


@app.cell
def __(mo):
    mo.md(r"""
## Notes for this case
- Inspect images for seeding density, illumination, reflections, and masks.
- Choose window size so each window contains enough particle pairs.
- Validate vectors (s2n, median) and consider multi-pass/window deformation for gradients.
- Compare with provided reference outputs in this folder if available (e.g. `.txt` results).

Back to [all cases](../all_cases.html) · [index](../index.html)
""")
    return


if __name__ == "__main__":
    app.run()
