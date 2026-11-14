import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    mo.md(r"""
# 05 — Masking & preprocessing

Real experiments often include walls, models, bubbles, reflections, or moving objects that should not be treated as flow. **Masking** excludes them.

## Types
- **Static mask**: known geometry (e.g. airfoil, wall) constant across frames.
- **Dynamic mask**: objects detected per image/frame (moving interfaces, bubbles, etc.).

## Preprocessing goals
- Improve particle contrast and uniformity.
- Reduce background illumination variations.
- Suppress features that confuse correlation.

## In OpenPIV
- `preprocess.dynamic_masking` can segment objects (edges/intensity/combined methods in newer versions).
- Masks can be applied so invalid regions produce no/invalid vectors.
""")
    return (mo,)


@app.cell
def __():
    import numpy as np
    from openpiv.tools import synimage
    from openpiv import preprocess
    img_a, _ = synimage.synimagegen(128)
    masked, mask = preprocess.dynamic_masking(img_a, method='edges')
    print('masked shape:', masked.shape, 'masked pixels:', int((mask > 0).sum()))
    return img_a, mask, masked


@app.cell
def __(img_a, mask, masked):
    import matplotlib.pyplot as plt1
    fig1, ax1 = plt1.subplots(1, 3, figsize=(10, 3))
    ax1[0].imshow(img_a, cmap='gray'); ax1[0].set_title('Original')
    ax1[1].imshow(mask, cmap='gray'); ax1[1].set_title('Mask')
    ax1[2].imshow(masked, cmap='gray'); ax1[2].set_title('Masked image')
    fig1
    return (fig1,)


@app.cell
def __(mo):
    mo.md(r"""
## Tips
- Inspect masks visually — over-masking removes valid flow, under-masking creates spurious vectors.
- Combine preprocessing carefully; aggressive filtering can distort particle images.
- For moving boundaries, dynamic masking per frame (or pair) is often necessary.

Next: [06 — Tips, tricks & pitfalls](06_tips_tricks_pitfalls.html)
""")
    return ()


if __name__ == "__main__":
    app.run()
