import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    mo.md(r"""
# 05 — Masking & preprocessing

Real experiments often include walls, models, bubbles, or moving objects that should not be treated as flow. **Masking** excludes them.

## Types
- **Static mask**: known geometry constant across frames.
- **Dynamic mask**: objects detected per image/frame.

## Learn from these examples
- **[dynamic_masking](marimo_notebooks/dynamic_masking.html)** ([interactive .py](marimo_notebooks/dynamic_masking.py), [molab](https://molab.marimo.io/github/openpiv/openpiv-python-examples/blob/main/marimo_notebooks/dynamic_masking.py)) — remove moving objects dynamically. *Use when:* objects move through the field of view.
- **[openpiv_static_masking_example](marimo_notebooks/openpiv_static_masking_example.html)** ([interactive .py](marimo_notebooks/openpiv_static_masking_example.py), [molab](https://molab.marimo.io/github/openpiv/openpiv-python-examples/blob/main/marimo_notebooks/openpiv_static_masking_example.py)) — apply a static mask (e.g. circular region) in multi-pass windef. *Use when:* geometry is fixed.
- **[preprocessing_filters](marimo_notebooks/preprocessing_filters.html)** ([interactive .py](marimo_notebooks/preprocessing_filters.py), [molab](https://molab.marimo.io/github/openpiv/openpiv-python-examples/blob/main/marimo_notebooks/preprocessing_filters.py)) — spatial/temporal filters for image enhancement and background subtraction.
- **[test21 case](marimo_cases/test21.html)** ([interactive .py](marimo_cases/test21.py)) — bubbles with a provided mask image (`bubblesMask.png`).
- **[phase_separation](marimo_notebooks/phase_separation.html)** ([interactive .py](marimo_notebooks/phase_separation.py), [molab](https://molab.marimo.io/github/openpiv/openpiv-python-examples/blob/main/marimo_notebooks/phase_separation.py)) — separate phases in two-phase PIV before analysis.
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
