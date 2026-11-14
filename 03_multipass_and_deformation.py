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
# 03 — Multi-pass & window deformation

A single pass with fixed windows struggles with strong gradients and large displacements. **Multi-pass** (multi-grid) PIV refines the estimate:

1. **Pass 1**: large windows to capture large displacements robustly.
2. **Pass 2+**: smaller windows, using previous pass to offset/deform windows, improving spatial resolution.
3. **Window deformation** warps images/windows to account for velocity gradients inside the window.

## Why it helps
- Better accuracy in shear, vortices, and boundary layers.
- Reduces in-plane loss of correlation.
- Allows smaller final windows than single-pass would permit.

## Learn from these examples
- **[tutorial_multipass](marimo_notebooks/tutorial_multipass.html)** ([interactive .py](marimo_notebooks/tutorial_multipass.py), [molab](https://molab.marimo.io/github/openpiv/openpiv-python-examples/blob/main/marimo_notebooks/tutorial_multipass.py)) — step-by-step multi-pass processing.
- **[test20_windef](marimo_notebooks/test20_windef.html)** ([interactive .py](marimo_notebooks/test20_windef.py), [molab](https://molab.marimo.io/github/openpiv/openpiv-python-examples/blob/main/marimo_notebooks/test20_windef.py)) — window deformation algorithm in action. *Use when:* you need higher accuracy in gradients.
- **[window_deformation_comparison](marimo_notebooks/window_deformation_comparison.html)** ([interactive .py](marimo_notebooks/window_deformation_comparison.py), [molab](https://molab.marimo.io/github/openpiv/openpiv-python-examples/blob/main/marimo_notebooks/window_deformation_comparison.py)) — compare deformation strategies.
- **[case_B_windef_small_window](marimo_notebooks/case_B_windef_small_window.html)** ([interactive .py](marimo_notebooks/case_B_windef_small_window.py), [molab](https://molab.marimo.io/github/openpiv/openpiv-python-examples/blob/main/marimo_notebooks/case_B_windef_small_window.py)) — PIV challenge case with multi-grid deformation.
- **[test20_multipass](marimo_notebooks/test20_multipass.html)** / **[test20 case](marimo_cases/test20.html)** — practical multi-pass on test20 data.
""")
    return (mo,)


@app.cell
def __():
    from openpiv import windef
    s = windef.Settings()
    s.windowsizes = (64, 32, 16)
    s.overlap = (32, 16, 8)
    s.num_iterations = 3
    print('windows:', s.windowsizes, 'overlap:', s.overlap, 'passes:', s.num_iterations)
    return (s,)


@app.cell
def __(mo):
    mo.md(r"""
## Tips
- Start coarse enough that pass 1 finds the main displacement reliably.
- Keep ~50% overlap as a good default.
- Enable validation between passes to prevent error propagation.
- Use deformation for strong gradients; symmetric deformation is often robust.

Next: [04 — Validation & filtering](04_validation_and_filtering.html)
""")
    return ()


if __name__ == "__main__":
    app.run()
