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

## Tradeoffs
- More computation.
- Bad first-pass vectors can bias later passes → validate between passes.
""")
    return (mo,)


@app.cell
def __(mo):
    mo.md("## Example with `windef` settings (conceptual)\nIn OpenPIV, `windef`/`PIVSettings` controls passes via window sizes and overlaps, e.g. `(64,32,16)` windows with `(32,16,8)` overlap.")
    return ()


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
