import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    mo.md(r"""
# 04 — Validation & filtering

PIV always produces some **spurious vectors** (outliers) due to low seeding, noise, loss of pairs, masks, or gradients. Validation detects them; filtering/replacement fixes gaps.

## Common methods
- **Signal-to-noise ratio (s2n)**: reject vectors with weak/ambiguous correlation peaks (e.g. peak2peak, peak2mean).
- **Global/range filter**: remove physically impossible velocities.
- **Local median / normalized median**: compare each vector to neighbors; outliers deviate strongly.
- **Standard deviation filter**: reject vectors far from local statistics.

## After detection
- Mark invalid vectors (mask/NaN).
- Replace with interpolated/local-mean values if needed for visualization or further processing — but keep track of what was replaced.
""")
    return (mo,)


@app.cell
def __():
    import numpy as np
    from openpiv.tools import synimage
    from openpiv import pyprocess
    img_a, img_b = synimage.synimagegen(128, dt=0.1)
    u, v, s2n = pyprocess.extended_search_area_piv(
        img_a.astype(np.int32), img_b.astype(np.int32),
        window_size=32, overlap=16, dt=0.1, search_area_size=64,
        sig2noise_method='peak2peak'
    )
    print('s2n min/median/max:', float(np.nanmin(s2n)), float(np.nanmedian(s2n)), float(np.nanmax(s2n)))
    return s2n, u, v


@app.cell
def __(s2n, u, v):
    # Simple s2n-based masking example
    mask_bad = s2n < 1.1
    u_clean = u.copy(); v_clean = v.copy()
    u_clean[mask_bad] = np.nan; v_clean[mask_bad] = np.nan
    print('flagged by s2n:', int(mask_bad.sum()), 'of', mask_bad.size)
    return ()


@app.cell
def __(mo):
    mo.md(r"""
## Tips
- **Do not smooth before validation** — smoothing can hide bad vectors.
- Tune thresholds on your data; synthetic tests help find sensible defaults.
- Inspect where outliers cluster (boundaries, masks, low seeding).
- Keep raw + validated results so decisions are reproducible.

Next: [05 — Masking & preprocessing](05_masking_and_preprocessing.html)
""")
    return ()


if __name__ == "__main__":
    app.run()
