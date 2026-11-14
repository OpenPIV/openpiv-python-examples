import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    mo.md(r"""
# 06 — Tips, tricks & pitfalls

A practical checklist for reliable PIV with OpenPIV, illustrated by repository examples.

## Experiment & images
- **Seeding density**: ~5–10 particle pairs per interrogation window.
- **Particle size**: ~2–3 pixels is often ideal for subpixel peak fitting.
- **Displacement**: aim for ~1/4 of window size between frames.
- **Avoid** saturation, motion blur, strong reflections, and large out-of-plane loss.

## Processing choices
- **Window size**: large = robust but low resolution; small = detailed but noisy.
- **Overlap**: 50% default; 75% smoother-looking but correlated.
- **Multi-pass**: for gradients (see [03](03_multipass_and_deformation.html), [test20_windef](marimo_notebooks/test20_windef.html)).
- **Search area**: must cover maximum expected displacement.

## Validation & post-processing
- Validate (s2n, median, range) before smoothing (see [04](04_validation_and_filtering.html)).
- Mask objects/boundaries (see [05](05_masking_and_preprocessing.html), [test21](marimo_cases/test21.html)).
- Replace/interpolate outliers cautiously; report how many were invalid.

## Learn by example (feature → where to look)
| Feature / need | Example(s) | Insight |
|---|---|---|
| Basic workflow | [tutorial1](marimo_notebooks/tutorial1.html), [test1](marimo_cases/test1.html) | Start simple, verify end-to-end |
| Many pairs / speed | [tutorial2](marimo_notebooks/tutorial2.html) | Multiprocessing |
| Multi-pass & deformation | [tutorial_multipass](marimo_notebooks/tutorial_multipass.html), [test20_windef](marimo_notebooks/test20_windef.html), [case_B_windef_small_window](marimo_notebooks/case_B_windef_small_window.html) | Resolve gradients, reduce bias |
| Correlation choices | [compare_correlations_algorithms](marimo_notebooks/compare_correlations_algorithms.html), [studying_correlations](marimo_notebooks/studying_correlations.html) | Linear vs circular, normalized effects |
| Outlier detection | [compare_median_filters](marimo_notebooks/compare_median_filters.html) | Normalized median is robust |
| Static/dynamic masking | [openpiv_static_masking_example](marimo_notebooks/openpiv_static_masking_example.html), [dynamic_masking](marimo_notebooks/dynamic_masking.html), [test21](marimo_cases/test21.html) | Exclude non-flow regions |
| Preprocessing | [preprocessing_filters](marimo_notebooks/preprocessing_filters.html) | Enhance contrast, remove background |
| Two-phase / separation | [phase_separation](marimo_notebooks/phase_separation.html), [two_phase_piv](marimo_cases/two_phase_piv.html) | Separate tracers from dispersed phase |
| Ensemble/steady flows | [ensemble_correlation](marimo_notebooks/ensemble_correlation.html) | Average correlations for stationary stats |
| 3D PIV | [PIV_3D_example](marimo_notebooks/PIV_3D_example.html), [output_3D_test](marimo_cases/output_3D_test.html) | Volumetric data workflows |
| Real-world cases | [test9](marimo_cases/test9.html) (Kármán), [openpiv_bridgepile_wake](marimo_notebooks/openpiv_bridgepile_wake.html), [test17](marimo_cases/test17.html), [active_nematics](marimo_notebooks/active_nematics.html) | Diverse flow phenomena |

## Common pitfalls
| Symptom | Likely cause | Try |
|---|---|---|
| Random/noisy vectors | Window too small, low seeding, wrong $\Delta t$ | Larger window, better seeding, adjust $\Delta t$ |
| Systematic bias | Peak-locking, poor subpixel fit, gradients | Better particle size, deformation, multi-pass |
| Missing vectors near objects | No mask / bad mask | Static/dynamic masking |
| Over-smoothed field | Aggressive smoothing | Reduce smoothing, validate first |

## Reproducibility
- Save settings (e.g. `PIVSettings` to YAML), record versions, and keep raw outputs.

Explore all: [All test cases & notebooks](all_cases.html) · [index](index.html)
""")
    return (mo,)


if __name__ == "__main__":
    app.run()
