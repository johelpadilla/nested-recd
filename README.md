# nested-recd

> **Correction (0.2.3).** Versions ≤ 0.2.2 computed `H_joint` and the pairwise
> entropies inside `Syn` with `np.unique` on a list of tuples and no `axis=0`.
> NumPy flattens that input, so `Syn`, `excess3` and binary `phi3` were not the
> published definition when `N ≥ 3`. `Surp` and `Res_pair` were not affected.
> For `N = 2`, `Syn` is 0 either way. 0.2.3 counts joint rows.
> `legacy_pooled_counting=True` reproduces the old numbers and warns.
> See `CHANGELOG.md`. This version:
> [10.5281/zenodo.22970079](https://doi.org/10.5281/zenodo.22970079).
> The 0.2.2 deposit remains
> [10.5281/zenodo.21937204](https://doi.org/10.5281/zenodo.21937204).

[![PyPI version](https://img.shields.io/pypi/v/nested-recd.svg)](https://pypi.org/project/nested-recd/)
[![Python](https://img.shields.io/pypi/pyversions/nested-recd.svg)](https://pypi.org/project/nested-recd/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Nested ordinal RECD** — pure-NumPy implementation of nested ordinal conjunction levels (Φ₁, Φ₂, Φ₃), the continuous **excess³** Level-3 *proxy* (T-IV; not the strong residual), optional **Res_pair** \(=\mathrm{KL}(P\|P^{(2)})\), Gibbs-α weights, and λ-weighted Discrete Extramental Clock (RECD) accumulation.

This is the **canonical nested-clock software** for the Systemic Tau / RECD stack (foundations paper, excess³ methods preprint, cardiac CCTP/SDDB pilot, Academy Learning Tau).

**Léxico (no negociable):** *strong Level-3* := `Res_pair`. *excess3* := exportable proxy. Report `w` with any Res_pair figure. Abundance (excess3) ≠ aggregation share \(f_3\).

## Install

```bash
pip install nested-recd
```

From source (development):

```bash
pip install -e ".[dev]"
```

## excess³ contract (methods)

Canonical specification: Padilla-Villanueva (2026), *excess³: A Pre-Specified Continuous Proxy…*  
DOI: [10.5281/zenodo.21385937](https://doi.org/10.5281/zenodo.21385937) · repo: [github.com/johelpadilla/excess3](https://github.com/johelpadilla/excess3)

| Rule | Software |
|------|----------|
| `excess3 = 0.6·Syn + 0.4·Surp` | `ALPHA_SYN`, `ALPHA_SURP` (fixed a priori) |
| Continuous primary | `out["excess3"]` / `compute_phi3_excess` |
| Φ₃ secondary | binary ticks of excess3 vs `theta3` |
| Null on full contrast | `surrogate_pvalue_delta_excess3` (phase-shuffle / permute on `\|Δ\|`) |
| Proxy ≠ full PID | documented; no complete Williams–Beer atom claimed |
| Parallel nesting | Φ₁, Φ₂, excess³ computed without hard Boolean ascent |

Defaults: `m=3`, delay `1`, window `13`, `theta3=0.10` (software); cardio-like reference `DEFAULT_THETA3_CARDIO=0.08`.

## Quick start

```python
import numpy as np
from nested_recd import compute_recd_from_conjunctions, ALPHA_SYN, ALPHA_SURP

rng = np.random.default_rng(0)
X = rng.normal(size=(800, 3)).cumsum(axis=0)

out = compute_recd_from_conjunctions(X, m=3, d=4, theta3=0.10)

print("mean excess³ (primary):", float(np.nanmean(out["excess3"])))
print("Φ3 active fraction:", float(np.nanmean(out["phi3"] > 0)))
print("weights:", out["params"]["alpha_syn"], out["params"]["alpha_surp"])
print("final T_recd:", float(out["T_recd"][-1]))
```

### Strong residual Res_pair (optional)

```python
out = compute_recd_from_conjunctions(X, compute_res=True)
# or, on a symbol window:
from nested_recd import compute_res_pair_window
# Always report window length with Res_pair (Nota 4: w=13 can bias chaos).
```

### Continuous excess³ only

```python
from nested_recd import generate_multivariate_symbols, compute_phi3_excess

S = generate_multivariate_symbols(X, m=3)
phi3, excess3 = compute_phi3_excess(S, window=13, theta=0.10, stride=1)
```

### Surrogate p-value on |Δ excess3|

```python
from nested_recd import surrogate_pvalue_delta_excess3

res = surrogate_pvalue_delta_excess3(X, split=400, n_surr=99, method="phase", seed=0)
print(res["delta_obs"], res["p_value"])
```

### Optional τ_s for λ regime weights

```python
out = compute_recd_from_conjunctions(X, tau_s=tau_s)
# or
out = compute_recd_from_conjunctions(X, lam_override=0.5)
```

**Note:** `delta_recd` / `T_recd` use **binary Φ₃** in the α-weighted clock (legacy Discrete Extramental Clock). For **exportable / scalable** Level-3 reporting, use continuous **excess³** and nulls on `|Δ excess3|`. For **strong** Level-3 claims, report **`res_pair`** (enable `compute_res=True`) **and the window `w`**. Do not identify the two.

## What it computes

| Symbol | Meaning |
|--------|---------|
| **Φ₁** | Co-occurrence of identical ordinal symbols across variable pairs |
| **Φ₂** | Persistence of pairwise ordinal relations over lag `d` |
| **excess³** | Continuous hybrid: `0.6·Syn + 0.4·Surp` (**exportable Level-3 proxy**; T-IV) |
| **Φ₃** | Binary ticks of excess³ above `theta3` (**secondary**) |
| **Res_pair** | Strong Level-3 residual `KL(P ‖ P^(2))` (optional; always report `w`) |
| **λ** | Chaos / reorganization intensity (from `τ_s` or `lam_override`) |
| **α(λ)** | Design-engine weights (template or Gibbs-α); **not** a joint observable |
| **ΔRECD** | `α₁Φ₁ + α₂Φ₂ + α₃Φ₃` (legacy clock; binary Φ₃) |
| **T_recd** | Cumulative sum of ΔRECD |

Ordinal symbols use Bandt–Pompe patterns (`m=3` by default).

## Surrogates

```python
from nested_recd import phase_shuffle_independent, random_permutation_independent

X_null = phase_shuffle_independent(X, seed=42)
X_perm = random_permutation_independent(X, seed=42)
```

## API surface

```python
from nested_recd import (
    ALPHA_SYN, ALPHA_SURP,
    DEFAULT_THETA3, DEFAULT_THETA3_CARDIO,
    compute_recd_from_conjunctions,
    compute_phi1, compute_phi2, compute_phi3,
    compute_phi3_excess, compute_excess3_window,
    mean_excess_pre_post, surrogate_pvalue_delta_excess3,
    compute_lambda, alpha_weights, alpha_weights_gibbs, regime_lambda_proxy,
    compute_res_pair, compute_res_pair_window,
    compute_weighted_contributions, high_level3_rate,
    generate_multivariate_symbols,
    phase_shuffle_independent, generate_surrogate_ensemble,
)
```

## Relation to other projects

| Project | Role |
|---------|------|
| [`systemictau`](https://pypi.org/project/systemictau/) | Full Systemic Tau library (τ_s, gate RECD, studio) — optional `[nested]` extra |
| [`systemictau-web`](https://github.com/johelpadilla/systemictau-web) | Streamlit app; depends on this package |
| [`excess3`](https://github.com/johelpadilla/excess3) | Methods + intro ES + primer (specification) |
| [`cctp-sddb-systemic-tau`](https://github.com/johelpadilla/cctp-sddb-systemic-tau) | Cardiac VF pilot |
| This package | Lightweight, installable nested-time / excess³ core |

## Citation

If you use excess³ / Level-3 from this package, cite the **methods** preprint (canonical specification):

> Padilla-Villanueva, J. (2026). *excess³: A Pre-Specified Continuous Proxy for Order-3 Synergistic Surplus* (methods). Zenodo. https://doi.org/10.5281/zenodo.21385937

This software (0.2.3):

> Padilla-Villanueva, J. (2026). *nested-recd 0.2.3: Nested ordinal RECD, excess³ proxy, and Res_pair*. Zenodo. https://doi.org/10.5281/zenodo.22970079

The previous release (0.2.2), which pooled symbols inside Syn, remains citable as https://doi.org/10.5281/zenodo.21937204.

Software / related stack:

> Padilla-Villanueva, J. (2026). Systemic Tau software archive. Zenodo. https://doi.org/10.5281/zenodo.20576241

CCTP/SDDB pilot:

> Padilla-Villanueva, J. (2026). *CCTP/SDDB: Systemic Tau and ordinal RECD before spontaneous ventricular fibrillation* (v1.0.1). Zenodo. https://doi.org/10.5281/zenodo.21270699

```bibtex
@software{padilla_nested_recd_2026,
  author  = {Padilla-Villanueva, Johel},
  title   = {nested-recd: Nested ordinal RECD levels and continuous excess³},
  year    = {2026},
  url     = {https://github.com/johelpadilla/nested-recd},
  version = {0.2.3},
  doi     = {10.5281/zenodo.22970079}
}
```

## License

MIT © 2026 Johel Padilla-Villanueva  
ORCID: [0000-0002-5797-6931](https://orcid.org/0000-0002-5797-6931)
