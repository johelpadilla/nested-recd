# nested-recd

[![PyPI version](https://img.shields.io/pypi/v/nested-recd.svg)](https://pypi.org/project/nested-recd/)
[![Python](https://img.shields.io/pypi/pyversions/nested-recd.svg)](https://pypi.org/project/nested-recd/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Nested ordinal RECD** — pure-NumPy implementation of nested ordinal conjunction levels (Φ₁, Φ₂, Φ₃) and λ-weighted Discrete Extramental Clock (RECD) accumulation.

This is the standalone library behind the nested-time structure used in the CCTP/SDDB cardiac pilot and the experimental design in *Conversación de la naturaleza del tiempo*.

## Install

```bash
pip install nested-recd
```

From source:

```bash
pip install git+https://github.com/johelpadilla/nested-recd.git
```

## Quick start

```python
import numpy as np
from nested_recd import compute_recd_from_conjunctions, compute_weighted_contributions

# Multivariate series: shape (T, N)
rng = np.random.default_rng(0)
X = rng.normal(size=(800, 3)).cumsum(axis=0)

out = compute_recd_from_conjunctions(X, m=3, d=4, theta3=0.10)

print("mean Φ1:", float(np.nanmean(out["phi1"])))
print("mean Φ2:", float(np.nanmean(out["phi2"])))
print("Φ3 active fraction:", float(np.nanmean(out["phi3"] > 0)))
print("final T_recd:", float(out["T_recd"][-1]))

w = compute_weighted_contributions(out)
print("frac level-3 contribution:", w["frac_contrib3"])
```

Optional: supply a Systemic Tau series `tau_s` (aligned in time) so that λ is derived empirically:

```python
out = compute_recd_from_conjunctions(X, tau_s=tau_s)
```

Or fix the regime weight with a scalar / array override:

```python
out = compute_recd_from_conjunctions(X, lam_override=0.5)
```

## What it computes

| Symbol | Meaning |
|--------|---------|
| **Φ₁** | Co-occurrence of identical ordinal symbols across variable pairs |
| **Φ₂** | Persistence of pairwise ordinal relations over lag `d` |
| **Φ₃** | Higher-order synergy proxy (total-correlation excess + joint surprise) |
| **λ** | Chaos / reorganization intensity (from `τ_s` or `lam_override`) |
| **α(λ)** | Level weights: α₁ decays with λ; α₂, α₃ grow with λ |
| **ΔRECD** | `α₁Φ₁ + α₂Φ₂ + α₃Φ₃` |
| **T_recd** | Cumulative sum of ΔRECD (nested-time clock) |

Ordinal symbols use Bandt–Pompe patterns (`m=3` by default).

## Surrogates

```python
from nested_recd import phase_shuffle_independent, random_permutation_independent

X_null = phase_shuffle_independent(X, seed=42)   # IAAFT per column
X_perm = random_permutation_independent(X, seed=42)
```

## API surface

```python
from nested_recd import (
    compute_recd_from_conjunctions,
    compute_phi1, compute_phi2, compute_phi3,
    compute_lambda, alpha_weights, regime_lambda_proxy,
    compute_weighted_contributions, high_level3_rate,
    generate_multivariate_symbols,
    phase_shuffle_independent, generate_surrogate_ensemble,
)
```

## Relation to other projects

| Project | Role |
|---------|------|
| [`systemictau`](https://pypi.org/project/systemictau/) | Full Systemic Tau library (τ_s, platform, studio) |
| [`cctp-sddb-systemic-tau`](https://github.com/johelpadilla/cctp-sddb-systemic-tau) | Cardiac VF pilot using this nested RECD pipeline |
| This package | Lightweight, installable nested-time / ordinal RECD core |

## Citation

If you use this software, please cite the CCTP/SDDB pilot archive:

> Padilla-Villanueva, J. (2026). *CCTP/SDDB: Systemic Tau and ordinal RECD before spontaneous ventricular fibrillation* (v1.0.1). Zenodo. https://doi.org/10.5281/zenodo.21270699

And the theoretical nested-time / RECD framework as appropriate for your venue.

```bibtex
@software{padilla_nested_recd_2026,
  author  = {Padilla-Villanueva, Johel},
  title   = {nested-recd: Nested ordinal RECD levels},
  year    = {2026},
  url     = {https://github.com/johelpadilla/nested-recd},
  version = {0.1.0}
}
```

## License

MIT © 2026 Johel Padilla-Villanueva  
ORCID: [0000-0002-5797-6931](https://orcid.org/0000-0002-5797-6931)
