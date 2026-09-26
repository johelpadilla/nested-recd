# Changelog

## 0.2.3 — 2026-09-25 (local correction; not yet deposited or released)

### Fixed — joint counting in `Syn` (affects `excess3`)

In versions through 0.2.2, `_joint_entropy_and_marginals` called
`np.unique` on a list of tuples without `axis=0`. NumPy flattens that
input. `H_joint` was the entropy of the pooled symbol values, and each
pairwise `H(S_i, S_j)` was the entropy of the pooled values of the two
columns. `TC`, `MI_pair` and `Syn = max(0, TC − (N−1)·mean MI)` were
therefore not the published definition. `excess3 = 0.6·Syn + 0.4·Surp`,
binary `phi3`, and the `phi3` term inside `delta_recd` / `T_recd`
inherited the error. The failure was silent.

Joint and pairwise counts now use `np.unique(..., axis=0)` on rows.

Not affected: `Surp` (`collections.Counter` on tuples), one-dimensional
marginals, `Res_pair` / IPF / KL, `generate_multivariate_symbols`,
`compute_phi1`, `compute_phi2`, the λ/α weights, and `surrogates.py`.
For `N = 2`, `Syn` is 0 under both countings, so bivariate `excess3`
does not change.

Checked against the 24 Sep 2026 cascade archive
(`peak_agg_20260924_145527`), six seeds, window 13, stride 1. The
unpatched 0.2.2 function reproduces the archive. The corrected six-seed
means are:

| r | archive excess3 | corrected excess3 | corrected Syn |
|---|---:|---:|---:|
| 3.30 | 1.757509 | 2.210069 | 0.797365 |
| 3.5675 | 2.378246 | 2.382588 | 0.013745 |
| 3.85 | 1.863299 | 2.802431 | 1.656969 |
| 3.95 | 1.899271 | 2.897386 | 1.781721 |

`Surp` and the tuple counts match the archive at these four stations.
On this window the corrected maximum is at the grid edge, not at
`r = 3.5675`.

### Added

- `legacy_pooled_counting: bool = False` on `compute_excess3_window`,
  `compute_phi3_excess`, `compute_phi3`, `compute_recd_from_conjunctions`
  and `surrogate_pvalue_delta_excess3`. `True` reproduces ≤ 0.2.2,
  including the bug, and emits a `UserWarning`.
- `tests/test_joint_counting.py`.

The Zenodo record 10.5281/zenodo.21937204 remains the 0.2.2 deposit.
This tree does not assign a new DOI.
