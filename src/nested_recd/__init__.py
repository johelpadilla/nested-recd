"""
nested-recd
===========

Nested ordinal RECD: Φ₁ / Φ₂ / Φ₃ conjunction levels, continuous excess³
(primary Level-3 readout), and λ-weighted Discrete Extramental Clock.

Quick start
-----------
>>> import numpy as np
>>> from nested_recd import compute_recd_from_conjunctions
>>> X = np.random.randn(500, 3).cumsum(axis=0)
>>> out = compute_recd_from_conjunctions(X)
>>> float(np.nanmean(out["excess3"]))  # continuous primary Level-3 score
"""

from nested_recd.ordinal_levels import (
    DEFAULT_M,
    DEFAULT_DELAY,
    DEFAULT_D_PERSIST,
    DEFAULT_WINDOW_TAU,
    DEFAULT_WINDOW,
    DEFAULT_THETA_CHAOS,
    DEFAULT_THETA3,
    DEFAULT_THETA3_CARDIO,
    DELTA_FEIGENBAUM,
    ALPHA_SYN,
    ALPHA_SURP,
    generate_multivariate_symbols,
    compute_phi1,
    compute_phi2,
    compute_phi3,
    compute_phi3_excess,
    compute_excess3_window,
    mean_excess_pre_post,
    surrogate_pvalue_delta_excess3,
    compute_lambda,
    alpha_weights,
    regime_lambda_proxy,
    compute_recd_from_conjunctions,
    simple_level_classification,
    compute_weighted_contributions,
    high_level3_rate,
    run_logical_demonstrations,
)
from nested_recd.surrogates import (
    iaaft_surrogate_1d,
    phase_shuffle_independent,
    random_permutation_independent,
    generate_surrogate_ensemble,
    compute_null_distribution,
)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "DEFAULT_M",
    "DEFAULT_DELAY",
    "DEFAULT_D_PERSIST",
    "DEFAULT_WINDOW_TAU",
    "DEFAULT_WINDOW",
    "DEFAULT_THETA_CHAOS",
    "DEFAULT_THETA3",
    "DEFAULT_THETA3_CARDIO",
    "DELTA_FEIGENBAUM",
    "ALPHA_SYN",
    "ALPHA_SURP",
    "generate_multivariate_symbols",
    "compute_phi1",
    "compute_phi2",
    "compute_phi3",
    "compute_phi3_excess",
    "compute_excess3_window",
    "mean_excess_pre_post",
    "surrogate_pvalue_delta_excess3",
    "compute_lambda",
    "alpha_weights",
    "regime_lambda_proxy",
    "compute_recd_from_conjunctions",
    "simple_level_classification",
    "compute_weighted_contributions",
    "high_level3_rate",
    "run_logical_demonstrations",
    "iaaft_surrogate_1d",
    "phase_shuffle_independent",
    "random_permutation_independent",
    "generate_surrogate_ensemble",
    "compute_null_distribution",
]
