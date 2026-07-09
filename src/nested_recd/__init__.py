"""
nested-recd
===========

Nested ordinal RECD: Φ₁ / Φ₂ / Φ₃ conjunction levels and λ-weighted
Discrete Extramental Clock accumulation.

Quick start
-----------
>>> import numpy as np
>>> from nested_recd import compute_recd_from_conjunctions
>>> X = np.random.randn(500, 3).cumsum(axis=0)
>>> out = compute_recd_from_conjunctions(X)
>>> out["T_recd"][-1]
"""

from nested_recd.ordinal_levels import (
    DEFAULT_M,
    DEFAULT_DELAY,
    DEFAULT_D_PERSIST,
    DEFAULT_WINDOW_TAU,
    DEFAULT_THETA_CHAOS,
    DEFAULT_THETA3,
    DELTA_FEIGENBAUM,
    generate_multivariate_symbols,
    compute_phi1,
    compute_phi2,
    compute_phi3,
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

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DEFAULT_M",
    "DEFAULT_DELAY",
    "DEFAULT_D_PERSIST",
    "DEFAULT_WINDOW_TAU",
    "DEFAULT_THETA_CHAOS",
    "DEFAULT_THETA3",
    "DELTA_FEIGENBAUM",
    "generate_multivariate_symbols",
    "compute_phi1",
    "compute_phi2",
    "compute_phi3",
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
