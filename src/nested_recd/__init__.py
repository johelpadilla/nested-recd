"""
nested-recd
===========

Nested ordinal RECD: Φ₁ / Φ₂ / Φ₃ conjunction levels, continuous excess³
(exportable Level-3 proxy; not the strong residual), optional Res_pair,
and λ-weighted Discrete Extramental Clock.

Quick start
-----------
>>> import numpy as np
>>> from nested_recd import compute_recd_from_conjunctions
>>> X = np.random.randn(500, 3).cumsum(axis=0)
>>> out = compute_recd_from_conjunctions(X)
>>> float(np.nanmean(out["excess3"]))  # continuous Level-3 *proxy* (T-IV)
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
    compute_res_pair_window,
    compute_res_pair,
    pairwise_maxent_ipf,
    kl_divergence,
    mean_excess_pre_post,
    surrogate_pvalue_delta_excess3,
    compute_lambda,
    alpha_weights,
    alpha_weights_gibbs,
    alpha_compare_template_gibbs,
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

__version__ = "0.2.3"

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
    "compute_res_pair_window",
    "compute_res_pair",
    "pairwise_maxent_ipf",
    "kl_divergence",
    "mean_excess_pre_post",
    "surrogate_pvalue_delta_excess3",
    "compute_lambda",
    "alpha_weights",
    "alpha_weights_gibbs",
    "alpha_compare_template_gibbs",
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
