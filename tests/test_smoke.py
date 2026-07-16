"""Smoke and invariant tests for nested-recd (excess³ contract)."""

import numpy as np
import pytest

from nested_recd import (
    __version__,
    ALPHA_SYN,
    ALPHA_SURP,
    DEFAULT_THETA3,
    DEFAULT_THETA3_CARDIO,
    compute_recd_from_conjunctions,
    compute_phi1,
    compute_phi3,
    compute_phi3_excess,
    compute_excess3_window,
    compute_weighted_contributions,
    generate_multivariate_symbols,
    mean_excess_pre_post,
    phase_shuffle_independent,
    random_permutation_independent,
    surrogate_pvalue_delta_excess3,
)


def test_version():
    assert __version__ == "0.2.0"


def test_weights_a_priori():
    assert ALPHA_SYN == 0.6
    assert ALPHA_SURP == 0.4
    assert abs(ALPHA_SYN + ALPHA_SURP - 1.0) < 1e-12
    assert DEFAULT_THETA3 == 0.10
    assert DEFAULT_THETA3_CARDIO == 0.08


def test_pipeline_shapes_and_bounds():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3)).cumsum(axis=0)
    out = compute_recd_from_conjunctions(X, m=3, d=4, theta3=0.10)

    T = out["phi1"].shape[0]
    assert out["phi2"].shape == (T,)
    assert out["phi3"].shape == (T,)
    assert out["excess3"].shape == (T,)
    assert out["delta_recd"].shape == (T,)
    assert out["T_recd"].shape == (T,)
    assert out["S"].shape[1] == 3

    assert np.all((out["phi1"] >= 0) & (out["phi1"] <= 1))
    assert np.all((out["phi2"] >= 0) & (out["phi2"] <= 1))
    assert np.nanmax(out["phi3"]) <= 1.0
    assert np.nanmin(out["phi3"]) >= 0.0 or np.isnan(np.nanmin(out["phi3"]))
    assert np.all(np.diff(out["T_recd"]) >= -1e-12)

    # Contract: continuous primary declared in params
    assert out["params"]["level3_primary"] == "excess3"
    assert out["params"]["alpha_syn"] == ALPHA_SYN
    assert out["params"]["alpha_surp"] == ALPHA_SURP


def test_identical_series_high_phi1():
    rng = np.random.default_rng(1)
    base = rng.normal(size=400).cumsum() * 0.05
    X = np.column_stack([base + rng.normal(0, 0.001, 400) for _ in range(3)])
    out = compute_recd_from_conjunctions(X, tau_s=np.zeros(400))
    assert float(np.nanmean(out["phi1"])) > 0.5


def test_lam_override_changes_alpha3():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 2)).cumsum(axis=0)
    low = compute_recd_from_conjunctions(X, lam_override=0.0)
    high = compute_recd_from_conjunctions(X, lam_override=1.0)
    assert float(np.mean(high["alpha3"])) > float(np.mean(low["alpha3"]))


def test_weighted_contributions():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(250, 3)).cumsum(axis=0)
    out = compute_recd_from_conjunctions(X, lam_override=0.5)
    w = compute_weighted_contributions(out)
    assert 0.0 <= w["frac_contrib3"] <= 1.0 + 1e-9
    assert abs(w["total_mean_delta"] - (w["mean_contrib1"] + w["mean_contrib2"] + w["mean_contrib3"])) < 1e-9


def test_surrogates_break_cross_corr():
    rng = np.random.default_rng(4)
    x = rng.normal(size=500).cumsum()
    X = np.column_stack([x, x * 0.9 + rng.normal(0, 0.1, 500)])
    corr0 = abs(np.corrcoef(X.T)[0, 1])
    Xs = random_permutation_independent(X, seed=99)
    corr1 = abs(np.corrcoef(Xs.T)[0, 1])
    assert corr0 > 0.5
    assert corr1 < corr0


def test_symbols_shape():
    X = np.random.default_rng(5).normal(size=(100, 4))
    S = generate_multivariate_symbols(X, m=3)
    assert S.ndim == 2
    assert S.shape[1] == 4
    assert S.shape[0] == 100 - (3 - 1) * 1


def test_phi1_single_variable_zeros():
    S = np.zeros((50, 1), dtype=int)
    phi1 = compute_phi1(S)
    assert np.all(phi1 == 0)


def test_phi3_delegates_to_phi3_excess():
    rng = np.random.default_rng(6)
    S = generate_multivariate_symbols(rng.normal(size=(120, 3)).cumsum(0), m=3)
    a = compute_phi3(S, window=13, theta=0.10)
    b = compute_phi3_excess(S, window=13, theta=0.10)
    np.testing.assert_allclose(a[0], b[0], equal_nan=True)
    np.testing.assert_allclose(a[1], b[1], equal_nan=True)


def test_excess3_window_weights():
    rng = np.random.default_rng(7)
    # XOR-like synergistic lock: pairs weak, triple determined
    n = 40
    a = rng.integers(0, 2, size=n)
    b = rng.integers(0, 2, size=n)
    c = a ^ b
    win = np.column_stack([a, b, c]).astype(int)
    score = compute_excess3_window(win)
    assert np.isfinite(score)
    assert score >= 0.0


def test_mean_excess_pre_post():
    x = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    pre, post, delta = mean_excess_pre_post(x, split=2)
    assert pre == 0.0
    assert post == 1.0
    assert delta == 1.0


def test_surrogate_pvalue_smoke():
    rng = np.random.default_rng(8)
    T = 180
    # Independent noise: |Δ| should not be extreme
    X = rng.normal(size=(T, 3))
    res = surrogate_pvalue_delta_excess3(
        X, split=T // 2, n_surr=19, method="permute", seed=1, window=11, stride=2
    )
    assert "p_value" in res
    assert 0.0 <= res["p_value"] <= 1.0
    assert res["n_surr"] >= 1
