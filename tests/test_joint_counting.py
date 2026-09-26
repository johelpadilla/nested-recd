"""Regression tests for the joint-counting fix (0.2.3).

nested-recd <= 0.2.2 computed H_joint and the pairwise H(S_i, S_j) inside
``_joint_entropy_and_marginals`` over pooled symbol values (np.unique on a
list of tuples flattens it). These tests pin the joint-tuple values.
``test_xor_synergy_is_one_bit`` and ``test_joint_entropy_counts_tuples``
fail on 0.2.2 and pass on 0.2.3.
"""
import inspect

import numpy as np
import pytest

from nested_recd import compute_excess3_window, compute_phi3_excess
from nested_recd.ordinal_levels import _joint_entropy_and_marginals


def _xor_window(reps: int = 3) -> np.ndarray:
    # columns: a, b independent fair bits; c = a XOR b; d constant
    base = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
        ]
    )
    return np.tile(base, (reps, 1))


def test_xor_synergy_is_one_bit():
    """XOR: every pair is independent (MI = 0) but the triple carries 1 bit.

    H_joint = 2, sum H_i = 3, TC = 1, mean MI = 0 -> Syn = 1 bit.
    0.2.2 returns H_joint = 0.954 and Syn = 0.263.
    """
    H_joint, H_sum, syn = _joint_entropy_and_marginals(_xor_window())
    assert H_joint == pytest.approx(2.0, abs=1e-9)
    assert H_sum == pytest.approx(3.0, abs=1e-9)
    assert syn == pytest.approx(1.0, abs=1e-9)


def test_joint_entropy_counts_tuples():
    """13 distinct joint tuples in a window of 13 -> H_joint = log2(13)."""
    rng = np.random.default_rng(0)
    while True:
        win = rng.integers(0, 6, size=(13, 4))
        if len({tuple(r) for r in win}) == 13:
            break
    H_joint, _, _ = _joint_entropy_and_marginals(win)
    assert H_joint == pytest.approx(np.log2(13), abs=1e-9)


def test_identical_columns_have_no_synergy():
    """Four copies of one series: TC = 3 H, every MI = H -> Syn = 0."""
    rng = np.random.default_rng(1)
    x = rng.integers(0, 6, size=40)
    win = np.stack([x, x, x, x], axis=1)
    H_joint, H_sum, syn = _joint_entropy_and_marginals(win)
    assert H_sum == pytest.approx(4 * H_joint, abs=1e-9)
    assert syn == pytest.approx(0.0, abs=1e-9)


def test_matches_counter_reference():
    """Random windows agree with an independent Counter-over-tuples version."""
    from collections import Counter

    def H(counter):
        c = np.array(list(counter.values()), float)
        p = c / c.sum()
        return float(-np.sum(p * np.log2(p + 1e-12)))

    rng = np.random.default_rng(2)
    for _ in range(200):
        win = rng.integers(0, 6, size=(13, 4))
        N = win.shape[1]
        rows = [tuple(r) for r in win.tolist()]
        Hj = H(Counter(rows))
        Hm = [H(Counter(win[:, k].tolist())) for k in range(N)]
        mi = [
            max(
                0.0,
                Hm[i]
                + Hm[j]
                - H(Counter(zip(win[:, i].tolist(), win[:, j].tolist()))),
            )
            for i in range(N)
            for j in range(i + 1, N)
        ]
        syn_ref = max(0.0, sum(Hm) - Hj - (N - 1) * np.mean(mi))
        H_joint, H_sum, syn = _joint_entropy_and_marginals(win)
        assert H_joint == pytest.approx(Hj, abs=1e-12)
        assert syn == pytest.approx(syn_ref, abs=1e-12)


def test_legacy_flag_reproduces_022():
    """legacy_pooled_counting=True reproduces nested-recd 0.2.2."""
    assert "legacy_pooled_counting" in inspect.signature(
        _joint_entropy_and_marginals
    ).parameters
    H_joint, _, syn = _joint_entropy_and_marginals(
        _xor_window(), legacy_pooled_counting=True
    )
    assert H_joint == pytest.approx(LEGACY_XOR_HJOINT, abs=1e-12)
    assert syn == pytest.approx(LEGACY_XOR_SYN, abs=1e-12)
    rng = np.random.default_rng(3)
    win = rng.integers(0, 6, size=(13, 4))
    assert compute_excess3_window(
        win, legacy_pooled_counting=True
    ) == pytest.approx(LEGACY_RANDOM_EXCESS3, abs=1e-12)


def test_legacy_flag_warns():
    S = np.random.default_rng(4).integers(0, 6, size=(40, 4))
    with pytest.warns(UserWarning, match="0.2.2"):
        compute_phi3_excess(S, legacy_pooled_counting=True)


def test_surp_unchanged_by_fix():
    """Surp always counted tuples: with Syn switched off both modes agree."""
    rng = np.random.default_rng(5)
    for _ in range(50):
        win = rng.integers(0, 6, size=(13, 4))
        a = compute_excess3_window(win, alpha_syn=0.0, alpha_surp=1.0)
        b = compute_excess3_window(
            win, alpha_syn=0.0, alpha_surp=1.0, legacy_pooled_counting=True
        )
        assert a == pytest.approx(b, abs=1e-15)


def test_bivariate_synergy_is_zero():
    """For N=2, Syn = TC - MI of that single pair, so both modes give 0."""
    rng = np.random.default_rng(8)
    win = rng.integers(0, 6, size=(13, 2))
    _, _, syn = _joint_entropy_and_marginals(win)
    _, _, legacy = _joint_entropy_and_marginals(win, legacy_pooled_counting=True)
    assert syn == pytest.approx(0.0, abs=1e-12)
    assert legacy == pytest.approx(0.0, abs=1e-12)


# Frozen outputs of nested-recd 0.2.2
LEGACY_XOR_HJOINT = 0.9544340029220797
LEGACY_XOR_SYN = 0.2624831837630126
LEGACY_RANDOM_EXCESS3 = 2.355186195068367
