"""
Surrogate helpers for nested ordinal RECD null models.

Destroy cross-variable ordinal / temporal dependence while preserving
marginal distributions (and, for IAAFT, approximate power spectra).

Use for null baselines of Φ₂ / Φ₃ and empirical p-values.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Dict


def iaaft_surrogate_1d(x: np.ndarray, n_iters: int = 50, seed: int = None) -> np.ndarray:
    """
    IAAFT (Iterative Amplitude Adjusted Fourier Transform) surrogate 1D.
    Preserva distribución de amplitudes y espectro de potencia aproximado.
    Bueno para series con estructura temporal.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x).ravel()
    n = len(x)
    
    # FFT original
    xf = np.fft.rfft(x)
    amp = np.abs(xf)
    
    # Fase aleatoria inicial
    phase = rng.uniform(0, 2*np.pi, len(xf))
    phase[0] = 0.0  # DC
    if len(xf) > 1:
        phase[-1] = 0.0  # Nyquist si aplica
    
    s = np.fft.irfft(amp * np.exp(1j * phase), n=n)
    
    # Iterar para ajustar distribución
    x_sorted = np.sort(x)
    for _ in range(n_iters):
        # Ajustar amplitudes en Fourier
        sf = np.fft.rfft(s)
        s = np.fft.irfft(amp * np.exp(1j * np.angle(sf)), n=n)
        # Ajustar distribución (rank order)
        ranks = np.argsort(np.argsort(s))
        s = x_sorted[ranks]
    return s


def phase_shuffle_independent(X: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Surrogate más agresivo y quirúrgico para H0 de "sin relaciones ordinales":
    - Para cada columna (variable) de forma INDEPENDIENTE:
      permuta aleatoriamente los valores (o mejor: phase shuffle por columna).
    Esto destruye TODAS las relaciones temporales y entre variables.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X).copy()
    T, N = X.shape
    for i in range(N):
        # Phase shuffle por columna
        X[:, i] = iaaft_surrogate_1d(X[:, i], n_iters=30, seed=rng.integers(0, 2**32))
    return X


def random_permutation_independent(X: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Versión aún más simple: permutación aleatoria de valores por columna.
    Destruye toda estructura temporal/ordinal.
    Útil como baseline extremo.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X).copy()
    for i in range(X.shape[1]):
        rng.shuffle(X[:, i])
    return X


def generate_surrogate_ensemble(
    X: np.ndarray,
    n_surrogates: int = 50,
    method: str = "iaaft",
    seed: int = 1234
) -> list:
    """
    Genera ensemble de surrogates.
    method: "iaaft" | "phase" | "permute"
    """
    rng = np.random.default_rng(seed)
    surrogates = []
    for k in range(n_surrogates):
        s = rng.integers(0, 2**32)
        if method == "iaaft":
            Xs = phase_shuffle_independent(X, seed=s)  # usa iaaft interno
        elif method == "phase":
            Xs = phase_shuffle_independent(X, seed=s)
        else:
            Xs = random_permutation_independent(X, seed=s)
        surrogates.append(Xs)
    return surrogates


def compute_null_distribution(
    X: np.ndarray,
    compute_fn: Callable,
    n_surrogates: int = 50,
    method: str = "iaaft",
    **compute_kwargs
) -> Dict[str, np.ndarray]:
    """
    compute_fn(X, **kwargs) -> dict con métricas (ej. mean_excess3, phi3_active_frac)
    Retorna distribuciones nulas.
    """
    surrogates = generate_surrogate_ensemble(X, n_surrogates=n_surrogates, method=method)
    nulls = {"phi3_active": [], "mean_excess3": [], "corr_excess3_lam": []}
    
    for Xs in surrogates:
        # Nota: el caller debe proveer tau_s consistente o recalcular dentro de compute_fn
        res = compute_fn(Xs, **compute_kwargs)
        nulls["phi3_active"].append(res.get("phi3_active_frac", np.nan))
        nulls["mean_excess3"].append(res.get("mean_excess3", np.nan))
        nulls["corr_excess3_lam"].append(res.get("corr_excess3_lambda", np.nan))
    
    return {k: np.array(v) for k, v in nulls.items()}


if __name__ == "__main__":
    # Smoke
    X = np.random.randn(500, 3).cumsum(0)
    Xs = phase_shuffle_independent(X)
    print("Original var col0:", X[:,0].var(), "Surrogate:", Xs[:,0].var())
    print("Cross corr orig:", np.corrcoef(X.T)[0,1])
    print("Cross corr surr:", np.corrcoef(Xs.T)[0,1])
    print("Surrogates OK")