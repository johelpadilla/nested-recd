"""
Nested ordinal RECD levels (Φ₁, Φ₂, Φ₃) and λ-weighted accumulation.

Reference implementation of nested ordinal conjunctions and the emergent
Discrete Extramental Clock (RECD) from the nested-time structure proposed in
"Conversación de la naturaleza del tiempo" (Padilla-Villanueva).

Levels
------
- Φ₁: co-occurrence of ordinal symbols across variables
- Φ₂: persistent pairwise relations over a short lag window
- Φ₃: higher-order synergy / irreducibility proxy (excess + joint surprise)
- λ(τ), α(λ): regime-dependent weights for ΔRECD = α₁Φ₁ + α₂Φ₂ + α₃Φ₃

Pure NumPy; no hard dependency on the full ``systemictau`` package.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import warnings

# Defaults quirúrgicos (ver diseño + excess³ methods contract)
DEFAULT_M = 3
DEFAULT_DELAY = 1
DEFAULT_D_PERSIST = 4
DEFAULT_WINDOW_TAU = 13
DEFAULT_WINDOW = DEFAULT_WINDOW_TAU  # alias (methods paper)
DEFAULT_THETA_CHAOS = 0.41
DELTA_FEIGENBAUM = 4.6692016091
DEFAULT_THETA3 = 0.10  # software default
DEFAULT_THETA3_CARDIO = 0.08  # methods cardio-like reference

# excess³ = α_Syn · Syn + α_Surp · Surp  (fixed a priori; never fit on target data)
ALPHA_SYN = 0.6
ALPHA_SURP = 0.4

def _gen_ordinal(x: np.ndarray, m: int = DEFAULT_M, delay: int = DEFAULT_DELAY) -> np.ndarray:
    """Bandt–Pompe ordinal symbols (pure NumPy, no numba required)."""
    x = np.asarray(x).ravel()
    n = len(x) - (m - 1) * delay
    if n <= 0:
        return np.array([], dtype=int)
    symbols = np.zeros(n, dtype=int)
    fact = np.array([1])
    for i in range(1, m):
        fact = np.append(fact, fact[-1] * i)
    for i in range(n):
        word = np.array([x[i + j * delay] for j in range(m)])
        perm = np.argsort(word)
        symbol = 0
        for j in range(m - 1):
            cnt = np.sum(perm[j + 1:] < perm[j])
            symbol += cnt * fact[m - 1 - j]
        symbols[i] = symbol
    return symbols


def generate_multivariate_symbols(
    X: np.ndarray,
    m: int = DEFAULT_M,
    delay: int = DEFAULT_DELAY
) -> np.ndarray:
    """
    Genera matriz de símbolos ordinales S[t, i] para cada variable.
    S shape: (T_eff, N)
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X debe ser (T, N)")
    T, N = X.shape
    symbols_list = []
    min_len = None
    for i in range(N):
        sym = _gen_ordinal(X[:, i], m=m, delay=delay)
        symbols_list.append(sym)
        min_len = len(sym) if min_len is None else min(min_len, len(sym))
    # Alinear al mínimo (por si hay bordes diferentes, raro)
    S = np.stack([s[:min_len] for s in symbols_list], axis=1)
    return S


# ============================================================
# NIVEL 1: Coincidencia
# ============================================================

def compute_phi1(S: np.ndarray) -> np.ndarray:
    """
    Φ₁(t): Fracción normalizada de pares de variables que comparten símbolo idéntico en t.
    Φ₁ ∈ [0, 1]
    """
    T_eff, N = S.shape
    if N < 2:
        return np.zeros(T_eff)
    pairs = N * (N - 1) / 2.0
    phi1 = np.zeros(T_eff)
    for t in range(T_eff):
        row = S[t]
        matches = 0
        for i in range(N):
            for j in range(i + 1, N):
                if row[i] == row[j]:
                    matches += 1
        phi1[t] = matches / pairs
    return phi1


# ============================================================
# NIVEL 2: Relación Persistente
# ============================================================

def _relation_code(a: int, b: int) -> int:
    """Codifica relación entre dos símbolos como entero simple."""
    if a == b:
        return 0  # EQ
    elif a > b:
        return 1  # GT
    else:
        return 2  # LT


def compute_persistent_relations(
    S: np.ndarray,
    d: int = DEFAULT_D_PERSIST,
    min_fraction: float = 0.75
) -> np.ndarray:
    """
    Para cada par y cada t, detecta si la relación actual se mantuvo en los últimos d pasos.
    Retorna máscara (T_eff, num_pairs) o score agregado.
    """
    T_eff, N = S.shape
    if N < 2 or T_eff < d:
        return np.zeros(T_eff)

    num_pairs = N * (N - 1) // 2
    persistence_score = np.zeros(T_eff)

    pair_idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            rel_history = np.array([_relation_code(S[t, i], S[t, j]) for t in range(T_eff)])
            for t in range(d - 1, T_eff):
                window = rel_history[t - d + 1 : t + 1]
                current = rel_history[t]
                # ¿Qué fracción del window coincide con la relación actual?
                match_frac = np.mean(window == current)
                if match_frac >= min_fraction:
                    persistence_score[t] += 1.0
            pair_idx += 1

    # Normalizar por número de pares
    persistence_score = persistence_score / num_pairs
    return persistence_score


def compute_phi2(
    S: np.ndarray,
    d: int = DEFAULT_D_PERSIST,
    min_fraction: float = 0.75,
    weight_eq: float = 1.0
) -> np.ndarray:
    """
    Φ₂(t): Promedio normalizado de relaciones persistentes.
    Versión quirúrgica simple: cuenta pares con persistencia + peso extra para igualdad si se desea.
    """
    T_eff, N = S.shape
    if N < 2:
        return np.zeros(T_eff)

    phi2 = np.zeros(T_eff)
    num_pairs = float(N * (N - 1) // 2)

    for i in range(N):
        for j in range(i + 1, N):
            rel_hist = np.array([_relation_code(S[t, i], S[t, j]) for t in range(T_eff)])
            for t in range(d - 1, T_eff):
                win = rel_hist[max(0, t - d + 1): t + 1]
                curr = rel_hist[t]
                frac = np.mean(win == curr)
                if frac >= min_fraction:
                    w = weight_eq if curr == 0 else 0.85
                    phi2[t] += w

    phi2 = phi2 / num_pairs
    # Clip a [0,1] por si pesos >1
    return np.clip(phi2, 0.0, 1.0)


# ============================================================
# NIVEL 3: Emergencia / Sinergia (proxy)
# ============================================================

def _joint_entropy_and_marginals(S_window: np.ndarray) -> Tuple[float, float, float]:
    """
    Calcula H(joint), promedio H(marginal), y proxy de info mutua pairwise promedio
    sobre una ventana de símbolos.
    Muy quirúrgico: conteos exactos (N pequeño, m=3 → 6^N factible).
    """
    T, N = S_window.shape
    # Joint tuples como tuplas hashables
    joint_tuples = [tuple(row) for row in S_window]
    unique_j, counts_j = np.unique(joint_tuples, return_counts=True)
    p_joint = counts_j / counts_j.sum()
    H_joint = -np.sum(p_joint * np.log2(p_joint + 1e-12))

    # Marginales por variable
    H_margs = []
    for k in range(N):
        _, c = np.unique(S_window[:, k], return_counts=True)
        p = c / c.sum()
        H_margs.append(-np.sum(p * np.log2(p + 1e-12)))
    H_marg_mean = float(np.mean(H_margs))
    H_marg_sum = float(np.sum(H_margs))

    # Pairwise MI promedio (aprox rápida)
    pair_mi = []
    for i in range(N):
        for j in range(i + 1, N):
            joint2 = list(zip(S_window[:, i], S_window[:, j]))
            _, cj = np.unique(joint2, return_counts=True)
            pj = cj / cj.sum()
            H2 = -np.sum(pj * np.log2(pj + 1e-12))

            _, ci = np.unique(S_window[:, i], return_counts=True)
            pi = ci / ci.sum()
            Hi = -np.sum(pi * np.log2(pi + 1e-12))

            _, cj2 = np.unique(S_window[:, j], return_counts=True)
            pj2 = cj2 / cj2.sum()
            Hj = -np.sum(pj2 * np.log2(pj2 + 1e-12))

            mi = Hi + Hj - H2
            pair_mi.append(max(0.0, mi))
    mi_pair_avg = float(np.mean(pair_mi)) if pair_mi else 0.0

    # Total correlation approx = sum H - H_joint
    tc = H_marg_sum - H_joint
    # "Synergy beyond pairwise" rough: tc - (N-1)*mi_pair_avg  (heurística; puede ser negativa)
    synergy_proxy = max(0.0, tc - (N - 1) * mi_pair_avg)

    return H_joint, H_marg_sum, synergy_proxy


def compute_excess3_window(
    win: np.ndarray,
    use_surprise: bool = True,
    alpha_syn: float = ALPHA_SYN,
    alpha_surp: float = ALPHA_SURP,
) -> float:
    """
    excess³ score on a single window of joint ordinal symbols (shape (w, N)).

    Canonical hybrid proxy (methods contract):
        excess3 = alpha_syn · Syn + alpha_surp · Surp
    with default weights (0.6, 0.4) fixed a priori — never optimised on the
    scientific dataset under test.

    Syn: synergistic residual multiinformation proxy
         max(0, TC − (N−1)·MĪ_pair).
    Surp: observed-vs-independence joint surprise (frequency-weighted log-ratio).

    This is a **proxy**, not a complete PID synergy atom.
    """
    win = np.asarray(win)
    if win.ndim != 2 or win.shape[0] == 0:
        return float("nan")
    if win.shape[1] < 2:
        return 0.0

    _, _, syn = _joint_entropy_and_marginals(win)
    if not use_surprise:
        return float(syn)

    from collections import Counter

    joint_tuples = [tuple(int(v) for v in row) for row in win]
    counter = Counter(joint_tuples)
    T_win = len(joint_tuples)
    surprises = []
    for u, cnt in counter.items():
        p_indep = 1.0
        for k, val in enumerate(u):
            p_indep *= max(float(np.mean(win[:, k] == val)), 1e-9)
        p_obs = cnt / T_win
        ratio = p_obs / max(p_indep, 1e-9)
        excess_log = max(0.0, np.log2(ratio)) if ratio > 1 else 0.0
        surprises.append(excess_log * (cnt / T_win))
    joint_surprise = float(np.sum(surprises)) if surprises else 0.0
    return float(alpha_syn * syn + alpha_surp * joint_surprise)


def compute_phi3_excess(
    S: np.ndarray,
    window: int = DEFAULT_WINDOW_TAU,
    theta: float = DEFAULT_THETA3,
    stride: int = 1,
    use_surprise: bool = True,
    alpha_syn: float = ALPHA_SYN,
    alpha_surp: float = ALPHA_SURP,
    fill: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Continuous excess³ (primary) and binary Φ₃ (secondary) on symbol matrix S.

    Returns
    -------
    phi3 : ndarray
        Binary ticks: 1 if excess3 > theta else 0 (NaN before first full window).
    excess3 : ndarray
        Continuous hybrid score (primary Level-3 readout).

    Parameters
    ----------
    fill : bool
        If True, forward-fill NaNs after the first finite score (plot convenience).
        Default False preserves sparse stride semantics.
    """
    S = np.asarray(S)
    T_eff, N = S.shape
    phi3 = np.full(T_eff, np.nan)
    excess = np.full(T_eff, np.nan)

    if T_eff < window or N < 2:
        return phi3, excess

    for t in range(window - 1, T_eff, stride):
        score = compute_excess3_window(
            S[t - window + 1 : t + 1],
            use_surprise=use_surprise,
            alpha_syn=alpha_syn,
            alpha_surp=alpha_surp,
        )
        excess[t] = score
        phi3[t] = 1.0 if score > theta else 0.0

    if fill:
        last = np.nan
        for t in range(T_eff):
            if np.isfinite(excess[t]):
                last = excess[t]
            elif np.isfinite(last):
                excess[t] = last
                phi3[t] = 1.0 if last > theta else 0.0

    return phi3, excess


def compute_phi3(
    S: np.ndarray,
    window: int = 13,
    theta: float = DEFAULT_THETA3,
    stride: int = 1,
    use_surprise: bool = True,
    alpha_syn: float = ALPHA_SYN,
    alpha_surp: float = ALPHA_SURP,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Φ₃(t) binary + continuous excess³ series.

    Continuous excess³ is the **primary** Level-3 magnitude (methods contract);
    Φ₃ only counts threshold crossings. Weights default to (ALPHA_SYN, ALPHA_SURP).

    Returns (phi3_binary, excess3_continuous).
    """
    return compute_phi3_excess(
        S,
        window=window,
        theta=theta,
        stride=stride,
        use_surprise=use_surprise,
        alpha_syn=alpha_syn,
        alpha_surp=alpha_surp,
        fill=False,
    )


def mean_excess_pre_post(
    excess3: np.ndarray,
    split: int,
) -> Tuple[float, float, float]:
    """
    Mean continuous excess³ before / after a split index, and Δ = post − pre.

    ``split`` is an index into the excess3 array (same time base as returned by
    compute_recd_from_conjunctions / compute_phi3_excess).
    """
    excess3 = np.asarray(excess3, dtype=float)
    if split <= 0 or split >= len(excess3):
        raise ValueError(f"split={split} out of range for length {len(excess3)}")
    pre = float(np.nanmean(excess3[:split]))
    post = float(np.nanmean(excess3[split:]))
    return pre, post, post - pre


def surrogate_pvalue_delta_excess3(
    X: np.ndarray,
    split: int,
    n_surr: int = 199,
    method: str = "phase",
    seed: int = 0,
    m: int = DEFAULT_M,
    window: int = DEFAULT_WINDOW_TAU,
    theta3: float = DEFAULT_THETA3,
    stride: int = 1,
    delay: int = DEFAULT_DELAY,
    alpha_syn: float = ALPHA_SYN,
    alpha_surp: float = ALPHA_SURP,
) -> Dict[str, float]:
    """
    Two-sided surrogate p-value for |Δ excess3| under independent phase-shuffle
    (default) or column permutation — methods-paper null protocol.

    Inference is on the **full contrast** |Δ|, not mean-vs-mean of means alone.

    Parameters
    ----------
    X : (T, N) array
        Multivariate series (raw values; symbols computed inside).
    split : int
        Split index on the **raw** time axis; mapped to symbol time via
        embedding length (m, delay).
    method : {"phase", "permute"}
        "phase" uses per-column IAAFT / phase-shuffle; "permute" shuffles time.
    """
    from nested_recd.surrogates import (
        phase_shuffle_independent,
        random_permutation_independent,
    )

    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be shape (T, N)")

    def _delta_on(arr: np.ndarray) -> float:
        S = generate_multivariate_symbols(arr, m=m, delay=delay)
        if len(S) == 0:
            return float("nan")
        # Map split from raw time to symbol index (conservative)
        emb = (m - 1) * delay
        split_s = int(np.clip(split - emb, 1, len(S) - 1))
        _, excess = compute_phi3_excess(
            S,
            window=window,
            theta=theta3,
            stride=stride,
            alpha_syn=alpha_syn,
            alpha_surp=alpha_surp,
        )
        _, _, d = mean_excess_pre_post(excess, split_s)
        return float(d)

    obs = _delta_on(X)
    null = []
    rng = np.random.default_rng(seed)
    for _ in range(n_surr):
        s = int(rng.integers(0, 2**31 - 1))
        if method == "permute":
            Xs = random_permutation_independent(X, seed=s)
        else:
            Xs = phase_shuffle_independent(X, seed=s)
        null.append(_delta_on(Xs))
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if not np.isfinite(obs) or len(null) == 0:
        return {
            "delta_obs": float(obs) if np.isfinite(obs) else float("nan"),
            "abs_delta_obs": float("nan"),
            "p_value": float("nan"),
            "n_surr": float(len(null)),
            "method": method,
        }
    abs_obs = abs(obs)
    # Add-one smoothing: (1 + #{|null| >= |obs|}) / (1 + n)
    p = (1.0 + float(np.sum(np.abs(null) >= abs_obs))) / (1.0 + len(null))
    return {
        "delta_obs": float(obs),
        "abs_delta_obs": float(abs_obs),
        "p_value": float(p),
        "n_surr": float(len(null)),
        "method": method,
        "null_mean_abs": float(np.mean(np.abs(null))),
        "null_std_abs": float(np.std(np.abs(null))),
    }


# ============================================================
# λ(t) y pesos α(λ)
# ============================================================

def compute_lambda(
    tau_s: np.ndarray,
    theta_chaos: float = DEFAULT_THETA_CHAOS,
    delta_f: float = DELTA_FEIGENBAUM,
    gamma_universal: float = 0.2
) -> np.ndarray:
    """
    λ(t) híbrido según documento:
    componente empírica (Tau) + factor de universalidad Feigenbaum.
    """
    tau = np.asarray(tau_s)
    lam_emp = np.maximum(0.0, (np.abs(tau) - theta_chaos) / theta_chaos)
    # Factor suave de universalidad
    univ_factor = 1.0 + gamma_universal * np.log(delta_f)
    lam = lam_emp * univ_factor
    return lam


def alpha_weights(
    lam: np.ndarray,
    alpha10: float = 1.0,
    alpha20: float = 1.0,
    alpha30: float = 1.0,
    beta1: float = 2.0,
    gamma2: float = 1.5,
    gamma3: float = 3.0,
    delta3: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Retorna α1(λ), α2(λ), α3(λ) según la forma propuesta.
    """
    a1 = alpha10 * np.exp(-beta1 * lam)
    a2 = alpha20 * (1.0 + gamma2 * lam)
    a3 = alpha30 * (1.0 + gamma3 * lam + delta3 * lam**2)
    return a1, a2, a3


def regime_lambda_proxy(
    r: float,
    r_onset: float = 3.57,
    r_full: float = 3.85,
    lam_full: float = 1.0
) -> float:
    """
    Mapea ground-truth r (parámetro del mapa logístico acoplado) a un λ efectivo.
    Esto permite aislar el efecto puro del régimen sobre el Nivel 3 (Opción 1 ligera),
    sin depender del |τ_s| empírico (que en proxies acoplados puede ser más alto en pre
    que en caos desarrollado).

    - r <= r_onset (pre-Feigenbaum): lam=0 → α3 baseline (menor peso ontológico)
    - r > r_onset (caos): lam crece → α3 aumenta (más peso al Nivel 3, como postula la tesis)

    Ramp simple hasta r_full. Transparente y falsable.
    """
    if r <= r_onset:
        return 0.0
    frac = min(1.0, (r - r_onset) / max(1e-9, (r_full - r_onset)))
    return frac * lam_full


# ============================================================
# RECD desde conjunciones (nuevo)
# ============================================================

def compute_recd_from_conjunctions(
    X: np.ndarray,
    tau_s: Optional[np.ndarray] = None,
    m: int = DEFAULT_M,
    d: int = DEFAULT_D_PERSIST,
    theta3: float = DEFAULT_THETA3,
    window_tau: int = DEFAULT_WINDOW_TAU,
    lam_override: Optional[np.ndarray] = None,
    stride: int = 1,
    alpha_syn: float = ALPHA_SYN,
    alpha_surp: float = ALPHA_SURP,
    **alpha_kwargs
) -> Dict[str, np.ndarray]:
    """
    Full nested ordinal RECD pipeline:
    - Bandt–Pompe symbols
    - Φ1, Φ2, Φ3 (binary) + continuous excess³ (primary Level-3 readout)
    - λ and α(λ) regime weights
    - ΔRECD and cumulative T_recd

    Level-3 contract (excess³ methods):
    - Continuous ``excess3`` is the primary magnitude for order-3 claims.
    - ``phi3`` is a secondary threshold discretisation of excess3.
    - ``delta_recd`` keeps the legacy λ-weighted clock using **binary** Φ₃
      (back-compat with CCTP / Discrete Extramental Clock nesting). Do not
      treat ``delta_recd`` alone as the Level-3 scientific readout.

    If tau_s is omitted, λ=0 (warning). Use ``lam_override`` for regime ground truth.
    """
    S = generate_multivariate_symbols(X, m=m)
    T_eff = S.shape[0]

    phi1 = compute_phi1(S)
    phi2 = compute_phi2(S, d=d)
    phi3, excess3 = compute_phi3(
        S,
        window=window_tau,
        theta=theta3,
        stride=stride,
        alpha_syn=alpha_syn,
        alpha_surp=alpha_surp,
    )

    # Determinar λ: override > tau-derived > zero
    if lam_override is not None:
        if np.isscalar(lam_override):
            lam = np.full(T_eff, float(lam_override))
        else:
            lo = np.asarray(lam_override).ravel()
            lam = lo[-T_eff:] if len(lo) >= T_eff else np.pad(lo, (T_eff - len(lo), 0), constant_values=0.0)
    elif tau_s is None:
        # Placeholder: el caller debe proveer tau_s alineado
        lam = np.zeros(T_eff)
        warnings.warn("tau_s no provisto. λ=0 para todos los t. Proporcione tau_s para resultados reales.")
    else:
        # Recortar o alinear de forma conservadora
        tau_aligned = tau_s[-T_eff:] if len(tau_s) > T_eff else np.pad(tau_s, (T_eff - len(tau_s), 0), constant_values=np.nan)
        lam = compute_lambda(tau_aligned)

    a1, a2, a3 = alpha_weights(lam, **alpha_kwargs)

    # Rellenar NaN en phi3 con 0 para acumulación (o mantener y usar nan-safe)
    phi3_safe = np.nan_to_num(phi3, nan=0.0)
    excess3_safe = np.nan_to_num(excess3, nan=0.0)

    # Legacy RECD clock: binary Φ₃ in the α-weighted sum (back-compat)
    delta_recd = a1 * phi1 + a2 * phi2 + a3 * phi3_safe
    T_recd = np.nancumsum(delta_recd)

    return {
        "S": S,
        "phi1": phi1,
        "phi2": phi2,
        "phi3": phi3,
        "excess3": excess3,
        "lambda": lam,
        "alpha1": a1,
        "alpha2": a2,
        "alpha3": a3,
        "delta_recd": delta_recd,
        "T_recd": T_recd,
        "params": {
            "m": m,
            "d": d,
            "theta3": theta3,
            "window_tau": window_tau,
            "stride": stride,
            "alpha_syn": alpha_syn,
            "alpha_surp": alpha_surp,
            "level3_primary": "excess3",
            "level3_secondary": "phi3",
            **alpha_kwargs,
        },
    }


# ============================================================
# Utilidades quirúrgicas
# ============================================================

def simple_level_classification(phi1: np.ndarray, phi2: np.ndarray, phi3: np.ndarray) -> np.ndarray:
    """
    Clasificación burda del nivel dominante en cada t.
    1,2,3 o 0 si todo bajo.
    """
    levels = np.zeros_like(phi1, dtype=int)
    for t in range(len(phi1)):
        p3 = phi3[t] if not np.isnan(phi3[t]) else 0
        p2 = phi2[t]
        p1 = phi1[t]
        if p3 > 0.1:
            levels[t] = 3
        elif p2 > 0.2:
            levels[t] = 2
        elif p1 > 0.1:
            levels[t] = 1
    return levels


def compute_weighted_contributions(res: Dict) -> Dict[str, np.ndarray]:
    """
    Retorna las contribuciones ponderadas reales de cada nivel al ΔRECD:
    contrib1 = α1(t) * Φ1(t), etc.
    Incluye también los totales medios para análisis rápido.
    """
    a1 = res["alpha1"]
    a2 = res["alpha2"]
    a3 = res["alpha3"]
    p1 = res["phi1"]
    p2 = res["phi2"]
    p3 = np.nan_to_num(res["phi3"], nan=0.0)

    c1 = a1 * p1
    c2 = a2 * p2
    c3 = a3 * p3

    return {
        "contrib1": c1,
        "contrib2": c2,
        "contrib3": c3,
        "mean_contrib1": float(np.nanmean(c1)),
        "mean_contrib2": float(np.nanmean(c2)),
        "mean_contrib3": float(np.nanmean(c3)),
        "total_mean_delta": float(np.nanmean(c1 + c2 + c3)),
        "frac_contrib3": float(np.nanmean(c3) / (np.nanmean(c1 + c2 + c3) + 1e-12)),
    }


def high_level3_rate(excess3: np.ndarray, thresh: float = 1.75) -> float:
    """Fracción de tiempo donde el exceso de Nivel 3 excede el umbral exigente."""
    excess3 = np.asarray(excess3)
    return float(np.nanmean(excess3 > thresh))


def run_logical_demonstrations(n_steps: int = 800, m: int = 3, d: int = 4, theta3: float = DEFAULT_THETA3):
    """
    Demostraciones lógicas / casos degenerados (analíticos + numéricos mínimos).
    Cumple el punto de mejora identificado: validación conceptual antes de confiar solo en numérico.

    Casos:
    1. Variables completamente independientes (coupling=0, ruido independiente) → Φ2 y Φ3 ~ 0.
    2. Variables idénticas (totalmente sincronizadas) → Φ1 alto, Φ2 alto (persistencia), Φ3 bajo (sin novedad irreducible).
    3. Acoplamiento fuerte con drive común → Φ3 detectable (configuraciones conjuntas específicas).
    """
    print("\n=== DEMOSTRACIONES LÓGICAS (Casos Degenerados) ===")
    rng = np.random.default_rng(123)

    # Caso 1: Independientes
    X_indep = rng.normal(size=(n_steps, 3))
    res1 = compute_recd_from_conjunctions(X_indep, tau_s=np.zeros(n_steps), m=m, d=d, theta3=theta3)
    print(f"1. Independientes (ruido blanco):")
    print(f"   Φ1={np.nanmean(res1['phi1']):.3f}  Φ2={np.nanmean(res1['phi2']):.3f}  Φ3_act={np.nanmean(res1['phi3']>0):.3f}  excess={np.nanmean(res1['excess3']):.4f}")
    print(f"   (Esperado: Φ3 bajo; solo fluctuaciones)")

    # Caso 2: Idénticas (sincronizadas triviales)
    base = rng.normal(size=n_steps).cumsum() * 0.05
    X_ident = np.stack([base + rng.normal(0, 0.01, n_steps) for _ in range(3)], axis=1)
    res2 = compute_recd_from_conjunctions(X_ident, tau_s=np.zeros(n_steps), m=m, d=d, theta3=theta3)
    print(f"2. Idénticas (sincronización trivial):")
    print(f"   Φ1={np.nanmean(res2['phi1']):.3f}  Φ2={np.nanmean(res2['phi2']):.3f}  Φ3_act={np.nanmean(res2['phi3']>0):.3f}  excess={np.nanmean(res2['excess3']):.4f}")
    print(f"   (Esperado: Φ1/Φ2 altos; Φ3 no mucho más alto que ruido)")

    # Caso 3: Acoplamiento fuerte (simulado con correlación ordinal fuerte + algo de drive)
    X_coupled = np.zeros((n_steps, 3))
    X_coupled[:, 0] = rng.normal(size=n_steps).cumsum() * 0.03
    for i in range(1, 3):
        X_coupled[:, i] = 0.85 * X_coupled[:, 0] + 0.15 * rng.normal(size=n_steps).cumsum() * 0.03
    res3 = compute_recd_from_conjunctions(X_coupled, tau_s=np.zeros(n_steps), m=m, d=d, theta3=theta3)
    print(f"3. Acoplamiento fuerte (drive común):")
    print(f"   Φ1={np.nanmean(res3['phi1']):.3f}  Φ2={np.nanmean(res3['phi2']):.3f}  Φ3_act={np.nanmean(res3['phi3']>0):.3f}  excess={np.nanmean(res3['excess3']):.4f}")
    print(f"   (Esperado: Φ3 más alto por estructuras conjuntas específicas)")

    return {"indep": res1, "identical": res2, "coupled": res3}


if __name__ == "__main__":
    # Smoke test quirúrgico + demos lógicas
    print("Smoke test nested_recd.ordinal_levels")
    np.random.seed(42)
    X_test = np.random.randn(200, 3).cumsum(axis=0)
    res = compute_recd_from_conjunctions(X_test, tau_s=np.zeros(200))
    print("phi1 mean:", round(np.nanmean(res["phi1"]), 4))
    print("phi2 mean:", round(np.nanmean(res["phi2"]), 4))
    print("phi3 active fraction:", round(np.nanmean(res["phi3"] > 0), 4))
    print("T_recd final:", round(res["T_recd"][-1], 2))

    # Ejecutar demostraciones lógicas
    run_logical_demonstrations(n_steps=500, theta3=0.10)
    print("\nOK - Incluye demostraciones lógicas de casos degenerados.")