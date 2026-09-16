"""Initial savings holdings and KZT-equivalent valuation / rebalancing."""

from __future__ import annotations

import numpy as np


def initial_holdings(W0: float, weights: np.ndarray, S_usd0: float, S_eur0: float) -> np.ndarray:
    """Return (A_kzt, A_usd, A_eur) in native currency units."""
    w = np.asarray(weights, dtype=float).reshape(3)
    if abs(w.sum() - 1.0) > 1e-10 or np.any(w < -1e-15):
        raise ValueError("weights must be non-negative and sum to 1")
    if W0 < 0:
        raise ValueError("W0 must be non-negative")
    a_kzt = w[0] * W0
    a_usd = (w[1] * W0) / S_usd0
    a_eur = (w[2] * W0) / S_eur0
    return np.array([a_kzt, a_usd, a_eur], dtype=float)


def kzt_wealth(holdings: np.ndarray, S_usd: np.ndarray, S_eur: np.ndarray) -> np.ndarray:
    """holdings last axis (kzt, usd, eur); FX broadcastable."""
    h = np.asarray(holdings, dtype=float)
    return h[..., 0] + h[..., 1] * S_usd + h[..., 2] * S_eur


def rebalance_holdings(
    wealth: np.ndarray,
    weights: np.ndarray,
    S_usd: np.ndarray,
    S_eur: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero-cost target-weight rebalancing (KZT wealth conserved)."""
    w = np.asarray(weights, dtype=float).reshape(3)
    wealth = np.maximum(np.asarray(wealth, dtype=float), 0.0)
    a_kzt = w[0] * wealth
    a_usd = (w[1] * wealth) / S_usd
    a_eur = (w[2] * wealth) / S_eur
    return a_kzt, a_usd, a_eur


def rebalance_holdings_with_cost(
    a_kzt: np.ndarray,
    a_usd: np.ndarray,
    a_eur: np.ndarray,
    weights: np.ndarray,
    S_usd: np.ndarray,
    S_eur: np.ndarray,
    conversion_cost: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Target-weight rebalancing with proportional conversion cost on buys.

    Cost convention
    ---------------
    Let V_i be current KZT-equivalent holdings and W = sum V_i.
    Desired KZT values at wealth W are w_i * W.
    One-way buy volume (KZT) = sum_i max(w_i W - V_i, 0).
    Fee = conversion_cost * buy_volume (single proportional charge; no double
    counting of the matching sell side).

    Remaining wealth W' = W - fee is then allocated exactly to target weights.
    When conversion_cost == 0 this matches :func:`rebalance_holdings`.
    """
    c = float(conversion_cost)
    if c < 0:
        raise ValueError("conversion_cost must be non-negative")
    w = np.asarray(weights, dtype=float).reshape(3)
    S_usd = np.asarray(S_usd, dtype=float)
    S_eur = np.asarray(S_eur, dtype=float)
    v_k = np.asarray(a_kzt, dtype=float)
    v_u = np.asarray(a_usd, dtype=float) * S_usd
    v_e = np.asarray(a_eur, dtype=float) * S_eur
    W = np.maximum(v_k + v_u + v_e, 0.0)
    if c == 0.0:
        hk, hu, he = rebalance_holdings(W, w, S_usd, S_eur)
        return hk, hu, he, np.zeros_like(W)

    target = np.stack([w[0] * W, w[1] * W, w[2] * W], axis=0)
    current = np.stack([v_k, v_u, v_e], axis=0)
    buys = np.maximum(target - current, 0.0).sum(axis=0)
    fee = c * buys
    # Fee cannot exceed wealth
    fee = np.minimum(fee, W)
    W_net = np.maximum(W - fee, 0.0)
    hk, hu, he = rebalance_holdings(W_net, w, S_usd, S_eur)
    return hk, hu, he, fee
