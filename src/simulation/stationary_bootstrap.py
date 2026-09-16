"""Joint stationary bootstrap (Politis & Romano, 1994) and optional moving-block.

Algorithm (stationary bootstrap)
--------------------------------
Let the historical sample be the bivariate series
    X_t = (r_USD,t, r_EUR,t), t = 0, ..., n-1.

Expected block length is L; the geometric success probability is p = 1/L.

For each simulated path and time s = 0, ..., H-1:
  - If s = 0, or a new block is drawn, choose a start index U uniformly in {0,...,n-1}.
  - Otherwise continue the block: use X_{(previous_index + 1) mod n}.
  - Independently, start a new block at the next date with probability p.

USD and EUR always advance together. Wrapping uses modular indexing so the
sample is treated as circular (standard for the stationary bootstrap).

Block lengths are random (geometric), which makes the bootstrap strictly
stationary (hence the name), unlike the moving-block bootstrap.
"""

from __future__ import annotations

import numpy as np

from .base import JointReturnSimulator


def geometric_success_probability(expected_block_length: float) -> float:
    if expected_block_length < 1:
        raise ValueError("expected block length L must be >= 1")
    return 1.0 / float(expected_block_length)


def _index_paths(
    n_obs: int,
    n_paths: int,
    horizon_days: int,
    p: float,
    rng: np.random.Generator,
    moving_block_length: int | None = None,
) -> np.ndarray:
    """Return integer indices of shape (n_paths, horizon_days)."""
    if moving_block_length is not None:
        l = int(moving_block_length)
        n_blocks = int(np.ceil(horizon_days / l))
        starts = rng.integers(0, n_obs, size=(n_paths, n_blocks))
        idx = np.empty((n_paths, horizon_days), dtype=np.int64)
        pos = 0
        for b in range(n_blocks):
            take = min(l, horizon_days - pos)
            offsets = np.arange(take)
            idx[:, pos : pos + take] = (starts[:, b][:, None] + offsets[None, :]) % n_obs
            pos += take
        return idx

    u = rng.random((n_paths, horizon_days))
    starts = rng.integers(0, n_obs, size=(n_paths, horizon_days))
    idx = np.empty((n_paths, horizon_days), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for t in range(1, horizon_days):
        cont = u[:, t] > p
        idx[:, t] = np.where(cont, (idx[:, t - 1] + 1) % n_obs, starts[:, t])
    return idx


def politis_white_mean_block_length(series: np.ndarray, max_lag: int | None = None) -> float:
    """Heuristic Politis–White (2004) style L for a 1-d series (diagnostic only).

    Uses the implied optimal stationary-bootstrap block length based on the
    autocovariance of the series. Applied here to |r_USD| as a clustering proxy.
    """
    x = np.asarray(series, dtype=float)
    x = x - x.mean()
    n = len(x)
    if n < 20:
        return 10.0
    max_lag = max_lag or min(int(np.floor(np.sqrt(n))), n - 2)
    acov0 = float(np.dot(x, x) / n)
    if acov0 <= 0:
        return 10.0
    gamma = []
    for k in range(1, max_lag + 1):
        gamma.append(float(np.dot(x[:-k], x[k:]) / n))
    g = acov0 + 2.0 * sum(gamma)
    d = 2.0 * (acov0 ** 2)
    # Stationary bootstrap formula (Politis & White 2004, approx.)
    if g <= 0 or d <= 0:
        return 10.0
    l = (2.0 * (g ** 2) / d) ** (1.0 / 3.0) * (n ** (1.0 / 3.0))
    return float(np.clip(l, 2.0, max(2.0, n / 5.0)))


class StationaryBootstrapSimulator(JointReturnSimulator):
    name = "stationary_bootstrap"

    def __init__(self, expected_block_length: float = 10.0) -> None:
        super().__init__()
        self.expected_block_length = float(expected_block_length)
        self.p = geometric_success_probability(self.expected_block_length)
        self.name = f"stationary_bootstrap_L{int(round(self.expected_block_length))}"

    def fit(self, historical_joint_returns: np.ndarray) -> "StationaryBootstrapSimulator":
        hist = np.asarray(historical_joint_returns, dtype=float)
        if hist.ndim != 2 or hist.shape[1] != 2:
            raise ValueError("historical_joint_returns must have shape (T, 2)")
        if np.isnan(hist).any():
            raise ValueError("historical returns contain NaN")
        self.history = hist
        self.fitted = True
        return self

    def simulate(
        self,
        n_paths: int,
        horizon_days: int,
        random_seed: int | None = None,
    ) -> np.ndarray:
        hist = self._check_history()
        rng = self.rng(random_seed)
        idx = _index_paths(len(hist), n_paths, horizon_days, self.p, rng)
        return hist[idx]


class MovingBlockBootstrapSimulator(JointReturnSimulator):
    """Secondary diagnostic: fixed overlapping blocks of length L, circular wrap."""

    def __init__(self, block_length: int = 10) -> None:
        super().__init__()
        self.block_length = int(block_length)
        self.name = f"moving_block_L{self.block_length}"

    def fit(self, historical_joint_returns: np.ndarray) -> "MovingBlockBootstrapSimulator":
        hist = np.asarray(historical_joint_returns, dtype=float)
        if hist.ndim != 2 or hist.shape[1] != 2:
            raise ValueError("historical_joint_returns must have shape (T, 2)")
        self.history = hist
        self.fitted = True
        return self

    def simulate(
        self,
        n_paths: int,
        horizon_days: int,
        random_seed: int | None = None,
    ) -> np.ndarray:
        hist = self._check_history()
        rng = self.rng(random_seed)
        idx = _index_paths(
            len(hist),
            n_paths,
            horizon_days,
            p=1.0,
            rng=rng,
            moving_block_length=self.block_length,
        )
        return hist[idx]
