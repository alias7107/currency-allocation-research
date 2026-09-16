"""Joint i.i.d. historical bootstrap (naive benchmark).

Each simulated day draws one historical row (r_USD, r_EUR) with replacement.
Contemporaneous dependence is preserved; serial dependence is destroyed.
"""

from __future__ import annotations

import numpy as np

from .base import JointReturnSimulator


class IIDBootstrapSimulator(JointReturnSimulator):
    name = "iid_bootstrap"

    def fit(self, historical_joint_returns: np.ndarray) -> "IIDBootstrapSimulator":
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
        idx = rng.integers(0, len(hist), size=(n_paths, horizon_days))
        return hist[idx]
