"""Common interface for joint USD/KZT–EUR/KZT scenario generators."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src import config


class JointReturnSimulator(ABC):
    """Fit on a (T, 2) array of [USD log return, EUR log return]."""

    name: str = "base"

    def __init__(self) -> None:
        self.history: np.ndarray | None = None
        self.fitted: bool = False

    @abstractmethod
    def fit(self, historical_joint_returns: np.ndarray) -> "JointReturnSimulator":
        raise NotImplementedError

    @abstractmethod
    def simulate(
        self,
        n_paths: int,
        horizon_days: int,
        random_seed: int | None = None,
    ) -> np.ndarray:
        """Return array of shape (n_paths, horizon_days, 2)."""
        raise NotImplementedError

    def rng(self, random_seed: int | None) -> np.random.Generator:
        seed = config.SIM_RANDOM_SEED if random_seed is None else random_seed
        return np.random.default_rng(seed)

    def _check_history(self) -> np.ndarray:
        if self.history is None or not self.fitted:
            raise RuntimeError(f"{self.name} must be fit before simulate().")
        return self.history
