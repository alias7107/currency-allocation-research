"""Bivariate Student-t and Gaussian straw-man joint simulators.

Student-t parameterization
--------------------------
A draw X ∈ R^2 has density of the multivariate t with
  - location μ ∈ R^2
  - positive-definite **scale** matrix Σ (not the covariance)
  - degrees of freedom ν > 2

    f(x) ∝ |Σ|^{-1/2} [1 + (1/ν)(x-μ)' Σ^{-1} (x-μ)]^{-(ν+2)/2}

For ν > 2 the covariance is
    Cov(X) = ν / (ν-2) * Σ.

Parameters are estimated by maximum likelihood (L-BFGS-B) on the baseline
sample. Sampling uses the scale-mixture representation:
    X = μ + L Z / sqrt(W/ν),  Z ~ N(0,I), W ~ χ²_ν,  Σ = L L'.

The Gaussian straw-man uses the sample mean and sample covariance (MLE under
normality) and is **not** a primary candidate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats
from scipy.special import gammaln

from .base import JointReturnSimulator


@dataclass
class StudentTParams:
    mu: np.ndarray
    scale: np.ndarray
    df: float
    covariance: np.ndarray
    nll: float
    success: bool
    message: str


def _pack(mu: np.ndarray, chol_l: np.ndarray, log_df_shift: float) -> np.ndarray:
    # chol_l is 2x2 lower triangular: l00, l10, l11 (log of diagonals)
    return np.array(
        [mu[0], mu[1], np.log(chol_l[0, 0]), chol_l[1, 0], np.log(chol_l[1, 1]), log_df_shift],
        dtype=float,
    )


def _unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    mu = theta[:2]
    l00 = np.exp(theta[2])
    l10 = theta[3]
    l11 = np.exp(theta[4])
    chol = np.array([[l00, 0.0], [l10, l11]])
    scale = chol @ chol.T
    df = 2.05 + np.exp(np.clip(theta[5], -20.0, 5.5))  # ν ∈ (~2.05, ~247)
    return mu, scale, df


def mvt_logpdf(x: np.ndarray, mu: np.ndarray, scale: np.ndarray, df: float) -> np.ndarray:
    d = 2
    diff = x - mu
    sign, logdet = np.linalg.slogdet(scale)
    if sign <= 0:
        return np.full(len(x), -1e12)
    inv = np.linalg.inv(scale)
    quad = np.einsum("ni,ij,nj->n", diff, inv, diff)
    const = (
        gammaln((df + d) / 2.0)
        - gammaln(df / 2.0)
        - 0.5 * d * np.log(df * np.pi)
        - 0.5 * logdet
    )
    return const - 0.5 * (df + d) * np.log1p(quad / df)


def fit_bivariate_student_t(x: np.ndarray) -> StudentTParams:
    x = np.asarray(x, dtype=float)
    mu0 = np.median(x, axis=0)
    emp_cov = np.cov(x, rowvar=False, ddof=1)
    # start with scale ≈ cov * (ν-2)/ν using ν=8
    df0 = 8.0
    scale0 = emp_cov * (df0 - 2.0) / df0
    try:
        chol0 = np.linalg.cholesky(scale0)
    except np.linalg.LinAlgError:
        chol0 = np.linalg.cholesky(emp_cov + 1e-8 * np.eye(2))

    def nll(theta: np.ndarray) -> float:
        mu, scale, df = _unpack(theta)
        ll = mvt_logpdf(x, mu, scale, df)
        val = -float(np.mean(ll))
        if not np.isfinite(val):
            return 1e12
        return val

    theta0 = _pack(mu0, chol0, np.log(df0 - 2.05))
    result = optimize.minimize(nll, theta0, method="L-BFGS-B")
    mu, scale, df = _unpack(result.x)
    cov = scale * df / (df - 2.0) if df > 2 else scale * np.nan
    return StudentTParams(
        mu=mu,
        scale=scale,
        df=float(df),
        covariance=cov,
        nll=float(result.fun) * len(x),
        success=bool(result.success),
        message=str(result.message),
    )


class StudentTSimulator(JointReturnSimulator):
    name = "student_t"

    def __init__(self) -> None:
        super().__init__()
        self.params: StudentTParams | None = None

    def fit(self, historical_joint_returns: np.ndarray) -> "StudentTSimulator":
        hist = np.asarray(historical_joint_returns, dtype=float)
        if hist.ndim != 2 or hist.shape[1] != 2:
            raise ValueError("historical_joint_returns must have shape (T, 2)")
        self.history = hist
        self.params = fit_bivariate_student_t(hist)
        self.fitted = True
        return self

    def simulate(
        self,
        n_paths: int,
        horizon_days: int,
        random_seed: int | None = None,
    ) -> np.ndarray:
        if self.params is None:
            raise RuntimeError("student_t must be fit before simulate().")
        rng = self.rng(random_seed)
        n = n_paths * horizon_days
        z = rng.standard_normal(size=(n, 2))
        w = np.maximum(rng.chisquare(self.params.df, size=n), 1e-12)
        chol = np.linalg.cholesky(self.params.scale)
        draws = self.params.mu + (z @ chol.T) / np.sqrt(w / self.params.df)[:, None]
        return draws.reshape(n_paths, horizon_days, 2)


class GaussianSimulator(JointReturnSimulator):
    """Straw-man bivariate Gaussian. Not a primary DGP."""

    name = "gaussian"

    def __init__(self) -> None:
        super().__init__()
        self.mu: np.ndarray | None = None
        self.cov: np.ndarray | None = None

    def fit(self, historical_joint_returns: np.ndarray) -> "GaussianSimulator":
        hist = np.asarray(historical_joint_returns, dtype=float)
        self.history = hist
        self.mu = hist.mean(axis=0)
        self.cov = np.cov(hist, rowvar=False, ddof=1)
        self.fitted = True
        return self

    def simulate(
        self,
        n_paths: int,
        horizon_days: int,
        random_seed: int | None = None,
    ) -> np.ndarray:
        if self.mu is None or self.cov is None:
            raise RuntimeError("gaussian must be fit before simulate().")
        rng = self.rng(random_seed)
        n = n_paths * horizon_days
        draws = rng.multivariate_normal(self.mu, self.cov, size=n)
        return draws.reshape(n_paths, horizon_days, 2)


def gaussian_logpdf(x: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    return stats.multivariate_normal.logpdf(x, mean=mu, cov=cov)
