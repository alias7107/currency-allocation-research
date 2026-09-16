"""Household income/expense currency profiles (normalized KZT-equivalent at t=0)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HouseholdProfile:
    name: str
    qI_kzt: float
    qI_usd: float
    qI_eur: float
    qE_kzt: float
    qE_usd: float
    qE_eur: float

    def __post_init__(self) -> None:
        si = self.qI_kzt + self.qI_usd + self.qI_eur
        se = self.qE_kzt + self.qE_usd + self.qE_eur
        if abs(si - 1.0) > 1e-12 or abs(se - 1.0) > 1e-12:
            raise ValueError(f"{self.name}: income/expense shares must sum to 1")
        for v in (
            self.qI_kzt,
            self.qI_usd,
            self.qI_eur,
            self.qE_kzt,
            self.qE_usd,
            self.qE_eur,
        ):
            if v < -1e-15:
                raise ValueError("shares must be non-negative")


P1_LOCAL_CONTROL = HouseholdProfile(
    "P1_local_control", 1.0, 0.0, 0.0, 1.0, 0.0, 0.0
)
P2_FOREIGN_INCOME = HouseholdProfile(
    "P2_foreign_income_mismatch", 0.0, 1.0, 0.0, 1.0, 0.0, 0.0
)
P3_MIXED_INCOME = HouseholdProfile(
    "P3_mixed_income", 0.50, 0.50, 0.0, 1.0, 0.0, 0.0
)
P4_MULTI_CURRENCY = HouseholdProfile(
    "P4_full_multi_currency", 0.0, 0.50, 0.50, 0.80, 0.0, 0.20
)

PRIMARY_PROFILES = (
    P1_LOCAL_CONTROL,
    P2_FOREIGN_INCOME,
    P3_MIXED_INCOME,
    P4_MULTI_CURRENCY,
)


def strategy_catalog(profile: HouseholdProfile) -> list[tuple[str, np.ndarray]]:
    """Named target weights; duplicates (exact) are dropped, first name kept."""
    raw = [
        ("B1_all_kzt", np.array([1.0, 0.0, 0.0])),
        ("B2_all_usd", np.array([0.0, 1.0, 0.0])),
        ("B3_all_eur", np.array([0.0, 0.0, 1.0])),
        ("B4_equal", np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])),
        (
            "B5_expense_matched",
            np.array([profile.qE_kzt, profile.qE_usd, profile.qE_eur]),
        ),
        (
            "B6_income_matched",
            np.array([profile.qI_kzt, profile.qI_usd, profile.qI_eur]),
        ),
    ]
    seen: dict[tuple, str] = {}
    out = []
    for name, w in raw:
        key = tuple(np.round(w, 10))
        if key in seen:
            continue
        seen[key] = name
        if abs(w.sum() - 1.0) > 1e-12 or np.any(w < -1e-15):
            raise ValueError(f"invalid weights {name}: {w}")
        out.append((name, w))
    return out
