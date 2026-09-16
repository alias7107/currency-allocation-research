"""Integer basis-point simplex grids for KZT/USD/EUR allocations.

Internal representation uses basis points with TOTAL_BP = 10_000 (= 100%).
Decimal weights are derived only when evaluating the household engine.

1% grid  → step_bp = 100  → 5151 unique portfolios
2% grid  → step_bp = 200  → 1326 unique portfolios
"""

from __future__ import annotations

import numpy as np

TOTAL_BP = 10_000


def generate_simplex_grid(step_bp: int) -> np.ndarray:
    """Complete nonnegative simplex on the integer bp lattice.

    Parameters
    ----------
    step_bp :
        Grid spacing in basis points (100 = 1%, 200 = 2%).

    Returns
    -------
    bp : (n, 3) int ndarray
        Columns (w_kzt_bp, w_usd_bp, w_eur_bp) with exact sum TOTAL_BP.
    """
    step_bp = int(step_bp)
    if step_bp <= 0 or TOTAL_BP % step_bp != 0:
        raise ValueError(f"step_bp={step_bp} must be a positive divisor of {TOTAL_BP}")
    rows: list[tuple[int, int, int]] = []
    for kzt_bp in range(0, TOTAL_BP + 1, step_bp):
        for usd_bp in range(0, TOTAL_BP - kzt_bp + 1, step_bp):
            eur_bp = TOTAL_BP - kzt_bp - usd_bp
            if eur_bp < 0 or eur_bp % step_bp != 0:
                continue
            rows.append((kzt_bp, usd_bp, eur_bp))
    bp = np.asarray(rows, dtype=np.int64)
    if len(bp) != expected_grid_size_bp(step_bp):
        raise RuntimeError(
            f"grid cardinality mismatch: got {len(bp)}, "
            f"expected {expected_grid_size_bp(step_bp)}"
        )
    if not np.all(bp.sum(axis=1) == TOTAL_BP):
        raise RuntimeError("basis-point weights must sum to TOTAL_BP")
    if np.any(bp < 0):
        raise RuntimeError("negative basis-point weights")
    # uniqueness
    if len({(int(a), int(b), int(c)) for a, b, c in bp}) != len(bp):
        raise RuntimeError("duplicate integer weight tuples in grid")
    return bp


def expected_grid_size_bp(step_bp: int) -> int:
    n = TOTAL_BP // int(step_bp)
    return (n + 1) * (n + 2) // 2


def bp_to_weights(bp: np.ndarray) -> np.ndarray:
    """Convert integer bp rows to decimal weights summing to 1."""
    arr = np.asarray(bp, dtype=np.int64)
    return arr.astype(float) / float(TOTAL_BP)


def weights_to_bp(weights: np.ndarray, step_bp: int | None = None) -> np.ndarray:
    """Round decimal weights to nearest integer bp (optionally snapped to step)."""
    w = np.asarray(weights, dtype=float)
    single = w.ndim == 1
    w = np.atleast_2d(w)
    bp = np.rint(w * TOTAL_BP).astype(np.int64)
    # Fix rounding so each row sums exactly to TOTAL_BP
    for i in range(len(bp)):
        diff = int(TOTAL_BP - bp[i].sum())
        if diff != 0:
            j = int(np.argmax(bp[i]))
            bp[i, j] += diff
        if step_bp is not None:
            # snap each component then repair
            bp[i] = (np.rint(bp[i] / step_bp) * step_bp).astype(np.int64)
            bp[i] = np.clip(bp[i], 0, TOTAL_BP)
            diff2 = int(TOTAL_BP - bp[i].sum())
            if diff2 != 0:
                j = int(np.argmax(bp[i]))
                bp[i, j] = int(np.clip(bp[i, j] + diff2, 0, TOTAL_BP))
                # final residual to another free component if needed
                rem = int(TOTAL_BP - bp[i].sum())
                if rem != 0:
                    for k in range(3):
                        if k == j:
                            continue
                        cand = bp[i, k] + rem
                        if 0 <= cand <= TOTAL_BP and (step_bp is None or cand % step_bp == 0):
                            bp[i, k] = cand
                            break
    return bp.reshape(3) if single else bp


def bp_key(bp_row: np.ndarray | tuple[int, int, int]) -> tuple[int, int, int]:
    a, b, c = (int(x) for x in bp_row)
    return (a, b, c)


def assert_unique_bp_keys(bp: np.ndarray) -> None:
    keys = [bp_key(r) for r in np.atleast_2d(bp)]
    if len(keys) != len(set(keys)):
        raise RuntimeError(
            f"duplicate integer bp keys: {len(keys)} rows, {len(set(keys))} unique"
        )


def merge_bp_grids(*grids: np.ndarray) -> np.ndarray:
    """Union of bp grids with integer-key deduplication; order stable by first seen."""
    seen: set[tuple[int, int, int]] = set()
    rows: list[tuple[int, int, int]] = []
    for g in grids:
        for r in np.atleast_2d(g):
            k = bp_key(r)
            if k not in seen:
                seen.add(k)
                rows.append(k)
    out = np.asarray(rows, dtype=np.int64)
    assert_unique_bp_keys(out)
    return out


def neighborhood_mask_bp(
    grid_bp: np.ndarray,
    centers_bp: np.ndarray,
    radius_bp: int,
) -> np.ndarray:
    """L∞ neighborhood in basis-point weight space."""
    grid_bp = np.atleast_2d(grid_bp).astype(np.int64)
    centers_bp = np.atleast_2d(centers_bp).astype(np.int64)
    keep = np.zeros(len(grid_bp), dtype=bool)
    for c in centers_bp:
        d = np.max(np.abs(grid_bp - c), axis=1)
        keep |= d <= int(radius_bp)
    return keep


def local_refined_grid_bp(
    coarse_bp: np.ndarray,
    coarse_cvar: np.ndarray,
    refined_step_bp: int,
    band: float,
    radius_bp: int,
    cvar_zero_tol: float = 1e-8,
) -> np.ndarray:
    """1%-style local refinement neighborhood in integer bp units.

    Includes all coarse centers (boundary-safe) snapped onto the refined lattice.
    """
    coarse_bp = np.atleast_2d(coarse_bp).astype(np.int64)
    coarse_cvar = np.asarray(coarse_cvar, dtype=float)
    cmin = float(np.min(coarse_cvar))
    if cmin <= cvar_zero_tol or not np.isfinite(cmin):
        centers = coarse_bp[coarse_cvar <= max(cmin, 0.0) + cvar_zero_tol]
        if len(centers) == 0:
            centers = coarse_bp[np.argmin(coarse_cvar)][None, :]
    else:
        thresh = cmin * (1.0 + band)
        centers = coarse_bp[coarse_cvar <= thresh + 1e-15]
        if len(centers) == 0:
            centers = coarse_bp[np.argmin(coarse_cvar)][None, :]

    fine = generate_simplex_grid(refined_step_bp)
    mask = neighborhood_mask_bp(fine, centers, radius_bp)
    out = fine[mask]
    # Always include each center snapped to refined lattice
    for c in centers:
        snapped = weights_to_bp(bp_to_weights(c), step_bp=refined_step_bp)
        if snapped.ndim == 1:
            snapped = snapped.reshape(1, 3)
        out = merge_bp_grids(out, snapped)
    assert_unique_bp_keys(out)
    # Boundary centers must remain admissible
    for c in centers:
        if int(np.min(c)) == 0 or int(np.max(c)) == TOTAL_BP:
            snapped = weights_to_bp(bp_to_weights(c), step_bp=refined_step_bp)
            key = bp_key(snapped.reshape(3))
            keys = {bp_key(r) for r in out}
            if key not in keys:
                raise RuntimeError(f"boundary center {key} missing from refined grid")
    return out


# ---------------------------------------------------------------------------
# Backward-compatible float API (decimal weights derived from integer grid)
# ---------------------------------------------------------------------------
def expected_grid_size(step: float) -> int:
    step_bp = int(round(float(step) * TOTAL_BP))
    return expected_grid_size_bp(step_bp)


def simplex_grid(step: float) -> np.ndarray:
    """Nonnegative decimal weights summing to 1 (from integer bp lattice)."""
    step_bp = int(round(float(step) * TOTAL_BP))
    return bp_to_weights(generate_simplex_grid(step_bp))


def neighborhood_mask(
    grid: np.ndarray,
    centers: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Boolean mask using L∞ radius in decimal weight space (via bp)."""
    grid_bp = weights_to_bp(grid)
    centers_bp = weights_to_bp(centers)
    radius_bp = int(round(radius * TOTAL_BP))
    return neighborhood_mask_bp(grid_bp, centers_bp, radius_bp)


def local_refined_grid(
    coarse_grid: np.ndarray,
    coarse_cvar: np.ndarray,
    refined_step: float,
    band: float,
    radius: float,
) -> np.ndarray:
    """Decimal-weight wrapper around :func:`local_refined_grid_bp`."""
    coarse_bp = weights_to_bp(coarse_grid)
    refined_step_bp = int(round(refined_step * TOTAL_BP))
    radius_bp = int(round(radius * TOTAL_BP))
    out_bp = local_refined_grid_bp(
        coarse_bp, coarse_cvar, refined_step_bp, band, radius_bp
    )
    return bp_to_weights(out_bp)


def merge_result_frames(frames: list) -> "object":
    """Merge evaluation DataFrames on integer bp keys; raise if duplicates remain."""
    import pandas as pd

    if not frames:
        return pd.DataFrame()
    surf = pd.concat(frames, ignore_index=True)
    if not {"w_kzt_bp", "w_usd_bp", "w_eur_bp"}.issubset(surf.columns):
        bp = weights_to_bp(surf[["w_kzt", "w_usd", "w_eur"]].to_numpy())
        surf["w_kzt_bp"] = bp[:, 0]
        surf["w_usd_bp"] = bp[:, 1]
        surf["w_eur_bp"] = bp[:, 2]
    before = len(surf)
    surf = surf.drop_duplicates(
        subset=["w_kzt_bp", "w_usd_bp", "w_eur_bp"], keep="last"
    ).reset_index(drop=True)
    keys = list(
        zip(
            surf["w_kzt_bp"].astype(int),
            surf["w_usd_bp"].astype(int),
            surf["w_eur_bp"].astype(int),
        )
    )
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate integer weight keys after merge")
    if len(surf) != len(set(keys)):
        raise RuntimeError("len(rows) != len(unique_integer_weight_keys)")
    # Sync decimal weights from bp (canonical)
    surf["w_kzt"] = surf["w_kzt_bp"].astype(float) / TOTAL_BP
    surf["w_usd"] = surf["w_usd_bp"].astype(float) / TOTAL_BP
    surf["w_eur"] = surf["w_eur_bp"].astype(float) / TOTAL_BP
    return surf
