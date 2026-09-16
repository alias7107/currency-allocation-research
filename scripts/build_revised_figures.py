#!/usr/bin/env python3
"""Build three revised publication figures (P2/P4 surfaces + P2 tail distribution)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.household.profiles import PRIMARY_PROFILES
from src.optimization.evaluator import (
    evaluate_weights,
    load_or_generate_fx_levels,
    monthly_fx_for_horizon,
    primary_cashflows,
)

OUT_DIR = PROJECT_ROOT / "results" / "final" / "figures" / "revised"
COARSE_PATH = config.OPT_TABLES / "coarse_grid_results.csv"
REFINED_PATH = config.OPT_TABLES / "refined_grid_results.csv"


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _load_surface_grid(profile: str, horizon_years: int = 3) -> pd.DataFrame:
    coarse = pd.read_csv(COARSE_PATH)
    refined = pd.read_csv(REFINED_PATH)
    parts = []
    for df in (coarse, refined):
        sub = df[(df.profile == profile) & (df.horizon_years == horizon_years)].copy()
        parts.append(sub)
    g = pd.concat(parts, ignore_index=True)
    g = g.drop_duplicates(subset=["w_kzt", "w_usd", "w_eur"])
    g = g[g.w_usd + g.w_eur <= 1.0 + 1e-9]
    return g.sort_values(["w_usd", "w_eur"]).reset_index(drop=True)


def _mask_infeasible(ax, n: int = 200) -> None:
    usd = np.linspace(0, 1, n)
    eur = np.linspace(0, 1, n)
    U, E = np.meshgrid(usd, eur)
    mask = U + E > 1.0
    ax.contourf(U, E, mask.astype(float), levels=[0.5, 1.5], colors=["#f0f0f0"], alpha=1.0, zorder=0)


def _plot_cvar_surface(
    profile: str,
    tag: str,
    benchmarks: list[tuple[str, str, float, float, float]],
    out_name: str,
) -> dict:
    g = _load_surface_grid(profile, horizon_years=3)
    cmin = float(g.cvar95.min())
    thr1 = 1.01 * cmin
    thr5 = 1.05 * cmin

    near1_kzt_min = float(g.loc[g.cvar95 <= thr1 + 1e-12, "w_kzt"].min())

    fig, ax = plt.subplots(figsize=(6.2, 5.2))

    _mask_infeasible(ax)

    tri = mtri.Triangulation(g.w_usd.values, g.w_eur.values)
    tcf = ax.tricontourf(
        tri,
        g.cvar95.values,
        levels=20,
        cmap="viridis_r",
        extend="max",
        zorder=1,
    )
    cbar = fig.colorbar(tcf, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("CVaR$_{95}$")

    # Near-optimal region boundaries from grid data (not hardcoded shapes)
    ax.tricontour(
        tri,
        g.cvar95.values,
        levels=[thr1],
        colors=["white"],
        linewidths=[2.0],
        linestyles=["-"],
        zorder=2,
    )
    ax.tricontour(
        tri,
        g.cvar95.values,
        levels=[thr5],
        colors=["black"],
        linewidths=[1.2],
        linestyles=["--"],
        zorder=2,
    )

    marker_map = {"star": "*", "triangle": "^", "circle": "o", "x": "x"}
    for label, mkind, w_kzt, w_usd, w_eur in benchmarks:
        kw = dict(
            marker=marker_map[mkind],
            s=120 if mkind == "star" else 90,
            zorder=4,
            label=label,
        )
        if mkind == "x":
            kw.update(c="black", linewidths=1.2)
        else:
            kw.update(c="white", edgecolors="black", linewidths=1.0)
        ax.scatter([w_usd], [w_eur], **kw)

    ax.plot([0, 1, 0, 0], [0, 0, 1, 0], "k-", lw=0.6, zorder=3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("USD weight ($w_{\\mathrm{USD}}$)")
    ax.set_ylabel("EUR weight ($w_{\\mathrm{EUR}}$)")
    ax.set_aspect("equal")

    legend_handles = [
        Line2D([0], [0], color="white", lw=2.0, label="+1% near-optimal"),
        Line2D([0], [0], color="black", lw=1.2, ls="--", label="+5% near-optimal"),
    ]
    leg1 = ax.legend(handles=legend_handles, loc="upper left", frameon=True, framealpha=0.95)
    ax.add_artist(leg1)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)

    fig.tight_layout()
    out = OUT_DIR / out_name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    return {
        "profile": tag,
        "n_grid_points": len(g),
        "cvar_min": cmin,
        "thr1": thr1,
        "thr5": thr5,
        "near1_kzt_min": near1_kzt_min,
        "near1_count": int((g.cvar95 <= thr1 + 1e-12).sum()),
        "near5_count": int((g.cvar95 <= thr5 + 1e-12).sum()),
        "output": str(out),
    }


def _load_p2_tail_losses() -> tuple[np.ndarray, float, float, float]:
    """Regenerate canonical P2 3Y all-KZT scenario losses (same path as original tail figure)."""
    daily, s0 = load_or_generate_fx_levels()
    monthly = monthly_fx_for_horizon(daily, 3)
    p2 = next(p for p in PRIMARY_PROFILES if p.name == "P2_foreign_income_mismatch")
    cf, _e0, w0 = primary_cashflows(p2, s0)
    ev = evaluate_weights(monthly, cf, w0, np.array([1.0, 0.0, 0.0]))
    losses = ev.path_losses

    p_short = float(np.mean(losses > 0))
    var95 = float(ev.var95)
    cvar95 = float(ev.cvar95)

    expected = {"p_short": 0.0491, "var95": 0.0, "cvar95": 1.1871802686857733}
    checks = {
        "p_short": np.isclose(p_short, expected["p_short"], rtol=0, atol=1e-4),
        "var95": np.isclose(var95, expected["var95"], rtol=0, atol=1e-12),
        "cvar95": np.isclose(cvar95, expected["cvar95"], rtol=0, atol=1e-6),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"P2 3Y reproduction mismatch: got P={p_short}, VaR={var95}, CVaR={cvar95}; "
            f"expected {expected}; checks={checks}"
        )
    return losses, p_short, var95, cvar95


def _plot_p2_tail_revised() -> dict:
    """Frequency-severity decomposition (revised layout; does not overwrite original PDF)."""
    losses, p_short, var95, cvar95 = _load_p2_tail_losses()

    p_no = (1.0 - p_short) * 100.0
    p_yes = p_short * 100.0
    pos = losses[losses > 0]
    n_pos = len(pos)
    tail_k = max(int(np.ceil(0.05 * len(losses))), 1)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # Left: shortfall frequency
    ax = axes[0]
    labels = ["No shortfall", "Positive shortfall"]
    values = [p_no, p_yes]
    colors = ["#ececec", "#6e6e6e"]
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6, width=0.52)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Share of simulated scenarios (%)")
    ax.set_title("(a) Shortfall frequency")
    ax.set_ylim(0, 108)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: severity among positive-loss paths
    ax = axes[1]
    if n_pos:
        ax.hist(
            pos,
            bins=min(40, max(10, n_pos // 8)),
            color="#cccccc",
            edgecolor="black",
            linewidth=0.4,
        )
    ax.set_xlabel("Cumulative shortfall $L$ (positive paths only)")
    ax.set_ylabel("Count")
    ax.set_title("(b) Shortfall severity conditional on $L > 0$")
    ax.text(
        0.03,
        0.97,
        f"$P(L>0)$ = {p_yes:.2f}%\nVaR$_{{95}}$ = {var95:.4f}\nCVaR$_{{95}}$ = {cvar95:.4f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", lw=0.5),
    )
    ax.text(
        0.03,
        0.72,
        f"CVaR$_{{95}}$ averages the worst 5\\% of all {len(losses):,} paths",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="#444444",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_pdf = OUT_DIR / "figure_p2_3y_shortfall_tail_revised.pdf"
    out_png = OUT_DIR / "figure_p2_3y_shortfall_tail_revised.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)

    worst_tail = np.sort(losses)[::-1][:tail_k]
    n_pos_in_tail = int(np.sum(worst_tail > 0))
    n_zero_in_tail = tail_k - n_pos_in_tail

    return {
        "scenario_losses_existed": False,
        "reproduced_from": "sb20_daily_levels.npy cache + evaluate_weights(all-KZT)",
        "p_shortfall": p_short,
        "var95": var95,
        "cvar95": cvar95,
        "n_paths": len(losses),
        "n_positive": n_pos,
        "tail_k": tail_k,
        "n_pos_in_worst_tail": n_pos_in_tail,
        "n_zero_in_worst_tail": n_zero_in_tail,
        "output_pdf": str(out_pdf),
        "output_png": str(out_png),
    }


def _plot_p2_tail() -> dict:
    losses, p_short, var95, cvar95 = _load_p2_tail_losses()

    pos = np.sort(losses[losses > 0])
    n_pos = len(pos)
    tail_k = max(int(np.ceil(0.05 * len(losses))), 1)
    tail_threshold = float(np.sort(losses)[::-1][tail_k - 1])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # Left: CCDF emphasizing mass at zero
    ax = axes[0]
    sorted_loss = np.sort(losses)
    n = len(sorted_loss)
    ccdf = 1.0 - np.arange(1, n + 1) / n
    ax.step(sorted_loss, ccdf, where="post", color="black", lw=1.2)
    ax.axvline(0, color="#888888", ls=":", lw=0.8)
    ax.set_xlabel("Cumulative shortfall $L$")
    ax.set_ylabel("$P(L > x)$")
    ax.set_xlim(left=-0.02)
    ax.set_ylim(0, 1.02)
    ax.annotate(
        f"$P(L>0)$ = {p_short:.2%}",
        xy=(0, 1 - p_short),
        xytext=(0.05, 0.55),
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
        fontsize=9,
    )

    # Right: tail among positive losses + VaR/CVaR markers
    ax = axes[1]
    if n_pos:
        ax.hist(pos, bins=min(40, max(10, n_pos // 8)), color="#cccccc", edgecolor="black", lw=0.4)
    ax.axvline(var95, color="black", ls="--", lw=1.0, label=f"VaR$_{{95}}$ = {var95:.4f}")
    ax.axvline(cvar95, color="black", ls="-", lw=1.2, label=f"CVaR$_{{95}}$ = {cvar95:.4f}")
    if tail_threshold > 0:
        ax.axvspan(tail_threshold, max(pos.max() * 1.05, cvar95 * 1.05), color="#e0e0e0", alpha=0.7, label="Worst 5% tail")
    ax.set_xlabel("Cumulative shortfall $L$  (positive paths only)")
    ax.set_ylabel("Count")
    ax.legend(frameon=True, fontsize=8, loc="upper right")
    ax.text(
        0.03,
        0.97,
        f"$P(L>0)$ = {p_short:.2%}\nVaR$_{{95}}$ = {var95:.4f}\nCVaR$_{{95}}$ = {cvar95:.4f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", lw=0.5),
    )

    fig.tight_layout()
    out = OUT_DIR / "figure_p2_3y_shortfall_tail.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    return {
        "scenario_losses_existed": False,
        "reproduced_from": "sb20_daily_levels.npy cache + evaluate_weights(all-KZT)",
        "p_shortfall": p_short,
        "var95": var95,
        "cvar95": cvar95,
        "n_paths": len(losses),
        "n_positive": n_pos,
        "output": str(out),
    }


def main() -> None:
    _setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    p2_info = _plot_cvar_surface(
        profile="P2_foreign_income_mismatch",
        tag="P2",
        benchmarks=[
            ("Optimum (100% KZT)", "star", 1.0, 0.0, 0.0),
            ("Income matched (100% USD)", "triangle", 0.0, 1.0, 0.0),
            ("Expense matched (100% KZT)", "circle", 1.0, 0.0, 0.0),
            ("Equal weight", "x", 1 / 3, 1 / 3, 1 / 3),
        ],
        out_name="figure_p2_3y_cvar_surface_revised.pdf",
    )

    p4_info = _plot_cvar_surface(
        profile="P4_full_multi_currency",
        tag="P4",
        benchmarks=[
            ("Optimum (100% KZT)", "star", 1.0, 0.0, 0.0),
            ("Income matched (50% USD, 50% EUR)", "triangle", 0.0, 0.5, 0.5),
            ("Expense matched (80% KZT, 20% EUR)", "circle", 0.8, 0.0, 0.2),
            ("Equal weight", "x", 1 / 3, 1 / 3, 1 / 3),
        ],
        out_name="figure_p4_3y_cvar_surface_revised.pdf",
    )

    tail_info = _plot_p2_tail()

    print("=== Revised figure build report ===")
    print("\nFigure 1 (P2 surface):")
    print(f"  Sources: {COARSE_PATH.name}, {REFINED_PATH.name}")
    print(f"  Grid points: {p2_info['n_grid_points']}, CVaR min: {p2_info['cvar_min']:.6f}")
    print(f"  +1% near-optimal: n={p2_info['near1_count']}, min KZT={p2_info['near1_kzt_min']:.2f}")
    print(f"  Output: {p2_info['output']}")

    print("\nFigure 2 (P4 surface):")
    print(f"  Sources: {COARSE_PATH.name}, {REFINED_PATH.name}")
    print(f"  Grid points: {p4_info['n_grid_points']}, CVaR min: {p4_info['cvar_min']:.6f}")
    print(f"  +1% near-optimal: n={p4_info['near1_count']}, min KZT={p4_info['near1_kzt_min']:.2f}")
    print(f"  Output: {p4_info['output']}")

    print("\nFigure 3 (P2 tail):")
    print(f"  Scenario-level losses existed: {tail_info['scenario_losses_existed']}")
    print(f"  Reproduced via: {tail_info['reproduced_from']}")
    print(
        f"  Verified: P(L>0)={tail_info['p_shortfall']:.4f}, "
        f"VaR95={tail_info['var95']}, CVaR95={tail_info['cvar95']:.6f}"
    )
    print(f"  Output: {tail_info['output']}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--tail-revised-only":
        _setup_style()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        tail_info = _plot_p2_tail_revised()
        print("=== P2 3Y shortfall tail (revised) ===")
        print(f"  n_paths: {tail_info['n_paths']}")
        print(
            f"  P(L>0)={tail_info['p_shortfall']:.4f}, "
            f"VaR95={tail_info['var95']}, CVaR95={tail_info['cvar95']:.4f}"
        )
        print(f"  Worst 5% tail: {tail_info['n_pos_in_worst_tail']} with L>0, "
              f"{tail_info['n_zero_in_worst_tail']} with L=0")
        print(f"  PDF: {tail_info['output_pdf']}")
        print(f"  PNG: {tail_info['output_png']}")
    else:
        main()
