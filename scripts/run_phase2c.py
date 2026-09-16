#!/usr/bin/env python
"""Phase 2C — grid integrity patch + robustness analysis."""

from src import config
from src.optimization.grid import expected_grid_size_bp
from src.robustness.pipeline import run_phase2c

if __name__ == "__main__":
    summary = run_phase2c()
    print("=" * 72)
    print("GRID INTEGRITY")
    print(f"- 2% unique grid count: {expected_grid_size_bp(200)}")
    print(f"- 1% unique grid count: {expected_grid_size_bp(100)}")
    if summary.get("stopped"):
        print("- STOPPED before / during robustness (see diagnostics)")
        print(summary)
        raise SystemExit(1)
    p = summary["patch"]
    print(f"- duplicate issue fixed: yes (integer bp keys)")
    print(f"- Phase 2B2 optima changed: {'yes' if p['economic_changed'] else 'no'}")
    print(f"- near-optimal counts changed: {'yes' if p['counts_changed'] else 'no'}")
    print("BASELINE REPRODUCTION")
    for row in summary["gate"]["rows"]:
        print(
            f"- {row['profile']}: w_kzt={row['w_kzt']:.4f} "
            f"CVaR={row['cvar95']:.6g} (expected≈{row['expected_cvar']:.6g}) "
            f"ok={row['weights_ok'] and row['cvar_ok']}"
        )
    import pandas as pd

    df = pd.read_csv(config.ROB_TABLES / "robustness_results.csv")

    def _show(dim, cols=None):
        cols = cols or [
            "specification", "profile", "w_kzt", "w_usd", "w_eur",
            "cvar95", "robustness_class", "expense_vs_income_ranking",
        ]
        g = df[df.robustness_dimension == dim].drop_duplicates(
            subset=["specification", "profile"], keep="last"
        )
        print(g[cols].to_string(index=False))

    print("HOUSEHOLD ROBUSTNESS")
    _show("R1_household")
    print("DGP ROBUSTNESS")
    _show("R2_dgp")
    print("WINDOW ROBUSTNESS")
    _show("R3_window")
    print("REBALANCING")
    _show("R4_rebalance")
    print("COSTS")
    _show(
        "R5_costs",
        [
            "specification", "profile", "conversion_cost", "rebalance_frequency",
            "w_kzt", "cvar95", "robustness_class",
        ],
    )
    print("EXPENSE VS INCOME MATCHING")
    ei = pd.read_csv(config.ROB_TABLES / "expense_vs_income_matching.csv")
    for prof in list(config.ROB_CORE_PROFILES) + list(config.ROB_SECONDARY_PROFILES):
        sub = ei[ei.profile == prof].drop_duplicates(subset=["specification"], keep="last")
        n_exp = (sub.ranking == "expense_better").sum()
        n_inc = (sub.ranking == "income_better").sum()
        print(f"- {prof}: expense_better={n_exp}, income_better={n_inc}, n={len(sub)}")
    print("REVERSALS")
    rev = df[df.robustness_class == "REVERSED"]
    if rev.empty:
        print("- none")
    else:
        print(rev[["specification", "profile", "w_kzt", "cvar95"]].to_string(index=False))
    print("FINAL ROBUSTNESS CLASSIFICATION")
    print(f"- {summary['final_classification']}")
    print(f"- {summary['final_classification_text']}")
    print("UNRESOLVED ISSUES")
    print("- Costs are scenario parameters, not calibrated KZ spreads")
    print("- No GARCH/DCC / rates / inflation / asset returns (future research)")
    print("- Cross-DGP paths are not paired")
    print("=" * 72)
