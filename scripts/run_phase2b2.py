from src.optimization.pipeline import run_phase2b2

if __name__ == "__main__":
    summary = run_phase2b2(run_full_1pct_check=True)
    print("=" * 72)
    print("OPTIMIZATION ENVIRONMENT")
    from src import config

    print(f"- DGP: joint stationary bootstrap L={config.OPT_BLOCK_LENGTH}")
    print(f"- seed={config.OPT_RANDOM_SEED}, paths={config.OPT_N_PATHS}")
    print(f"- E/I={config.OPT_EXPENSE_INCOME_RATIO}, B={config.OPT_BUFFER_MONTHS}")
    print(f"- horizons={config.OPT_HORIZON_YEARS}")
    print("GRID")
    from src.optimization.grid import expected_grid_size

    print(f"- coarse step={config.OPT_COARSE_STEP}, portfolios={expected_grid_size(config.OPT_COARSE_STEP)}")
    print(f"- refined step={config.OPT_REFINED_STEP}, local neighborhoods ±{config.OPT_REFINE_NEIGHBORHOOD}")
    print("OPTIMAL ALLOCATIONS")
    for rec in summary["optimal"]:
        print(
            f"  {rec['profile']} {rec['horizon_years']}Y: "
            f"({rec['w_kzt']:.2f},{rec['w_usd']:.2f},{rec['w_eur']:.2f}) "
            f"CVaR={rec['cvar95']:.6g} P(SF)={rec['shortfall_probability']:.4f} "
            f"{rec['identification_class']}"
        )
    print("50K CONFIRMATION stable flags:", summary.get("confirm_stable"))
    print("reproduction_ok:", summary.get("reproduction_ok"))
    print("global_1pct_check:", summary.get("global_1pct_check"))
    print("=" * 72)
    print(summary)
