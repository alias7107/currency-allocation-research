# 03 — Phase 2B1: household engine

См. мастер §7,15–18; код: `src/household/`.

## Entry

`scripts/run_phase2b1.py` → `run_phase2b1()`  
**Файл:** `src/household/calibration.py:354–378`

## FX paths

`generate_common_fx_paths` (`calibration.py:32–46`):
- `StationaryBootstrapSimulator(expected_block_length=HH_BLOCK_LENGTH=20)`
- `simulate(10000, 1260, seed=42)`
- `returns_to_levels` → cache `sb20_daily_levels.npy`

Один набор путей для **всех** household/strategy комбинаций (CRN).

## Профили

`src/household/profiles.py` — dataclass shares, сумма = 1.  
`strategy_catalog` — B1…B6 с dedup точных дубликатов (напр. P1: expense=income=all-KZT).

## Cashflows

`nominal_cashflows` (`cashflows.py:38–58`):
```
I_usd = qI_usd * I0 / S_usd0
E_eur = qE_eur * E0 / S_eur0
```

`daily_paths_to_monthly_rates` (`cashflows.py:70–87`):
- вход `(N, days+1, 2)`
- month m → index `(m+1)*21`
- 756 days → 36 months → `(N, 36, 2)`

## Settlement (критично)

`simulate_household` (`settlement.py:81–232`):

Порядок месяца: FX → income → native expense pay → pro-rata liquidation → shortfall → optional rebalance.

Pro-rata: `frac = (take + fee) / W`; holdings *= (1-frac).  
Shortfall = непокрытый gap в KZT-эквиваленте.  
Нет займов; negative holdings запрещены.

Conservation:
`wealth_before + income_kzt - funded_expenses - liq_fee = wealth_after`  
(тест: `tests/test_accounting_conservation.py`).

## Risk

`empirical_cvar` (`risk_metrics.py:28–40`): mean of largest `ceil((1-α)N)` losses.  
N=10000, α=0.95 → k=500.

## Stress calibration

`run_stress_calibration` (`calibration.py:49–110`):  
horizons × profiles × E/I × B × strategies → `stress_calibration.csv`.

`select_baseline_candidate` (`:199–248`):  
правило P1≤5% и FX в 5–40%; сортировка предпочитает horizon=3, E/I≈0.95, **B≈6**.

**Фактический Phase 2B2 env:** `OPT_EXPENSE_INCOME_RATIO=0.95`, `OPT_BUFFER_MONTHS=1` в config — задан явно, не автозагрузкой кандидата.

## Выходы

- `results/household/tables/stress_calibration.csv`
- `results/household/diagnostics/calibration_report.md`
- figures `figA`…`figE`
- cache `sb20_daily_levels.npy`
