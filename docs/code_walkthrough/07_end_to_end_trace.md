# 07 — End-to-end trace + toy example

См. мастер §10–11.

## Полный след (реальные функции)

```
NBK Excel USD/KZT cell
  → data_cleaning.parse_nbk_file / load_and_merge_raw
  → returns.build_daily_processed  (weekday log return)
  → simulation.validation.load_baseline_bundle  → hist[t, 0]
  → StationaryBootstrapSimulator.simulate       → r[s, d, 0]
  → path_builder.returns_to_levels              → S[s, d, 0]
  → cashflows.daily_paths_to_monthly_rates      → S_m[s, m, 0]
  → cashflows.nominal_cashflows                 → I_usd, E_kzt, …
  → settlement.simulate_household               → monthly_shortfall
  → risk_metrics.cumulative_shortfall           → L[s]
  → risk_metrics.empirical_cvar                 → CVaR(w)
  → optimization.grid.generate_simplex_grid
  → optimization.evaluator.evaluate_grid
  → near_optimal.pick_argmin / classify_identification
  → robustness.classify.final_paper_classification → RESULT B
```

## Размерности (запомнить)

| Array | Shape | Смысл |
|-------|-------|-------|
| Historical returns | `(T, 2)` | T≈2705 weekdays, 2 currencies |
| Simulated returns | `(N, H, 2)` | N paths, H days |
| Levels | `(N, H+1, 2)` | includes t=0 |
| Monthly FX | `(N, M, 2)` | M=H/21 |
| Path losses | `(N,)` | one L per scenario |
| Weight bp | `(G, 3)` | G=1326 or 5151 |

Индексация валют: `0=USD/KZT`, `1=EUR/KZT` почти везде в симуляции.  
Веса портфеля: `(w_kzt, w_usd, w_eur)`.

## EDUCATIONAL TOY (не эмпирика)

См. полный расчёт в `COMPLETE_CODE_WALKTHROUGH_RU.md` §11.

Кратко:
- P2, I0=1, E0=0.95, W0=0.95, S0=500, `I_usd=0.002`
- All-KZT: после native gaps часто покрываются без FX stress на спокойных путях
- All-USD: каждый месяц нужен FX conversion для KZT expenses → выше shortfall risk при неблагоприятном S

Логика совпадает с `simulate_household` при `conversion_cost=0` и monthly rebalance.
