# Phase input/output — быстрая шпаргалка

Команды, главные файлы, что читается vs config. Подробности → `08_data_lineage_and_files.md`.

---

## PHASE 1 — FX diagnostics

**Команда:** `PYTHONPATH=. python scripts/run_phase1.py`  
**Entry:** `src/pipeline.py::run_phase1()`

**INPUT:**
- `data/raw/nbk_official_rates_YYYY.xlsx` (NBK)
- `src/config.py` (окна, weekday rule, NBK IDs)

**IN-MEMORY:** merged panel → daily weekday returns

**OUTPUT (core):**
- `data/processed/fx_daily_2010_2025.csv` ← **главный для всех фаз**
- `data/processed/fx_monthly_2010_2025.csv`

**OUTPUT (tables/diagnostics):**
- `results/tables/descriptive_statistics.csv`, `distribution_tests.csv`, …
- `results/diagnostics/raw_file_manifest.csv`, `data_quality_report.md`
- `results/figures/fig01_* … fig12_*`

**Дальше читает:** Phase 2A/2B1/2C через `load_baseline_bundle()` → processed CSV

---

## PHASE 2A — Simulation model comparison

**Команда:** `PYTHONPATH=. python scripts/run_phase2a.py`  
**Entry:** `src/simulation/model_comparison.py::run_phase2a()`

**INPUT:**
- `data/processed/fx_daily_2010_2025.csv` (baseline filter 2015-08-20 – 2025-12-31)
- `config.SIM_RANDOM_SEED=42`, `STATIONARY_BLOCK_LENGTHS=(5,10,20)`

**IN-MEMORY:** fitted models; 10k×252 sim paths — **НЕ сохраняются**

**OUTPUT (core):**
- `results/simulation/diagnostics/phase2a_run_summary.json` (`recommended_L=20`)
- `results/simulation/tables/model_scores.csv`

**OUTPUT (validation tables):** marginal, tail, dependence, temporal, vol, multihorizon, pseudo_oos, float_shock

**Дальше читает:** **ничего напрямую** — L=20 и seed зашиты в config `HH_BLOCK_LENGTH=20`; paths генерируются заново в 2B1

---

## PHASE 2B1 — Household calibration

**Команда:** `PYTHONPATH=. python scripts/run_phase2b1.py`  
**Entry:** `src/household/calibration.py::run_phase2b1()`

**INPUT:**
- processed CSV (via `load_baseline_bundle()`)
- **config only:** L=20, seed=42, profiles, E/I∈{0.8,0.95,1.05}, B∈{1,3,6,12}, horizons 1/3/5Y
- **НЕ читает** Phase 2A output files

**IN-MEMORY:** SB20 paths (10k×1260); 684 household sims

**OUTPUT (core):**
- `results/household/cache/sb20_daily_levels.npy` ← **FX cache для 2B2/2C**
- `results/household/tables/stress_calibration.csv` (684 rows)

**OUTPUT (docs):**
- `results/household/diagnostics/phase2b1_run_summary.json` (candidate B=1 — **документ**, не runtime)
- `calibration_report.md`, figures figA–figE

**Дальше читает 2B2:**
- `.npy` cache (via `load_or_generate_fx_levels`)
- `stress_calibration.csv` — **только verify**, не baseline selection
- Baseline E/I=0.95, B=1 — из **`config.OPT_*`**, не из JSON

---

## PHASE 2B2 — CVaR optimization

**Команда:** `PYTHONPATH=. python scripts/run_phase2b2.py`  
**Entry:** `src/optimization/pipeline.py::run_phase2b2()`

**INPUT:**
- `results/household/cache/sb20_daily_levels.npy` (or regenerate seed 42)
- `config`: E/I=0.95, B=1, grid 2%+1% bp, CVaR95, monthly rebalance, cost=0
- `src/household/profiles.py`
- `stress_calibration.csv` — verify only

**IN-MEMORY:** coarse+refined grid eval; path losses for bootstrap

**OUTPUT (core):**
- `results/optimization/tables/optimal_allocations.csv` — **LEGACY n_surface**
- `results/optimization/tables/benchmark_comparison.csv` — **OK for metrics**
- `coarse_grid_results.csv`, `refined_grid_results.csv`

**OUTPUT (extra):** `optimization_uncertainty.csv`, `confirmation_50k.csv`, `global_1pct_check_*.csv`

**Дальше читает 2C:**
- `optimal_allocations.csv` — compare in grid_patch (не overwrite)
- Пересчитывает optima самостоятельно

---

## PHASE 2C — Grid patch + robustness

**Команда:** `PYTHONPATH=. python scripts/run_phase2c.py`  
**Entry:** `src/robustness/pipeline.py::run_phase2c()`

**INPUT:**
- baseline: `sb20_daily_levels.npy`
- `optimal_allocations.csv` (old, for patch diff)
- processed CSV (alt windows / alt DGPs)
- config: R1–R5 scenario params, seeds

**OUTPUT (canonical):**
- `results/robustness/grid_patch/corrected_optimal_allocations.csv`
- `results/robustness/grid_patch/grid_patch_comparison.csv`
- `results/robustness/tables/robustness_results.csv`
- `robustness_summary.csv`, sensitivity slices
- `phase2c_run_summary.json` (RESULT B)

**OUTPUT (cache):** `results/robustness/cache/{SB10,SB56,IID,recent,confirm50k}_*.npy`

**Robustness matrix:**

| Case | Changed param | Key output spec |
|------|---------------|-----------------|
| R0 | baseline | `baseline_SB20_EI95_B1_monthly_c0` |
| R1 | E/I=1.05, B=3 | `EI105_B3` |
| R2 | SB10/SB56/IID | `SB10`, `SB56`, `IID` |
| R3 | window 2020–2025 | `recent_2020_2025_SB20` |
| R4 | rebalance q/none | `rebalance_quarterly_c0`, `rebalance_none_c0` |
| R5 | cost 0.25–1% | `cost_0.0025_monthly`, … |

---

## FINAL — Publication freeze

**Команда:** `PYTHONPATH=. .venv/bin/python scripts/run_final_cleanup.py`  
**Entry:** `src/final/pipeline.py::run_final_cleanup()`

**INPUT (read-only aggregation):**
- `corrected_optimal_allocations.csv` → `table_optimal_allocations.csv`
- `benchmark_comparison.csv` → `table_benchmark_3y.csv`
- `robustness_results.csv` → `table_robustness_summary.csv`
- `descriptive_statistics.csv` → `table_fx_descriptive_statistics.csv`
- `model_scores.csv` → `table_simulation_model_comparison.csv`
- config + phase summaries → `canonical_specification.json`

**OUTPUT:** `results/final/` (tables, figures, audit, manifest, claims, summary)

**Не перезапускает** Phase 1–2C (кроме lightweight audit sims).

---

## Одна строка: что куда течёт

```
raw xlsx → processed CSV → [2A tables] + [2B1 npy + calibration] → [2B2 LEGACY opt] → [2C corrected + robustness] → results/final/
```

## Canonical vs LEGACY (для статьи)

| Использовать | Не использовать |
|--------------|-----------------|
| `results/final/tables/*` | `optimal_allocations.csv` (n_surface) |
| `corrected_optimal_allocations.csv` | `horizon_allocation_comparison.csv` |
| `benchmark_comparison.csv` (metrics OK) | Phase 1–2C figures (есть final figures) |
| `robustness_results.csv` | — |

## Критично помнить

1. **Paths:** 2A не сохраняет; 2B1 сохраняет `.npy`; 2B2/2C читают cache.
2. **Baseline B=1:** config, не JSON Phase 2B1.
3. **Processed CSV** — единственный FX-файл между Phase 1 и всем остальным.
