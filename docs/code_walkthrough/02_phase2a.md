# 02 — Phase 2A: joint FX simulation

См. мастер §5–6,13; код: `src/simulation/`.

## Entry

`scripts/run_phase2a.py` → `run_phase2a()`  
**Файл:** `src/simulation/model_comparison.py:773–823`

## Шаги

1. `load_baseline_bundle()` (`validation.py:168–185`)  
   Читает `fx_daily_2010_2025.csv`, фильтр `[BASELINE_START, BASELINE_END]`, dropna returns.  
   - `returns`: shape `(T, 2)` = `[usd_kzt_log_return, eur_kzt_log_return]`  
   - `s0`: последние уровни окна (shape `(2,)`)

2. `build_models()` — словарь моделей:
   - `IIDBootstrapSimulator`
   - `StationaryBootstrapSimulator(L)` для L∈{5,10,20}
   - `MovingBlockBootstrapSimulator(10)`
   - Student-t / Gaussian (`student_t.py`)

3. `fit_all` → `simulate_all(..., n_paths=10000, horizon=252, seed=42)`  
   Каждая модель → array `(10000, 252, 2)`.

4. Validation:
   - `marginal_validation`, `tail_validation`, `dependence_validation`
   - `temporal_validation`, `vol_clustering_validation`, `multihorizon_validation`
   - `pseudo_oos_table`, `shock_sensitivity_table`, `convergence_table`

5. `score_models` → ranking; `recommend_block_length` среди L∈{5,10,20}.

6. Politis–White heuristic `politis_white_mean_block_length(|r_USD|)` → JSON (диагностика; ~56).

7. `write_report` → `results/simulation/diagnostics/model_comparison_report.md`.

## Stationary bootstrap (ядро)

**Файл:** `src/simulation/stationary_bootstrap.py:35–64, 113–122`

```
p = 1/L
idx[0] = U ~ Uniform{0..n-1}
for t:
  with prob 1-p: idx[t] = (idx[t-1]+1) mod n
  with prob p:   idx[t] = new U
out = history[idx]   # history rows are (r_USD, r_EUR)
```

Seed через `np.random.default_rng` (`base.py:35–37`).

## Levels

`returns_to_levels` (`path_builder.py:8–31`):  
`S_t = S_0 * exp(cumsum r)` → shape `(N, H+1, 2)`.

## Важно для следующих фаз

Household/optimization **не** перечитывают recommendation автоматически.  
Они используют `HH_BLOCK_LENGTH = 20` и кэш `results/household/cache/sb20_daily_levels.npy`.

## Выходы

`results/simulation/tables/*_validation.csv`, figures, `model_comparison_report.md`, `phase2a_run_summary.json`.
