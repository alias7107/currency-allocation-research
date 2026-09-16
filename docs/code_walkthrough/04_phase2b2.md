# 04 — Phase 2B2: CVaR optimization

См. мастер §8,20–29; код: `src/optimization/`.

## Entry

`scripts/run_phase2b2.py` → `run_phase2b2()`  
**Файл:** `src/optimization/pipeline.py` (функция в конце файла)

## Что оптимизатор делает / не делает

**Делает:** полный (coarse+refine) перебор весов на simplex, для каждого веса — household MC, считает CVaR95(L), берёт argmin.  
**Не делает:** ML, нейросети, максимизацию terminal wealth.

## Контрольный поток

1. `load_or_generate_fx_levels` — предпочитает Phase 2B1 cache  
2. `verify_phase2b1_benchmarks` — STOP при mismatch  
3. `optimize_all_cells` — 4 профиля × 3 горизонта  
4. `run_uncertainty` — paired bootstrap 1000  
5. `run_50k_confirmation` — P2/P4 3Y  
6. optional full 1% check P2/P4 3Y  
7. figures + `optimization_report.md`

## Grid (канон после 2C patch)

`generate_simplex_grid(step_bp)` (`grid.py:17–53`):
- integer bp, sum=10000
- step 200 → 1326; step 100 → 5151
- merge только по `(w_kzt_bp, w_usd_bp, w_eur_bp)`

Старый float-bug (`n_surface=5280`) задокументирован в Phase 2C `grid_patch_report.md`; текущий `grid.py` его не воспроизводит.

## Coarse / refine

`local_refined_grid_bp`: центры = coarse points с CVaR в пределах `OPT_REFINE_CVAR_BAND` (или все нули); радиус `±500` bp; boundary centers включаются.

`pick_argmin`: min CVaR, tie-break max KZT.

## Near-optimal / identification

`near_optimal.py`:
- relative 1%/5% bands
- absolute band if CVaR*≈0
- A/B/C по `frac` порогам из config

## Uncertainty

`paired_bootstrap_cvar`: одни и те же resampled indices для всех labels → CI на разности.  
Это **MC/scenario** uncertainty.

## Ключевые выходы

| Файл | Роль |
|------|------|
| `optimal_allocations.csv` | w*, CVaR, identification, near-opt ranges |
| `benchmark_comparison.csv` | vs B1–B6 |
| `optimization_uncertainty.csv` | bootstrap |
| `confirmation_50k.csv` | stability |
| `optimization_report.md` | narrative RQ1–4 |

## Типичный численный факт (из принятых прогонов)

P2 3Y: argmin (1,0,0), CVaR≈1.187, VaR95=0, P(SF)≈4.9%, identification A; near-opt KZT roughly 0.92–1.00 (corrected counts in grid_patch).
