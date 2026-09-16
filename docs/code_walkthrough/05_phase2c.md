# 05 — Phase 2C: grid patch + robustness

См. мастер §9,31–37; код: `src/robustness/`.

## Entry

`scripts/run_phase2c.py` → `run_phase2c()`  
**Файл:** `src/robustness/pipeline.py`

## Порядок (обязателен)

### PART A — `run_grid_patch` (`grid_patch.py`)
- Пересчёт всех Phase 2B2 cells с integer bp
- Сравнение с `results/optimization/tables/optimal_allocations.csv`
- Пишет в `results/robustness/grid_patch/` (оригинал 2B2 **не** затирается)
- Если optima/CVaR меняются сверх tolerance → `grid_integrity_failure_report.md` и **STOP**

Фактический исход прогона: optima/CVaR неизменны; counts 5280→5151 и т.п. исправлены.

### Gate
`verify_baseline_gate`: P2/P4 3Y ≈ 1.187 / 0.508, w=(1,0,0).

### PART B — `run_robustness_suite`
| Stage | Spec | Core profiles |
|-------|------|---------------|
| R0 | baseline | P2,P3,P4 |
| R1 | E/I=1.05, B=3 (+P1 control) | |
| R2 | SB10, SB56, IID (+SB20 copy) | P2,P4 |
| R3 | 2020–2025 SB20 | P2,P4 |
| R4 | quarterly / none | P2,P4 |
| R5 | costs 0.25/0.5/1% + 0.5%×rebalance | P2,P4 |

Оптимизация клетки: `robustness.optimize.optimize_cell` (coarse 2% + refine 1% bp).

## Rebalance modes (`settlement._should_rebalance`)
- monthly: каждый месяц
- quarterly: `(m+1) % 3 == 0`
- none: никогда; surplus остаётся в валюте дохода

## Costs
- liquidation fee `c * take` with gross-up
- rebalance fee `c * buy_volume`
- documented as scenario parameters, not KZ market spreads

## Classification
`classify_robustness` → ROBUST_EXACT / MECHANISM / WEAKENED / REVERSED / UNIDENTIFIED  
`final_paper_classification` → A/B/C/D  

Фактический итог: **RESULT B** (есть UNIDENTIFIED на no-rebalance при точном argmin всё ещё 100% KZT).

## Выходы
`robustness_results.csv`, `expense_vs_income_matching.csv`, `dgp/window/rebalancing/transaction_cost_sensitivity.csv`, `robustness_report.md`, figures R1–R6.
