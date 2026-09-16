# 06 — Test suite

Всего **57** функций `test_*` в `tests/`. Исследовательский код тестами не меняется; тесты защищают инварианты.

## Phase 1 — `tests/test_phase1_utils.py`
| Test | Что проверяет |
|------|----------------|
| `test_log_return_two_prices` | `ln(S1/S0)` |
| `test_monthly_compounding_identity` | sum logs → monthly |
| `test_currency_column_identification` | NBK column heuristics |
| `test_duplicate_date_keep_last_logic` | merge policy |
| `test_parse_decimal_comma_and_dates` | locale parsing |

## Phase 2A
| File | Focus |
|------|-------|
| `test_iid_bootstrap.py` | joint rows together; no independent scramble; seed |
| `test_stationary_bootstrap.py` | p=1/L; wrap; short vs long blocks; Politis–White >0 |
| `test_path_builder.py` | level identity / positivity / no clipping |
| `test_student_t.py` | MLE df order; simulate finite |
| `test_simulation_reproducibility.py` | same seed → same output |

## Phase 2B1
| File | Focus |
|------|-------|
| `test_household_profiles.py` | shares sum 1; P1 dedup; P4 distinct |
| `test_cashflow_normalization.py` | I0/E0 KZT-eq; P2 USD nominal |
| `test_initial_portfolio.py` | holdings value to W0 |
| `test_monthly_settlement.py` | monthly shape; P1 surplus no SF |
| `test_rebalancing.py` | target weights restored |
| `test_accounting_conservation.py` | max err < 1e-8; no neg; no NaN |
| `test_shortfall.py` | structural deficit; CVaR largest-k |
| `test_fx_directionality.py` | stronger USD raises P2 income KZT-eq |

## Phase 2B2 — `test_optimization.py`
- grid sizes 1326 / 5151
- boundaries, no negatives
- empirical CVaR k=500 definition
- synthetic argmin
- local refine includes boundary
- near-optimal zero-CVaR
- paired bootstrap reproducibility
- benchmark dedup P1

## Phase 2C — `test_phase2c.py`
- integer bp uniqueness / merge
- zero-cost monthly ≡ default API
- quarterly / none accounting
- positive costs → positive fees
- recent window date filter
- SB56 block length attribute
- classification A vs B logic
- float 0.14 vs 0.14000000000000001 → same bp key

## Как запускать
```bash
PYTHONPATH=. python -m pytest tests/ -q
```
