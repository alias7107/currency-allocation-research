# 00 — Архитектура проекта

См. также: `docs/COMPLETE_CODE_WALKTHROUGH_RU.md` (§1–3).

## Назначение репозитория

Эмпирико-симуляционное исследование аллокации ликвидных сбережений (KZT/USD/EUR) для домохозяйств с мультивалютными доходами/расходами. Риск-метрика — **CVaR95 кумулятивного shortfall**, не ожидаемое богатство.

## Канонические entry points

| Команда | Функция |
|---------|---------|
| `PYTHONPATH=. python scripts/run_phase1.py` | `src.pipeline.run_phase1` |
| `PYTHONPATH=. python scripts/run_phase2a.py` | `src.simulation.model_comparison.run_phase2a` |
| `PYTHONPATH=. python scripts/run_phase2b1.py` | `src.household.calibration.run_phase2b1` |
| `PYTHONPATH=. python scripts/run_phase2b2.py` | `src.optimization.pipeline.run_phase2b2` |
| `PYTHONPATH=. python scripts/run_phase2c.py` | `src.robustness.pipeline.run_phase2c` |

Notebooks (`notebooks/01_…09_…`) — вспомогательные; финальный pipeline = scripts.

## Поток данных (сжато)

```
NBK Excel (data/raw)
  → processed weekday returns (data/processed)
  → joint MC paths (simulation / household cache)
  → monthly FX (21-day blocks)
  → household shortfalls → CVaR(w)
  → grid argmin + robustness
  → results/*/
```

## Конфиг

Единственный центральный модуль: `src/config.py`.

Ключевые «принятые» параметры после ревью фаз:
- DGP: stationary bootstrap **L=20** (`HH_BLOCK_LENGTH`)
- N=10_000, seed=42
- Primary household: E/I=0.95, B=1
- CVaR α=0.95 → k=500

Замечание: `PRIMARY_BLOCK_LENGTH=10` в config — устаревший ярлык кандидата Phase 2A; household использует 20.

## Где что лежит

| Слой | Пакет |
|------|--------|
| Phase 1 | `src/*.py` top-level |
| Phase 2A | `src/simulation/` |
| Phase 2B1 | `src/household/` |
| Phase 2B2 | `src/optimization/` |
| Phase 2C | `src/robustness/` |
| Tests | `tests/` |
| Outputs | `results/` |

## Research questions

- RQ1–RQ4 → Phase 2B2 (+ профили/горизонты из 2B1)
- RQ5 → Phase 2C
