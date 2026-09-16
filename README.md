# Currency Allocation Research — Phase 1

**Title:** Optimizing Currency Allocation for Individuals with Multi-Currency Income and Expenses: A Monte Carlo Simulation Approach Using Historical Exchange Rate Data.

This repository currently implements **Phase 1 only**: official FX data acquisition, cleaning, exploratory analysis, and statistical diagnostics for **USD/KZT** and **EUR/KZT** with **KZT** as the base currency. It does **not** implement portfolio optimisation or the Monte Carlo allocator (Phase 2).

## Dataset source

Primary source: **National Bank of Kazakhstan (NBK)**  
Product: **Daily Official (Market) Foreign Exchange Rates**  
Interface used: official Excel export

https://nationalbank.kz/en/exchangerates/ezhednevnye-oficialnye-rynochnye-kursy-valyut

USD/KZT is the NBK official (KASE-based) dollar rate. EUR/KZT is an NBK-published **cross rate**. Both are stored as **KZT per one unit** of foreign currency.

Documented regime date in `src/config.py`: **20 August 2015** (NBK *Annual Report 2015*: transition to a freely floating tenge).  
Methodology window: **6 January 2016 – 1 February 2016** (NBK note on official vs market calculation).

### Suggested citation

National Bank of Kazakhstan. Daily Official (Market) Foreign Exchange Rates. Retrieved from https://nationalbank.kz/

National Bank of the Republic of Kazakhstan (2016). *Report of the National Bank of the Republic of Kazakhstan for 2015*. (Float from 20 August 2015.)

## Project structure

```text
currency-allocation-research/
  data/raw/          original NBK files (never overwritten)
  data/interim/      merged calendar series and rolling diagnostics
  data/processed/    estimation-ready daily and monthly files
  src/               reusable Python modules
  notebooks/         thin notebooks that call src/
  results/figures/   PNG and PDF figures
  results/tables/    CSV (and LaTeX) tables
  results/diagnostics/
  tests/
  scripts/run_phase1.py
```

## Installation

```bash
cd currency-allocation-research
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## How to run acquisition

```bash
PYTHONPATH=. python -c "from src.data_acquisition import acquire_nbk_yearly_excel, build_raw_file_manifest; acquire_nbk_yearly_excel(); build_raw_file_manifest()"
```

Yearly workbooks are stored as `data/raw/nbk_official_rates_YYYY.xlsx`. Existing raw files are **not** overwritten.

If a year fails to download, export that year from the NBK page (From/To + USD and EUR checked + Export to XLSX) and save it under the same filename. The merger reads every `.xlsx`/`.xls`/`.csv` in `data/raw/`.

## How to run preprocessing and all diagnostics

```bash
PYTHONPATH=. python scripts/run_phase1.py
```

## Phase 2A — joint simulation (no household model)

```bash
PYTHONPATH=. python scripts/run_phase2a.py
```

Outputs are written only under `results/simulation/`. Phase 1 tables/figures are not overwritten.
Primary estimation sample remains weekday official rates, 2015-08-20 to 2025-12-31.

## Phase 2B1 — household engine (no allocation optimization)

```bash
PYTHONPATH=. python scripts/run_phase2b1.py
```

Uses the accepted joint stationary bootstrap **L=20**, 10,000 paths, seed 42.
Outputs only under `results/household/`.

## Phase 2B2 — CVaR currency allocation optimization

```bash
PYTHONPATH=. python scripts/run_phase2b2.py
```

Primary environment from Phase 2B1 rule: E/I=0.95, B=1. Uses joint SB20, 10,000 paths, seed 42.
Outputs only under `results/optimization/`. Does not overwrite Phase 1 / 2A / 2B1 results.



This downloads missing years (if needed), validates, writes processed datasets, tables, figures, and the two markdown reports.

## Reproduce tables

After `run_phase1.py`, tables are in `results/tables/`:

- `descriptive_statistics.csv` / `.tex`
- `distribution_tests.csv`
- `autocorrelation_tests.csv`
- `arch_tests.csv`
- `garch11_diagnostics.csv` (diagnostic only)
- `structural_break_tests.csv`
- `methodology_window_sensitivity.csv`
- `extreme_returns.csv`
- `return_correlations.csv`

## Reproduce figures

Figures are written to `results/figures/` as PNG and PDF (`fig01_…` through ACF plots). Re-run `src.visualization.save_all_figures` via the pipeline.

## Notebooks

```bash
jupyter notebook notebooks/01_data_acquisition.ipynb
```

Each notebook imports `src` rather than duplicating logic.

## Tests

```bash
PYTHONPATH=. pytest tests -q
```

## Known data limitations

- NBK Excel exports can include **weekend rows that repeat** the previous official rate. Estimation uses **weekdays only** (`ESTIMATION_CALENDAR` in `config.py`).
- The RSS date service did not return 2010–2021 observations in the development environment; yearly Excel export was used instead.
- Open Data (`data.egov.kz`) requires an API key and is not the primary source.
- Extreme moves (e.g. August 2015) are flagged, not deleted.
- EUR/KZT correlation with USD/KZT is partly mechanical because EUR/KZT is a cross rate.

Configuration (windows, float date, rolling lengths, annualisation factor 252) lives in `src/config.py`.
