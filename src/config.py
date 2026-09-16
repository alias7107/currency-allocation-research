"""Central configuration for Phase 1 FX diagnostics.

Dates follow the National Bank of Kazakhstan (NBK) documentation where available.
The August 2015 float date is taken from the NBK Annual Report 2015
(transition to a freely floating tenge from 20 August 2015).
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
DIAGNOSTICS_DIR = RESULTS / "diagnostics"

# ---------------------------------------------------------------------------
# Sample windows
# ---------------------------------------------------------------------------
RAW_START = "2010-01-01"
RAW_END = "2025-12-31"

# Window A: full historical context
WINDOW_A_START = RAW_START
WINDOW_A_END = RAW_END

# Window B: baseline estimation under the floating regime
BASELINE_START = "2015-08-20"
BASELINE_END = RAW_END

# Window C: recent robustness sample
RECENT_START = "2020-01-01"
RECENT_END = RAW_END

# Official float transition (NBK Annual Report 2015; Joint Statement 20 Aug 2015)
FLOAT_TRANSITION_DATE = "2015-08-20"

# NBK methodology window: official vs market calculation differed
METHODOLOGY_WINDOW_START = "2016-01-06"
METHODOLOGY_WINDOW_END = "2016-02-01"

# ---------------------------------------------------------------------------
# Returns / rolling windows
# ---------------------------------------------------------------------------
ANNUALIZATION_FACTOR = 252  # weekday observations (see weekend diagnostics)
ROLLING_VOL_WINDOWS = (30, 90)
ROLLING_CORR_WINDOWS = (30, 90, 252)
ACF_MAX_LAG = 30
LJUNG_BOX_LAGS = (5, 10, 20, 30)
ARCH_LM_LAGS = (5, 10, 20)
EXTREME_JUMP_ABS_LOG_RETURN = 0.05  # flag only; do not delete
TOP_EXTREMES = 10

# Estimation calendar after weekend diagnostics
# "weekdays": drop Saturday/Sunday before computing returns used in estimation
ESTIMATION_CALENDAR = "weekdays"

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# NBK acquisition
# ---------------------------------------------------------------------------
NBK_RATES_PAGE = (
    "https://nationalbank.kz/en/exchangerates/"
    "ezhednevnye-oficialnye-rynochnye-kursy-valyut"
)
NBK_EXCEL_EXPORT = f"{NBK_RATES_PAGE}/excel"
NBK_RSS_RATES = "https://nationalbank.kz/rss/get_rates.cfm"
# Checkbox values on the official daily-rates page (verified 2026-08-24)
NBK_USD_RATE_ID = 5
NBK_EUR_RATE_ID = 6
REQUEST_TIMEOUT = 60
REQUEST_PAUSE_SECONDS = 0.4
USER_AGENT = (
    "currency-allocation-research/0.1 "
    "(academic Phase-1 diagnostics; National Bank of Kazakhstan official rates)"
)

TARGET_CURRENCIES = ("USD", "EUR")

# ---------------------------------------------------------------------------
# Phase 2A — joint scenario generation (does not alter Phase 1 outputs)
# ---------------------------------------------------------------------------
SIMULATION_DIR = RESULTS / "simulation"
SIM_FIGURES = SIMULATION_DIR / "figures"
SIM_TABLES = SIMULATION_DIR / "tables"
SIM_DIAGNOSTICS = SIMULATION_DIR / "diagnostics"

SIM_RANDOM_SEED = 42
SIM_EXAMPLE_PATH_SEED = 42

HORIZON_1Y_DAYS = 252
HORIZON_3Y_DAYS = 756
HORIZON_5Y_DAYS = 1260
MULTIHORIZON_WINDOWS = (5, 21, 63, 252)

N_PATHS_CONVERGENCE = (1_000, 5_000, 10_000, 50_000)
N_PATHS_PRIMARY = 10_000
N_PATHS_TEMPORAL = 400
N_PATHS_EXAMPLE = 8

STATIONARY_BLOCK_LENGTHS = (5, 10, 20)
PRIMARY_BLOCK_LENGTH = 10  # default candidate; validation may override recommendation

# Pseudo-out-of-sample split (distributional, not path forecasting)
OOS_TRAIN_START = BASELINE_START
OOS_TRAIN_END = "2019-12-31"
OOS_EVAL_START = RECENT_START
OOS_EVAL_END = BASELINE_END

# Narrow float-shock sensitivity window (NOT a revised baseline).
# Covers the documented 20 August 2015 float through the end of that calendar
# month, matching Phase 1 ruptures dates around 17–24 August 2015.
FLOAT_SHOCK_SENSITIVITY_START = "2015-08-20"
FLOAT_SHOCK_SENSITIVITY_END = "2015-08-31"

TAIL_QUANTILES = (0.005, 0.01, 0.025, 0.05, 0.95, 0.975, 0.99, 0.995)
COEXCEED_QUANTILES = (0.05, 0.95)
ACF_COMPARE_LAGS = (1, 5, 10, 20)

# Equal-weight category scores are disclosed in the comparison report.
COMPARISON_CATEGORIES = (
    "marginal",
    "tail",
    "dependence",
    "temporal",
    "vol_clustering",
    "multihorizon",
    "pseudo_oos",
)

# ---------------------------------------------------------------------------
# Phase 2B1 — household cash-flow engine (no allocation optimization)
# ---------------------------------------------------------------------------
HH_DIR = RESULTS / "household"
HH_FIGURES = HH_DIR / "figures"
HH_TABLES = HH_DIR / "tables"
HH_DIAGNOSTICS = HH_DIR / "diagnostics"

HH_RANDOM_SEED = SIM_RANDOM_SEED
HH_N_PATHS = N_PATHS_PRIMARY
HH_BLOCK_LENGTH = 20  # accepted Phase 2A primary DGP
HH_WEEKDAYS_PER_MONTH = 21
HH_I0 = 1.0
HH_EXPENSE_INCOME_RATIOS = (0.80, 0.95, 1.05)
HH_BUFFER_MONTHS = (1, 3, 6, 12)
HH_HORIZON_YEARS = (1, 3, 5)
HH_CVAR_ALPHA = 0.95
HH_CONVERSION_COST = 0.0
HH_REBALANCE_FREQUENCY = "monthly"  # Phase 2C may add quarterly / none
HH_ACCOUNTING_ATOL = 1e-8
HH_WEALTH_DEPLETE_TOL = 1e-10
HH_EXAMPLE_PATH_IDS = (0, 1, 2)

# ---------------------------------------------------------------------------
# Phase 2B2 — CVaR currency-allocation optimization (no Phase 2C robustness)
# ---------------------------------------------------------------------------
OPT_DIR = RESULTS / "optimization"
OPT_FIGURES = OPT_DIR / "figures"
OPT_TABLES = OPT_DIR / "tables"
OPT_DIAGNOSTICS = OPT_DIR / "diagnostics"

# Primary household environment (Phase 2B1 pre-specified-rule baseline)
OPT_EXPENSE_INCOME_RATIO = 0.95
OPT_BUFFER_MONTHS = 1
OPT_I0 = HH_I0
# W0 = OPT_BUFFER_MONTHS * OPT_EXPENSE_INCOME_RATIO * OPT_I0

OPT_N_PATHS = HH_N_PATHS
OPT_BLOCK_LENGTH = HH_BLOCK_LENGTH
OPT_RANDOM_SEED = HH_RANDOM_SEED
OPT_CVAR_ALPHA = HH_CVAR_ALPHA
OPT_HORIZON_YEARS = HH_HORIZON_YEARS

OPT_COARSE_STEP = 0.02
OPT_REFINED_STEP = 0.01
OPT_NEAR_OPTIMAL_REL_TOL_1PCT = 0.01
OPT_NEAR_OPTIMAL_REL_TOL_5PCT = 0.05
# When min CVaR ≈ 0, absolute near-optimal band (normalized KZT shortfall units)
OPT_NEAR_OPTIMAL_ABS_TOL = 1e-6
# Extra absolute band as fraction of annual expenses when CVaR≈0
OPT_NEAR_OPTIMAL_ABS_FRAC_ANNUAL = 0.001

OPT_REFINE_CVAR_BAND = 0.02  # coarse points within 2% of min for neighborhood
OPT_REFINE_NEIGHBORHOOD = 0.05  # ±0.05 in each weight around selected coarse points

# Identification thresholds (fraction of evaluated grid in 1% near-optimal set)
OPT_WELL_IDENTIFIED_MAX_FRAC = 0.05
OPT_NOT_IDENTIFIED_MIN_FRAC = 0.50
OPT_CVAR_ZERO_TOL = 1e-8

OPT_BOOTSTRAP_REPLICATES = 1000
OPT_BOOTSTRAP_SEED = 4242
OPT_CONFIRM_N_PATHS = 50_000
OPT_CONFIRM_SEED = 424242
OPT_CONFIRM_PROFILES = ("P2_foreign_income_mismatch", "P4_full_multi_currency")
OPT_CONFIRM_HORIZON_YEARS = 3

# Phase 2B1 benchmark reproduction tolerance (relative on CVaR when CVaR>0)
OPT_BENCHMARK_REPRO_ATOL = 1e-6
OPT_BENCHMARK_REPRO_RTOL = 1e-4

# ---------------------------------------------------------------------------
# Phase 2C — grid integrity patch + robustness / sensitivity
# ---------------------------------------------------------------------------
ROB_DIR = RESULTS / "robustness"
ROB_FIGURES = ROB_DIR / "figures"
ROB_TABLES = ROB_DIR / "tables"
ROB_DIAGNOSTICS = ROB_DIR / "diagnostics"
ROB_GRID_PATCH = ROB_DIR / "grid_patch"

ROB_N_PATHS = N_PATHS_PRIMARY
ROB_RANDOM_SEED = 42
ROB_HORIZON_YEARS = 3
ROB_COARSE_STEP_BP = 200  # 2%
ROB_REFINED_STEP_BP = 100  # 1%
ROB_REFINE_CVAR_BAND = OPT_REFINE_CVAR_BAND
ROB_REFINE_NEIGHBORHOOD_BP = 500  # ±5%

# Baseline reference (Phase 2B2 primary)
ROB_BASELINE_EI = OPT_EXPENSE_INCOME_RATIO
ROB_BASELINE_B = OPT_BUFFER_MONTHS
ROB_BASELINE_DGP = "SB20"
ROB_BASELINE_BLOCK = 20
ROB_BASELINE_WINDOW_START = BASELINE_START
ROB_BASELINE_WINDOW_END = BASELINE_END
ROB_BASELINE_REBALANCE = "monthly"
ROB_BASELINE_COST = 0.0

# Expected baseline CVaR at 10k seed 42 (gate)
ROB_GATE_P2_CVAR = 1.1871802686857733
ROB_GATE_P4_CVAR = 0.5082906243903793
ROB_GATE_CVAR_RTOL = 1e-4
ROB_GATE_CVAR_ATOL = 1e-6
ROB_GATE_WEIGHT_ATOL = 0.011  # 1 percentage point

# R1 alternative household
ROB_ALT_EI = 1.05
ROB_ALT_B = 3

# R2 DGP seeds (documented; within-DGP CRN for portfolios)
ROB_DGP_SPECS = (
    {"name": "SB20", "kind": "stationary_bootstrap", "block": 20, "seed": 42},
    {"name": "SB10", "kind": "stationary_bootstrap", "block": 10, "seed": 4210},
    {"name": "SB56", "kind": "stationary_bootstrap", "block": 56, "seed": 4256},
    {"name": "IID", "kind": "iid_bootstrap", "block": None, "seed": 4240},
)

# R3 recent window
ROB_RECENT_START = RECENT_START
ROB_RECENT_END = RECENT_END
ROB_RECENT_SEED = 4220

# R4 rebalancing
ROB_REBALANCE_FREQS = ("monthly", "quarterly", "none")

# R5 conversion costs (scenario-based; NOT empirical KZ spreads)
ROB_CONVERSION_COSTS = (0.0, 0.0025, 0.005, 0.01)
ROB_COST_REBALANCE_CROSS = 0.005  # informative positive cost for rebalance cross

ROB_CORE_PROFILES = (
    "P2_foreign_income_mismatch",
    "P4_full_multi_currency",
)
ROB_SECONDARY_PROFILES = ("P3_mixed_income",)

ROB_CONFIRM_N_PATHS = 50_000
ROB_CONFIRM_SEED = 424242

# Exact-allocation robustness: within 5pp per currency
ROB_EXACT_WEIGHT_TOL = 0.05
# Mechanism: expense-matched still beats income-matched
ROB_MECHANISM_CVAR_EDGE = 1e-8
