#!/usr/bin/env python3
"""Build chatgpt_results_package/ by COPYING existing research artifacts only."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "chatgpt_results_package"
ZIP_PATH = PROJECT_ROOT / "chatgpt_results_package.zip"

EXCLUDE_DIRS = {".venv", ".git", "__pycache__", ".pytest_cache", "node_modules"}
FIGURE_EXTS = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}

# (source_glob_or_path, dest_subdir, optional_note)
COPY_RULES: list[tuple[str, str]] = []


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_copy(src: Path, dest_dir: Path, registry: dict[str, Path], preserve_name: str | None = None) -> Path | None:
    """Copy src into dest_dir; prefix on filename collision."""
    if not src.is_file():
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_name = preserve_name or src.name
    dest_path = dest_dir / dest_name
    if dest_path.exists() or str(dest_path.relative_to(PACKAGE_ROOT)) in registry:
        rel = src.relative_to(PROJECT_ROOT)
        prefix = str(rel.parent).replace("/", "_").replace("\\", "_")
        if prefix == ".":
            prefix = "root"
        stem, suffix = Path(dest_name).stem, Path(dest_name).suffix
        dest_name = f"{prefix}__{stem}{suffix}"
        dest_path = dest_dir / dest_name
        n = 2
        while dest_path.exists():
            dest_name = f"{prefix}__{stem}_{n}{suffix}"
            dest_path = dest_dir / dest_name
            n += 1
    shutil.copy2(src, dest_path)
    registry[str(dest_path.relative_to(PACKAGE_ROOT))] = src
    return dest_path


def copy_tree_files(src_dir: Path, dest_dir: Path, registry: dict[str, Path]) -> int:
    """Copy all files under src_dir into dest_dir preserving relative paths."""
    if not src_dir.exists():
        return 0
    count = 0
    for p in sorted(src_dir.rglob("*")):
        if not p.is_file():
            continue
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        rel = p.relative_to(src_dir)
        target_dir = dest_dir / rel.parent
        if safe_copy(p, target_dir, registry, preserve_name=rel.name):
            count += 1
    return count


def copy_if_exists(src_rel: str, dest_subdir: str, registry: dict[str, Path]) -> bool:
    src = PROJECT_ROOT / src_rel
    if src.is_file():
        safe_copy(src, PACKAGE_ROOT / dest_subdir, registry)
        return True
    return False


def collect_figures(registry: dict[str, Path]) -> int:
    count = 0
    search_roots = [
        PROJECT_ROOT / "results",
        PROJECT_ROOT / "figures",
    ]
    for root in search_roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix.lower() not in FIGURE_EXTS:
                continue
            if any(part in EXCLUDE_DIRS for part in p.parts):
                continue
            rel = p.relative_to(PROJECT_ROOT)
            target_dir = PACKAGE_ROOT / "08_figures_existing" / rel.parent
            if safe_copy(p, target_dir, registry, preserve_name=rel.name):
                count += 1
    return count


def main() -> int:
    if PACKAGE_ROOT.exists():
        print(f"ERROR: {PACKAGE_ROOT} already exists. Remove manually to rebuild.")
        return 1

    dirs = [
        "01_canonical_final",
        "02_optimal_allocations",
        "03_benchmark_results",
        "04_optimization_surfaces",
        "05_scenario_level_losses",
        "06_robustness",
        "07_simulation_validation",
        "08_figures_existing",
        "09_household_calibration",
        "10_configs_and_assumptions",
        "11_summaries",
        "12_reproducibility_metadata",
    ]
    for d in dirs:
        (PACKAGE_ROOT / d).mkdir(parents=True, exist_ok=True)

    registry: dict[str, Path] = {}  # package_rel -> original Path

    # --- 01 canonical final ---
    copy_tree_files(PROJECT_ROOT / "results" / "final", PACKAGE_ROOT / "01_canonical_final", registry)

    # --- 02 optimal allocations ---
    opt_paths = [
        "results/optimization/tables/optimal_allocations.csv",
        "results/optimization/tables/horizon_allocation_comparison.csv",
        "results/optimization/tables/optimization_uncertainty.csv",
        "results/robustness/grid_patch/corrected_optimal_allocations.csv",
        "results/robustness/grid_patch/corrected_near_optimal_regions.csv",
        "results/robustness/grid_patch/grid_patch_comparison.csv",
        "results/final/tables/table_optimal_allocations.csv",
        "results/final/audit/grid_reconciliation.csv",
        "results/final/audit/grid_integrity.csv",
        "results/optimization/diagnostics/optimization_report.md",
        "results/optimization/diagnostics/phase2b2_run_summary.json",
    ]
    for rel in opt_paths:
        copy_if_exists(rel, "02_optimal_allocations", registry)

    # --- 03 benchmark results ---
    bench_paths = [
        "results/final/tables/table_benchmark_3y.csv",
        "results/optimization/tables/benchmark_comparison.csv",
        "results/optimization/tables/confirmation_50k.csv",
        "results/robustness/tables/confirmation_50k_robustness.csv",
    ]
    for rel in bench_paths:
        copy_if_exists(rel, "03_benchmark_results", registry)

    # --- 04 optimization surfaces ---
    surface_paths = [
        "results/optimization/tables/coarse_grid_results.csv",
        "results/optimization/tables/refined_grid_results.csv",
        "results/optimization/tables/global_1pct_check_P2_P4_3Y.csv",
    ]
    for rel in surface_paths:
        copy_if_exists(rel, "04_optimization_surfaces", registry)

    # --- 05 scenario-level losses (search; copy only if found) ---
    scenario_loss_found = False
    loss_patterns = ["*loss*", "*scenario*", "*shortfall*"]
    loss_exts = {".csv", ".json", ".npy", ".npz", ".parquet", ".pkl", ".pickle"}
    for root in [PROJECT_ROOT / "results", PROJECT_ROOT / "outputs", PROJECT_ROOT / "cache"]:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in loss_exts:
                continue
            name_lower = p.name.lower()
            if "daily_levels" in name_lower or "fx_" in name_lower:
                continue
            if any(k in name_lower for k in ("scenario", "loss", "shortfall_per")):
                if "stress_calibration" in name_lower or "benchmark" in name_lower:
                    continue
                safe_copy(p, PACKAGE_ROOT / "05_scenario_level_losses", registry)
                scenario_loss_found = True

    # --- 06 robustness (exclude large FX path caches) ---
    rob_src = PROJECT_ROOT / "results" / "robustness"
    if rob_src.exists():
        for p in sorted(rob_src.rglob("*")):
            if not p.is_file():
                continue
            if "cache" in p.parts and p.suffix == ".npy":
                continue
            if any(part in EXCLUDE_DIRS for part in p.parts):
                continue
            rel = p.relative_to(rob_src)
            target_dir = PACKAGE_ROOT / "06_robustness" / rel.parent
            safe_copy(p, target_dir, registry, preserve_name=rel.name)

    # --- 07 simulation validation ---
    for sub in ["simulation", "tables", "diagnostics"]:
        copy_tree_files(PROJECT_ROOT / "results" / sub, PACKAGE_ROOT / "07_simulation_validation" / sub, registry)

    # --- 08 figures ---
    collect_figures(registry)

    # --- 09 household calibration (exclude FX path cache) ---
    hh_src = PROJECT_ROOT / "results" / "household"
    if hh_src.exists():
        for p in sorted(hh_src.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix == ".npy":
                continue
            rel = p.relative_to(hh_src)
            target_dir = PACKAGE_ROOT / "09_household_calibration" / rel.parent
            safe_copy(p, target_dir, registry, preserve_name=rel.name)
    copy_if_exists(
        "results/optimization/diagnostics/phase2b1_reproduction_check.csv",
        "09_household_calibration",
        registry,
    )

    # --- 10 configs ---
    config_paths = [
        "results/final/canonical_specification.json",
        "results/final/canonical_specification.md",
        "results/final/ASSUMPTIONS_AND_LIMITATIONS.md",
        "src/config.py",
        "requirements.txt",
        "results/simulation/diagnostics/politis_white_block_length.json",
        "results/simulation/diagnostics/phase2a_run_summary.json",
        "results/household/diagnostics/phase2b1_run_summary.json",
        "results/optimization/diagnostics/phase2b2_run_summary.json",
        "results/robustness/diagnostics/phase2c_run_summary.json",
        "results/robustness/diagnostics/recent_window_block_diagnostic.json",
    ]
    for rel in config_paths:
        copy_if_exists(rel, "10_configs_and_assumptions", registry)

    for script in sorted((PROJECT_ROOT / "scripts").glob("run_*.py")):
        safe_copy(script, PACKAGE_ROOT / "10_configs_and_assumptions", registry)

    # --- 11 summaries ---
    summary_paths = [
        "README.md",
        "results/final/FINAL_RESULTS_SUMMARY.md",
        "results/final/PAPER_SAFE_CLAIMS.md",
        "results/final/PUBLICATION_ARTIFACT_MANIFEST.md",
        "results/final/figure_captions.md",
        "results/final/ASSUMPTIONS_AND_LIMITATIONS.md",
        "results/diagnostics/simulation_method_recommendation.md",
        "results/diagnostics/data_quality_report.md",
        "results/household/diagnostics/calibration_report.md",
        "results/simulation/diagnostics/model_comparison_report.md",
        "results/optimization/diagnostics/optimization_report.md",
        "results/robustness/diagnostics/robustness_report.md",
        "results/robustness/diagnostics/grid_patch_report.md",
    ]
    for rel in summary_paths:
        copy_if_exists(rel, "11_summaries", registry)

    copy_tree_files(PROJECT_ROOT / "docs", PACKAGE_ROOT / "11_summaries", registry)

    # --- 12 reproducibility metadata ---
    meta_dir = PACKAGE_ROOT / "12_reproducibility_metadata"
    inventory_path = meta_dir / "file_inventory.csv"
    snapshot_path = meta_dir / "project_snapshot.txt"

    rows = []
    for pkg_rel, orig in sorted(registry.items()):
        pkg_path = PACKAGE_ROOT / pkg_rel
        stat = pkg_path.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        rows.append(
            {
                "package_path": pkg_rel,
                "original_path": str(orig.relative_to(PROJECT_ROOT)),
                "filename": pkg_path.name,
                "extension": pkg_path.suffix.lower(),
                "file_size_bytes": stat.st_size,
                "last_modified": mtime,
                "sha256": sha256_file(pkg_path),
            }
        )

    with inventory_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "package_path",
                "original_path",
                "filename",
                "extension",
                "file_size_bytes",
                "last_modified",
                "sha256",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # project snapshot
    git_commit = "N/A"
    git_branch = "N/A"
    git_status = "N/A"
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
        git_status = subprocess.check_output(
            ["git", "status", "--short"], cwd=PROJECT_ROOT, text=True
        ).strip() or "(clean)"
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    py_version = sys.version.replace("\n", " ")
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    snapshot = f"""Project snapshot
================
Package created: {created}
Project root: {PROJECT_ROOT}
Git commit: {git_commit}
Git branch: {git_branch}
Git status --short:
{git_status}

Python version: {py_version}
Operating system: {platform.system()} {platform.release()} ({platform.machine()})
"""
    snapshot_path.write_text(snapshot, encoding="utf-8")

    # --- MANIFEST.md ---
    total_bytes = sum(r["file_size_bytes"] for r in rows)
    n_files = len(rows)

    coarse = PROJECT_ROOT / "results/optimization/tables/coarse_grid_results.csv"
    refined = PROJECT_ROOT / "results/optimization/tables/refined_grid_results.csv"
    global_1pct = PROJECT_ROOT / "results/optimization/tables/global_1pct_check_P2_P4_3Y.csv"

    surfaces_found = coarse.exists() and refined.exists()
    robustness_raw = (PROJECT_ROOT / "results/robustness/tables/robustness_results.csv").exists()
    canonical_found = all(
        (PROJECT_ROOT / f"results/final/tables/{t}").exists()
        for t in [
            "table_fx_descriptive_statistics.csv",
            "table_simulation_model_comparison.csv",
            "table_optimal_allocations.csv",
            "table_benchmark_3y.csv",
            "table_robustness_summary.csv",
        ]
    )

    manifest = f"""# ChatGPT Results Package — Manifest

## 1. Purpose

This package collects **existing, unmodified** research outputs from the currency-allocation
Monte Carlo project so an independent reviewer (or LLM-assisted analysis) can reconstruct
Results tables, figures, robustness discussion, and verify reported values **without rerunning
simulations**.

All files were **copied** from the repository; originals were not modified, renamed, moved, or deleted.

**Package created:** {created}  
**Files copied:** {n_files}  
**Total size (excluding this manifest):** {total_bytes / (1024 * 1024):.2f} MiB

---

## 2. Directory structure

| Folder | Contents |
|--------|----------|
| `01_canonical_final/` | Frozen publication layer (`results/final/`) |
| `02_optimal_allocations/` | Optimal weights, near-optimal regions, grid patch, uncertainty |
| `03_benchmark_results/` | Benchmark comparisons (3Y canonical + full Phase 2B2 tables) |
| `04_optimization_surfaces/` | Raw grid-level CVaR surfaces (coarse 2%, refined 1%, 1% argmin check) |
| `05_scenario_level_losses/` | Per-scenario household loss arrays (**see §8**) |
| `06_robustness/` | Phase 2C robustness tables, diagnostics, figures (excludes FX `.npy` caches) |
| `07_simulation_validation/` | Phase 1 diagnostics + Phase 2A model validation |
| `08_figures_existing/` | All research figures (PNG/PDF/SVG) |
| `09_household_calibration/` | Phase 2B1 stress calibration (E/I × B × horizon) |
| `10_configs_and_assumptions/` | Canonical spec, `config.py`, run summaries, entry scripts |
| `11_summaries/` | README, phase reports, documentation walkthrough |
| `12_reproducibility_metadata/` | `file_inventory.csv` (SHA-256) + `project_snapshot.txt` |

---

## 3. CANONICAL publication results (cite these)

Located primarily in `01_canonical_final/`:

| Artifact | Path |
|----------|------|
| FX descriptives | `01_canonical_final/tables/table_fx_descriptive_statistics.csv` |
| Simulation model comparison | `01_canonical_final/tables/table_simulation_model_comparison.csv` |
| Optimal allocations | `01_canonical_final/tables/table_optimal_allocations.csv` |
| 3Y benchmarks | `01_canonical_final/tables/table_benchmark_3y.csv` |
| Robustness summary | `01_canonical_final/tables/table_robustness_summary.csv` |
| Frozen specification | `01_canonical_final/canonical_specification.json` |
| Safe claims | `01_canonical_final/PAPER_SAFE_CLAIMS.md` |
| Final summary | `01_canonical_final/FINAL_RESULTS_SUMMARY.md` |
| Assumptions | `01_canonical_final/ASSUMPTIONS_AND_LIMITATIONS.md` |
| Publication figures | `01_canonical_final/figures/figure01–07_*.pdf` |
| Audits | `01_canonical_final/audit/*.md`, `*.csv` |

---

## 4. Raw / intermediate outputs

| Type | Location | Notes |
|------|----------|-------|
| Coarse 2% grid (all cells) | `04_optimization_surfaces/coarse_grid_results.csv` | ~1326 pts/cell |
| Refined 1% local neighborhoods | `04_optimization_surfaces/refined_grid_results.csv` | P2 3Y: 199 rows; P4 3Y: 105 rows |
| Full 1% simplex argmin only | `04_optimization_surfaces/global_1pct_check_P2_P4_3Y.csv` | Confirms optimum, not full surface |
| Legacy optima (pre-patch) | `02_optimal_allocations/results_optimization_tables__optimal_allocations.csv` | **Do not cite** — inflated `n_surface=5280` |
| Corrected patch source | `02_optimal_allocations/corrected_optimal_allocations.csv` | Integer-bp grid patch |
| Full robustness experiments | `06_robustness/robustness_results.csv` | Raw R1–R5 runs |
| Phase 1 raw tables | `07_simulation_validation/` | Descriptives, ACF, ARCH, breaks |
| FX path caches (NOT copied) | `results/*/cache/*_daily_levels.npy` | ~2.1 GB; FX levels only, not losses |

---

## 5. Which files to use for each paper section

| Paper section | Primary files |
|---------------|---------------|
| **Results — FX data** | `01_canonical_final/tables/table_fx_descriptive_statistics.csv`, `08_figures_existing/figure01_*` |
| **Results — model selection** | `01_canonical_final/tables/table_simulation_model_comparison.csv`, `07_simulation_validation/model_comparison_report.md` |
| **Benchmark table (3Y)** | `01_canonical_final/tables/table_benchmark_3y.csv` (canonical); detail in `03_benchmark_results/benchmark_comparison.csv` |
| **Optimal allocations table** | `01_canonical_final/tables/table_optimal_allocations.csv` |
| **Optimization / simplex figure** | `08_figures_existing/figure04_p2_3y_cvar_surface.pdf`, `figure05_p4_3y_cvar_surface.pdf`; grid data in `04_optimization_surfaces/coarse_grid_results.csv` (filter P2/P4, 3Y) |
| **VaR / CVaR / loss distribution figure** | Benchmark + uncertainty: `03_benchmark_results/`, `02_optimal_allocations/optimization_uncertainty.csv`; **no per-scenario loss vectors** (§8) |
| **Robustness section** | `01_canonical_final/tables/table_robustness_summary.csv`, `06_robustness/robustness_results.csv`, robustness figures in `08_figures_existing/figR*` |
| **Simulation validation** | `07_simulation_validation/` (marginal, tail, temporal, dependence validation tables) |
| **Methods framework figure** | `08_figures_existing/research_framework/research_framework.pdf` or `.svg` |
| **Household baseline selection** | `09_household_calibration/stress_calibration.csv`, `01_canonical_final/audit/buffer_selection_audit.md` |

---

## 6. Missing requested artifacts

| Requested item | Status |
|----------------|--------|
| **Full 1% simplex surface CSV (5151 rows) for P2/P4 3Y** | **NOT PERSISTED.** Only coarse 2% grid + local 1% refinement + argmin confirmation exist. |
| **Scenario-level household loss arrays (P2 3Y)** | **NOT PERSISTED.** See §8. |
| **ASSUMPTIONS_AND_LIMITATIONS.md at repo root** | **Missing at root**; present in `01_canonical_final/ASSUMPTIONS_AND_LIMITATIONS.md`. |
| **YAML/TOML config files** | **None** — parameters in `src/config.py` and `canonical_specification.json`. |
| **NPZ / PKL / Parquet research outputs** | **None found.** |

---

## 7. Duplicates / legacy — DO NOT USE for paper

- `02_optimal_allocations/optimal_allocations.csv` — legacy pre-grid-patch counts
- `02_optimal_allocations/horizon_allocation_comparison.csv` — duplicate of legacy optima
- Pre-patch near-optimal counts with `n_surface=5280` (see `grid_patch_report.md` in `11_summaries/`)

**Use instead:** `01_canonical_final/tables/table_optimal_allocations.csv` and `02_optimal_allocations/corrected_*` files.

---

## 8. Scenario-level household losses

**Scenario-level household loss outputs were not persisted.**

The pipeline computes per-scenario cumulative shortfall vectors in memory during grid search
and CVaR evaluation, but only **aggregated** metrics (CVaR95, VaR95, P(shortfall)) were written
to CSV. No files with `scenario_id` + per-scenario `loss` columns exist.

FX Monte Carlo **path caches** (`*_daily_levels.npy`, ~192 MB–962 MB each) store simulated
exchange-rate levels, not household losses. These were **excluded** from this package due to
size (~2.1 GB total) and because they are not loss arrays.

---

## 9. Full allocation-grid / surface file locations

| File | Package path | Coverage |
|------|--------------|----------|
| Coarse 2% simplex grid | `04_optimization_surfaces/coarse_grid_results.csv` | All profiles P1–P4; horizons 1Y, 3Y, 5Y |
| Refined 1% neighborhoods | `04_optimization_surfaces/refined_grid_results.csv` | Local refinement around coarse optima |
| Global 1% argmin check | `04_optimization_surfaces/global_1pct_check_P2_P4_3Y.csv` | P2 & P4, 3Y baseline only (optimum row) |

To rebuild ternary plots: filter `coarse_grid_results.csv` by `profile` and `horizon_years`;
columns include `w_kzt`, `w_usd`, `w_eur`, `cvar95`, `var95`, `shortfall_probability`.

---

## 10. Final baseline specification

From `01_canonical_final/canonical_specification.json`:

| Parameter | Value |
|-----------|-------|
| FX DGP | Joint stationary bootstrap |
| Expected block length | **L = 20** (SB20) |
| Monte Carlo scenarios | **10,000** (seed 42) |
| Estimation window | 2015-08-20 to 2025-12-31 (post-float) |
| Household E/I ratio | **0.95** |
| Buffer months | **B = 1** |
| Currencies | KZT / USD / EUR |
| Rebalancing | Monthly (baseline) |
| Transaction costs | 0% (baseline) |
| Primary risk measure | CVaR95 |
| Optimization grid | Integer basis points; coarse 2% then local 1% refinement |

---

## 11. Legacy vs final config conflicts

| Issue | Resolution |
|-------|------------|
| Grid patch duplicate counts (`n_surface=5280` vs 5151) | **Canonical:** integer-bp patch in `corrected_optimal_allocations.csv`; economic optima unchanged |
| `OPT_BUFFER_MONTHS=1` in Phase 2B2 vs `select_baseline_candidate()` preferring B≈6 | Documented in `buffer_selection_audit.md`; **baseline for paper: B=1** per canonical spec |
| Legacy `results/optimization/tables/optimal_allocations.csv` | Superseded by `01_canonical_final/tables/table_optimal_allocations.csv` |

---

## 12. Integrity verification

See `12_reproducibility_metadata/file_inventory.csv` for SHA-256 hashes of every copied file
and `project_snapshot.txt` for git commit / environment at package creation time.
"""
    (PACKAGE_ROOT / "MANIFEST.md").write_text(manifest, encoding="utf-8")

    # --- ZIP ---
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    shutil.make_archive(str(ZIP_PATH.with_suffix("")), "zip", root_dir=PROJECT_ROOT, base_dir="chatgpt_results_package")

    zip_size = ZIP_PATH.stat().st_size if ZIP_PATH.exists() else 0

    # --- Terminal report ---
    print("=" * 60)
    print("CHATGPT RESULTS PACKAGE — BUILD REPORT")
    print("=" * 60)
    print(f"Package path:     {PACKAGE_ROOT}")
    print(f"Zip path:         {ZIP_PATH}")
    print(f"Files copied:     {n_files}")
    print(f"Uncompressed:     {total_bytes / (1024 * 1024):.2f} MiB")
    print(f"Zip size:         {zip_size / (1024 * 1024):.2f} MiB")
    print(f"Full opt surfaces: {'YES (coarse 2% + refined 1%; NO full 5151-row 1% CSV)' if surfaces_found else 'NO'}")
    print(f"Scenario losses:  {'YES' if scenario_loss_found else 'NO — not persisted'}")
    print(f"Robustness raw:   {'YES' if robustness_raw else 'NO'}")
    print(f"Canonical tables: {'YES' if canonical_found else 'PARTIAL/MISSING'}")
    print()
    print("Missing / partial:")
    print("  - Full 1% simplex surface CSV (5151 rows) for P2/P4 3Y")
    print("  - Scenario-level household loss arrays")
    print("  - FX path .npy caches (~2.1 GB, excluded by design)")
    print("  - ASSUMPTIONS_AND_LIMITATIONS.md at repo root (in 01_canonical_final/)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
