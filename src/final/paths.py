"""Output paths for publication freeze."""

from __future__ import annotations

from pathlib import Path

from src import config

FINAL_DIR = config.RESULTS / "final"
AUDIT_DIR = FINAL_DIR / "audit"
TABLES_DIR = FINAL_DIR / "tables"
FIGURES_DIR = FINAL_DIR / "figures"
