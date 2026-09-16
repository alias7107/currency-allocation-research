#!/usr/bin/env python3
"""Generate publication-facing artifacts under results/final/."""

from __future__ import annotations

from src.final.pipeline import print_final_console, run_final_cleanup


def main() -> int:
    state = run_final_cleanup()
    print_final_console(state)
    return 1 if state.stopped else 0


if __name__ == "__main__":
    raise SystemExit(main())
