"""Replot one result CSV for the E20-E24 traceability tree."""
from __future__ import annotations

import argparse
from pathlib import Path

from tools.replot_results_e20_e24 import _replot_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if _replot_csv(args.input, args.output) is None:
        raise SystemExit(f"No numeric columns available in {args.input}")


if __name__ == "__main__":
    main()
