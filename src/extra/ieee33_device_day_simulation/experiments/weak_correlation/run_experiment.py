from __future__ import annotations

from pathlib import Path

from ..run_experiment import run_default


def main() -> None:
    run_default(
        mode="weak_correlation",
        results_root=Path("results/ieee33_device_day_simulation/weak_correlation"),
    )


if __name__ == "__main__":
    main()
