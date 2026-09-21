from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np

from experiments import run_experiment as exp
from src.extra.paper_figures import generate_paper_figures


logger = logging.getLogger(__name__)


def _set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        logger.warning("Torch seed setup skipped", exc_info=True)


def run_no_profile_perturbation_experiments(
    *,
    result1: bool = False,
    result2: bool = False,
    result3: bool = False,
    result4: bool = False,
    all_results: bool = False,
    supplementary: bool = False,
    n_devices: int = 5000,
    n_runs: int = 1,
    make_figures: bool = False,
    figure_output_dir: str | Path | None = None,
) -> None:

    run_r1 = result1 or all_results
    run_r2 = result2 or all_results
    run_r3 = result3 or all_results
    run_r4 = result4 or all_results

    if not any([run_r1, run_r2, run_r3, run_r4, supplementary]):
        raise ValueError("Select at least one result, --all, or --supplementary")

    exp.PROFILE_R_NOISE_STD = 0.0

    config = exp.ExperimentConfig(num_devices=n_devices, num_runs=n_runs)
    _set_reproducible_seed(config.random_seed)

    logger.info("Running no-profile-perturbation experiments")
    logger.info("PROFILE_R_NOISE_STD=%s", exp.PROFILE_R_NOISE_STD)
    logger.info("n_devices=%s, n_runs=%s", n_devices, n_runs)

    if supplementary:
        exp._run_supplementary(config)
    if run_r1:
        exp._run_result1(config)
    if run_r2:
        exp._run_result2(config)
    if run_r3:
        exp._run_result3(config)
    if run_r4:
        exp._run_result4(config)

    if make_figures:
        output_dir = Path(figure_output_dir) if figure_output_dir else None
        generated = generate_paper_figures(results_root=Path("results/experiments"), output_dir=output_dir)
        for figure in generated:
            logger.info("Generated figure: %s", figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run experiments with scenario r(t) perturbation disabled"
    )
    parser.add_argument("--result1", action="store_true", help="Run Result 1 / Fig. 2 data")
    parser.add_argument("--result2", action="store_true", help="Run Result 2 / Fig. 3 data")
    parser.add_argument("--result3", action="store_true", help="Run Result 3 / Fig. 4 data")
    parser.add_argument("--result4", action="store_true", help="Run Result 4 / Fig. 5 data")
    parser.add_argument("--all", action="store_true", help="Run Results 1-4")
    parser.add_argument("--supplementary", action="store_true", help="Run supplementary experiment")
    parser.add_argument("--n-devices", type=int, default=5000)
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument("--make-figures", action="store_true", help="Generate paper figures after runs")
    parser.add_argument("--figure-output-dir", default=None)
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = build_parser().parse_args()
    run_no_profile_perturbation_experiments(
        result1=args.result1,
        result2=args.result2,
        result3=args.result3,
        result4=args.result4,
        all_results=args.all,
        supplementary=args.supplementary,
        n_devices=args.n_devices,
        n_runs=args.n_runs,
        make_figures=args.make_figures,
        figure_output_dir=args.figure_output_dir,
    )


if __name__ == "__main__":
    main()
