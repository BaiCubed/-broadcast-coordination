# Code and execution protocol

The code directory contains the complete experiment implementations, deterministic data-generation implementation, figure/table/report builders, and verification utilities. All commands below are run from `sg_part/code`.

## Installation

Python 3.10 or newer is required. Install the pinned lower-bound dependencies in an isolated environment:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The optional IEEE-123 OpenDSS validation requires `requirements-e24-multiphase.txt`. CPU-only PyTorch is sufficient. For stable noninteractive plotting, set `MPLCONFIGDIR` to a writable temporary directory.

## Commands

- `scripts/run_reproduction.sh --check`: validate layout, seeds, protocol values, syntax, English documentation, and output-manifest generation.
- `scripts/run_reproduction.sh --data`: preprocess downloaded sources into canonical 288-point device-day caches and generate deterministic mixed and pairwise fleets.
- `scripts/run_reproduction.sh --e1-e4`: prepare dataset-specific configurations and run the workbook experiments E1-E4.
- `scripts/run_reproduction.sh --experiments`: run E20-E24 in dependency order and write result manifests.
- `scripts/run_reproduction.sh --artifacts`: regenerate plots, composite panels, Appendix, supplementary tables, presentations and reports from completed results.
- `python tools/verify_outputs.py`: check generated files against `audit/OUTPUT_REPRODUCTION_MANIFEST.csv`.
- `python scripts/verify_release.py`: verify the release-level summaries and figures without running long experiments.

The runner creates a `code/data` link to the sibling release `data/` directory because the original experiment modules use the conventional `data/` path. Generated `results/`, `outputs/`, logs and checkpoints remain local and are ignored by the release `.gitignore`.

The short verification command is the recommended check for a normal checkout:

```bash
python scripts/verify_release.py
```

It validates the 14 mixed-scenario CSVs, 105 pairwise CSVs, release protocol, English documentation, summary-table checksums and regenerated figure files. It deliberately does not start the long network experiments.

## Data and splits

`generation/preprocess.py` reads `data/configuration/preprocessing_config.json`, converts each downloaded source to canonical device-day records, and caches them under the data root. `python -m src.extra.dataset_combinations all` uses the recorded scenario and pairwise indexes, partition seeds, 30 test seeds, and fixed sampling modes. No test device-day is used for estimator fitting or calibration.

The estimator protocol is recorded in `config/reproducibility.json`: response-sign-stratified 70% fit and 30% calibration, an internal seed-42 validation split for neural-network early stopping, dual direction-specific quantile heads at 0.1/0.5/0.9, hidden layers 128-64-32 with ReLU and 0.15 dropout, AdamW learning rates 0.003 cold and 0.001 warm, weight decay 0.001, batch size 32, minimum epoch budget 150, patience 40, gradient clipping 1.0, and 90% conformal coverage.

## Artifact mapping

`audit/OUTPUT_REPRODUCTION_MANIFEST.csv` is generated from the reference `outputs/` tree and has one row per file. The `stage`, `generator`, and `command` columns provide the complete chain for Appendix assets, supplementary tables, composite figures, reports, publication figures, data-availability files, and the mirrored E20-E24 source figures. The manifest is the acceptance checklist; a missing generated artifact is a failed verification rather than an omitted result.
