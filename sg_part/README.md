# Broadcast Coordination Reproduction Release

This directory is the local draft of the release branch. It is organised as two independent components:

```text
sg_part/
├── code/   # source code, protocols, generators, experiment runners and build scripts
└── data/   # derived public release data, indexes and source metadata
```

The release does not contain third-party raw archives, trained checkpoints, or the large local `results/` tree. Those inputs are downloaded from the original providers listed in [`data/README.md`](data/README.md), then converted by the included preprocessing and deterministic fleet-composition code. The included derived CSV files let a reviewer validate the published composition and summary-data contract without downloading all raw data.

## Reproduction scope

The reproducibility target is every artifact in the working `outputs/` tree: the E1/E20/E21/E22/E23/E24 experiment results, publication figures, composite figures, Appendix figures and document, supplementary figures and tables, editable presentations, the mixed-methods report, and the E23/E24 report package. The artifact-level mapping is [`code/audit/OUTPUT_REPRODUCTION_MANIFEST.csv`](code/audit/OUTPUT_REPRODUCTION_MANIFEST.csv); its JSON summary records the current reference artifact count and distinguishes generated artifacts from metadata or manually edited deliverables.

Each generated artifact is tied to a script, upstream inputs, and a command in that manifest. A file classified as `metadata_or_manual` is retained as a reference or editorial asset; its upstream source is explicitly recorded instead of claiming that a plotting script can recreate manual prose or an imported reference image.

Experiment ownership and plot ownership are checked by [`code/audit/OWNERSHIP_AUDIT.md`](code/audit/OWNERSHIP_AUDIT.md). Shared numerical foundations are declared explicitly. E23 is allowed to read E22 model/result artifacts, and E24 is allowed to read E22/E23 artifacts; these are read-only data inputs. Each leaf plot has a dedicated entrypoint under `code/plots/leaf/`; Appendix, supplementary tables, composite figures and reports are registered as composition stages.

## Quick start

```bash
cd sg_part/code
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/run_reproduction.sh --check
```

`--check` validates the release layout, protocol values, English documentation, source-code syntax, and the complete output manifest. It does not run a simulation. For the requested short reproducibility verification, run `python scripts/verify_release.py`; this regenerates the included release-level tables and figures and checks their deterministic checksums without running the long experiments.

## Full chain

Run stages in order. The data stage requires the official source directories named in `data/configuration/preprocessing_config.json`.

```bash
cd sg_part/code
./scripts/run_reproduction.sh --data
./scripts/run_reproduction.sh --e1-e4
./scripts/run_reproduction.sh --experiments
./scripts/run_reproduction.sh --artifacts
python tools/verify_outputs.py --manifest audit/OUTPUT_REPRODUCTION_MANIFEST.csv --output-root outputs
```

The equivalent full-chain command is `./scripts/run_reproduction.sh --all`. It is not required for the short verification described above. E20-E24 are intentionally separate because their 30-seed network experiments can require many hours, substantial RAM, and several gigabytes of temporary results. Every stage writes a log under `code/reproduction_logs/` and reuses existing checkpoints/results where the upstream runner supports resume.

To rebuild only figures, tables, Appendix, presentations and reports from completed result tables, run `./scripts/run_reproduction.sh --artifacts`. To inspect one experiment's exact parameters, read its `EXPERIMENT_DESIGN.md` and the corresponding runner under `code/src/extra/ieee33_device_day_simulation/figures/`.

## Protocol commitments

The canonical protocol is [`code/config/reproducibility.json`](code/config/reproducibility.json). It records the 288-step (5-minute) device-day resolution, global and paired seeds, composition and estimator splits, neural-network architecture and optimization parameters, device limits, and E22-E24 scenario parameters. Experiment-specific manifests under `results/E20` through `results/E24` are written by the runners and take precedence over generic defaults when an experiment overrides a parameter.

The estimator uses a 70/30 fit/calibration split with response-sign stratification; the neural-network validation split is internal to the fit subset and uses seed 42. The first calibration seed is `42 + fit_count` in the estimator implementation. E22 has its own `BASE_SEED = 22_000_000`; the global composition seed is not silently substituted for experiment seeds.

## Verification policy

A run is considered verified only when the command exits successfully, the expected files in the artifact manifest exist, and numerical comparisons against the reference tables pass the tolerances stated by the relevant experiment manifest. The release scripts never label an unexecuted long experiment as verified. `audit/output_verification.json` is produced by `tools/verify_outputs.py` and is the machine-readable coverage report.

No upload, push, or remote branch operation is performed by these scripts.
