# Broadcast coordination of distributed energy storage

Code accompanying *Population scale enables reliable dispatch of distributed
energy storage without real-time device-level sensing*.

A population of distributed storage devices receives one unidirectional
broadcast signal and responds locally, with no device-level telemetry returning
to the operator during routine dispatch. The code here builds device populations
from public metering datasets, simulates their response inside a radial
distribution feeder, calibrates an aggregate-response model, and measures when
that aggregate is accurate enough to dispatch: how the effective population size
behaves, how response structure, synchronization and availability limit it, how
it drifts and recovers, and how much of it survives the physical network.

The archived release contains the workflow for population construction, storage
simulation, aggregate-response calibration, effective-population estimation,
response-structure perturbations, synchronization analysis, availability
experiments, drift detection and aggregate-only recalibration,
network-constrained dispatch, statistical analysis and figure generation.
Environment specifications, experiment configuration files, pseudorandom seeds,
calibration/test partitions and minimal validation cases are included.

## Layout

```text
src/
  signal/        broadcast encoding, decoding, optimisation and validation
  edge/          battery model, device constraints, state machine
  estimation/    aggregate-response estimator and conformal prediction
  simulation/    fleet simulator, scenarios, constants
  analysis/      statistics used by the experiment runner
  extra/
    dataset_experiment/          canonical adapter for the 15 public datasets
    dataset_combinations/        standalone generator for the 119 mixed populations
    ieee33_device_day_simulation/
      population/                device-day population construction
      network/                   radial DistFlow feeder models
      experiments/               the simulation protocol
      figures/                   the network-deliverability experiments
    nc_excel_experiments/        the population-scale experiment runner and its configs
    <dataset>_ieee33_real_load/  one entry point per public dataset
experiments/run_experiment.py    the original single-experiment CLI
tools/
  data/          dataset download, verification and device-day pool construction
  protocols/     configuration and protocol generation (seeds, mixtures, topologies)
  analysis/      readouts and statistical analysis of the experiment output
scripts/         campaign drivers
figures/         main figures, supplementary figures, source data and tables
tests/           unit tests
download_data.sh dataset acquisition
```

## Environment

The reported runs used Python 3.10.20 on Ubuntu 22.04 (x86-64, 144 cores,
976 GB RAM). The experiments are CPU-bound and need no GPU; `torch` is used only
by the neural estimator in `src/estimation`.

```bash
conda env create -f environment.yml
conda activate broadcast-coordination
```

or

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` lists the minimum versions; `requirements-lock.txt` is the
exact package set of the environment that produced the reported results. The
package is not installed, so run everything from the repository root with

```bash
export PYTHONPATH="$PWD"
```

## Data

The 15 public datasets are not redistributed here. `download_data.sh` fetches
them from their original sources and verifies every file against a recorded
md5/sha256:

```bash
bash download_data.sh
```

Two datasets need a manual step, which the script prints: the Irish domestic
smart meters archive has to be downloaded from its Figshare page, and the RAR5
archives need `bsdtar`, `7z` or `unar`. For a faster restart,
`tools/data/make_manifest.sh` exports the full download manifest and
`tools/data/par_download.py` fetches it in parallel with resume and per-file
checksum verification; `tools/data/segmented_fetch.py` finishes rate-limited
large files with parallel byte-range segments.

After the download:

```bash
python tools/data/prepare_dataset_configs.py --all
```

writes one configuration per dataset under
`results/<dataset>_ieee33_real_load/config/` and runs the hardware preflight.

## Workflow

| Step | Code |
|---|---|
| Population construction | `src/extra/dataset_experiment/canonical_adapter.py`, `src/extra/ieee33_device_day_simulation/population/`, `tools/data/prepare_dataset_configs.py`, `tools/data/build_bdg1_cache.py` |
| Mixed-population construction | `src/extra/dataset_combinations/`, `tools/data/build_combo_pools.py`, `tools/data/verify_combo_pool.py` |
| Storage simulation | `src/edge/`, `src/simulation/`, `src/signal/`, `src/extra/ieee33_device_day_simulation/experiments/protocol.py`, `.../original_adapter/eps_adapter.py` |
| Aggregate-response calibration | `src/estimation/`, `run_e1` in `src/extra/nc_excel_experiments/run.py` |
| Effective-population estimation | `src/extra/nc_excel_experiments/metrics.py`, `.../coupling.py` |
| Response-structure perturbations | `src/extra/nc_excel_experiments/control_logic.py`, `.../local_policy.py`, `.../dominant.py` |
| Synchronization analysis | `run_e2`, `run_e3` and `src/extra/nc_excel_experiments/phase.py` |
| Availability experiments | `run_e6`, `run_e15` |
| Drift detection and aggregate-only recalibration | `run_e4` and `src/extra/nc_excel_experiments/drift.py` |
| Network-constrained dispatch | `src/extra/ieee33_device_day_simulation/network/`, `src/extra/nc_excel_experiments/topology.py`, `src/extra/dataset_experiment/run_constrained_experiment.py`, `src/extra/ieee33_device_day_simulation/figures/` |
| Statistical analysis | `src/analysis/statistics.py`, `tools/analysis/` |
| Figure generation | `figures/` |

## Reproducing the figures

Figure 1 is a schematic and is not produced by this code.

### Figures 2 to 4 and Supplementary Figs. S1-S22

These come from one runner and one protocol file per experiment. `results_root`
inside each configuration decides where the output goes; the runner is
restartable and skips cells that are already complete.

| Figure | Experiment | Command |
|---|---|---|
| Fig. 2, and S1-S4, S6-S8 | the scale boundary of the 15 published populations | `python -m src.extra.nc_excel_experiments.run --protocol src/extra/nc_excel_experiments/configs/e1_full_v2.yaml --experiments E1` |
| Fig. 2, and S2, S7 | the same on the 119 mixed populations | `bash scripts/run_mix_batches.sh E1 110 0 29` |
| Fig. 3c-f, and S12 | controller synchronization and homogeneity | `python -m src.extra.nc_excel_experiments.run --protocol src/extra/nc_excel_experiments/configs/e2_full_v2.yaml --experiments E2` |
| Fig. 3g, and S13, S14 | phase coherence under periodic broadcast | `python -m src.extra.nc_excel_experiments.run --protocol src/extra/nc_excel_experiments/configs/e3_full_v2.yaml --experiments E3` |
| Fig. 3a-b, h, and S5, S11 | local control and response mechanisms | `python -m src.extra.nc_excel_experiments.run --protocol src/extra/nc_excel_experiments/configs/protocol_supp_full.yaml --experiments E13` |
| S9 | capacity concentration | `python -m src.extra.nc_excel_experiments.run --protocol src/extra/nc_excel_experiments/configs/protocol_supp_full.yaml --experiments E14` |
| Fig. 4a-c, and S16 | availability structure and participation | `python -m src.extra.nc_excel_experiments.run --protocol src/extra/nc_excel_experiments/configs/protocol_e6_full.yaml --experiments E6` |
| S17, S18 | the second family of availability structures | `python -m src.extra.nc_excel_experiments.run --protocol src/extra/nc_excel_experiments/configs/protocol_e15_full.yaml --experiments E15` |
| Fig. 4d-h, and S19, S20 | controller drift, detection and aggregate-only recalibration | `python -m src.extra.nc_excel_experiments.run --protocol src/extra/nc_excel_experiments/configs/e4_v2.yaml --experiments E4` |
| S21 | the same on the mixed populations | `bash scripts/run_mix_any.sh E4 110 0 0` |
| S19, long-horizon closed-loop state | 30-day closed-loop audit | `python -m src.extra.nc_excel_experiments.run --protocol src/extra/nc_excel_experiments/configs/protocol_e9_phase1.yaml --experiments E9` |
| S22 | the scale boundary under three feeder topologies | `bash scripts/run_topology_mix.sh` |

The identifiers `E1` to `E15` are the internal run names of the experiments; they
appear in the command line, in the configuration file names and in the output
directory names, and the table above is the mapping to the figures. The runner
also carries four diagnostic experiments that no figure uses.

The whole pipeline on the 15 published populations:

```bash
bash scripts/run_all.sh
```

The mixed populations, one seed batch at a time, or as one campaign:

```bash
bash scripts/run_mix_batches.sh E1 110 0 29
bash scripts/run_mix_any.sh E13 110 0 0
bash scripts/mix_campaign.sh 25 110 0
```

For the feeder-topology arms, `NC_TOPOLOGY` selects the case file,
`NC_NETWORK_FEEDBACK=1` lets the network constraint feed back into the dispatch
instead of being recorded as a diagnostic, and `NC_NETWORK_DERATE` scales the
capacities:

```bash
python tools/protocols/calibrate_topology.py <dataset>   # writes results/_topology_calibration.json
bash scripts/run_topology_mix.sh
```

The figures themselves are then drawn by `figures/`; see `figures/README.md` for
the script-to-figure map, the expected input layout and the mapping of
`AppFig1..AppFig22` to Supplementary Figs. S1-S22.

### Figure 5 and Supplementary Figs. S23-S40

The network-deliverability experiments live in
`src/extra/ieee33_device_day_simulation/figures/` and run their own simulations,
comparing the broadcast controller with eight reference controllers under feeder
constraints. Each script writes its data as CSV and JSON and its figures as PNG
and PDF into its own result directory. The controllers, the stress scenarios and
the metrics are defined in
`src/extra/ieee33_device_day_simulation/figures/README.md`.

| Supplementary note | Figures | Scripts |
|---|---|---|
| Note 12, transferability across datasets and operating domains | S23-S25 | `run_e20_transfer.py`, `run_e20_rotations.py`, `run_e20_fig3a_transfer.py`, `plot_e22_interim_r2_scaling.py` |
| Note 13, effective population size and predictable aggregate response | S26-S31 | `run_e21_mixed_scenarios.py`, `run_e21_mixed_gamma_boundary.py`, `run_e21_gamma_universal_boundary.py`, `run_e21_pairwise_r2.py`, `run_e21_pairwise_curtailment.py`, `run_e21_curtailment_baseline_supplement.py` |
| Note 14, algorithm robustness and network implementation | S32-S34 | `run_e22_ieee69_complexity.py`, `run_e22_trained_eps_comparison.py`, `fig4d_final_protocol.py`, `fig4d_extra_baselines.py`, `plot_fig4d_final_overall.py`, `plot_dataset_algorithm_comparison.py` |
| Note 15, physical feasibility boundaries under network stress | S35-S38 | `run_e23_ieee69_relative_boundary.py`, `run_e23_ieee69_failure_boundary.py`, `run_e23_neff.py` |
| Note 16, safety auditing under realistic feeder constraints | S39 | `run_e24_ieee123_audit.py` |
| Note 17, component-level network safety across IEEE-69 stress conditions | S40 | `run_e22_ieee69_complexity.py`, panel `ieee69_constraint_component_audit` |

Figure 5 draws on the same family: the network acceptance of the nine methods,
the curtailment reduction on the IEEE-33, IEEE-69 and IEEE-123 feeders, and the
dataset-level distributions.

```bash
bash scripts/run_network_audits.sh --all              # IEEE-69 boundary and IEEE-123 audit
python -m src.extra.ieee33_device_day_simulation.figures.run_e22_ieee69_complexity
python -m src.extra.ieee33_device_day_simulation.figures.run_e21_curtailment_baseline_supplement
python scripts/train_eps69.py <dataset>               # per-dataset model for the IEEE-69 arm
```

The script names carry the internal campaign numbers under which these runs were
executed; the table above is the mapping to the supplementary notes and figures.

## Configurations, seeds and partitions

All protocol files are in `src/extra/nc_excel_experiments/configs/`. Naming:
`<experiment>_full*.yaml` for the 15 published populations,
`<experiment>_mix_s<NN>.yaml` for mixed-population seed batch NN,
`e1_topo_<topology>_{fb,bypass}.yaml` for the topology arms, and `*_smoke.yaml`
for the minimal validation cases. The mixed-population and topology protocols are
generated, not hand-written:

```bash
python tools/protocols/make_mix_protocol.py    --experiment E1  --seed-index 0 --results-root results/e1_mix_s00  --workers 110 --out <config>
python tools/protocols/make_mix_protocol_v2.py --experiment E13 --seed-index 0 --results-root results/e13_mix_s00 --workers 110 --out <config>
python tools/protocols/make_protocols.py
python tools/protocols/make_mix_protocols.py
```

Randomness is fixed by `random_seed` in the protocol, `20260720` in the reported
runs. Each experiment offsets it by a constant and each cell derives its own
stream from that offset, so a batch can be re-run or resumed without changing any
other cell. The mixed populations carry their own fixed seeds, documented in
`src/extra/dataset_combinations/README.md`.

Calibration and test replications are disjoint by construction. The scale
experiment draws `train_replications + test_replications` independent
replications per cell, fits the aggregate response on the first
`train_replications`, freezes it, and evaluates only on the remaining
`test_replications`, 10 and 120 in the reported runs. The drift experiment adds a
validation block for detector calibration and a hold-out block that the
recalibration arm never sees. The mixed populations are partitioned into train,
validation and test before any experiment runs, under the partition seed fixed in
`src/extra/dataset_combinations/constants.py`.

## Minimal validation cases

None of these need the full datasets except where noted:

```bash
python -m pytest tests -q -o addopts=""              # 365 unit tests
python -m src.extra.dataset_combinations.self_test --data-root data
bash scripts/run_smoke.sh                            # the smoke protocol of every experiment
```

`scripts/run_smoke.sh` runs each experiment on a reduced grid and needs the
datasets. `tools/data/verify_combo_pool.py` re-checks a built mixed-population
pool point by point against its source datasets.

## Notes on this release

The reported results were produced in several working copies of this repository,
one per experiment family, so that concurrent campaigns could not overwrite each
other's `results/`. This release is the merge of those copies into a single tree.
The merge is exact: the edits of the different copies were disjoint, every
experiment's code path is the one that produced its results, and the full test
suite passes on the merged tree. One behaviour is the union rather than a single
copy: the scaling grid `_legal_n_values` admits populations down to N=1, which
the small-N and topology campaigns required, where the earlier copies stopped at
N=2. For every published configuration the two rules give the same grid, because
no reported candidate grid contains N=1 and every reported population is far
above the point where the rule differs.

`results/` and `data/` are not part of the release. The experiment outputs that
the figures consume are deposited with the paper as Source Data.

## License

MIT, see `LICENSE`.
