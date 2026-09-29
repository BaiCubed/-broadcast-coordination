<div align="center">

<h1>Population scale enables reliable dispatch of distributed energy storage without real-time device-level sensing</h1>

<p><b>One broadcast signal, no routine device telemetry: when does a storage population become predictable, controllable and physically deliverable?</b></p>

<p>
<a href="https://www.python.org/"><img alt="Python 3.10" src="https://img.shields.io/badge/python-3.10-3776AB?logo=python&logoColor=white"></a>
<a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-2EA043"></a>
</p>

<p>
<a href="#overview">Overview</a> ·
<a href="#installation">Installation</a> ·
<a href="#data">Data</a> ·
<a href="#quick-start">Quick start</a> ·
<a href="#reproducing-the-paper">Reproducing the paper</a> ·
<a href="#configurations-seeds-and-data-splits">Seeds and splits</a> ·
<a href="#citation">Citation</a>
</p>

</div>

---

## Overview

Large populations of distributed energy storage can be coordinated through common signals, but it is not obvious when routine dispatch stays reliable without continuous real-time device-level telemetry. In the architecture studied here, the operator sends **one broadcast signal**, every device converts it locally into a power response from its own state, and **no device state returns to the operator during routine dispatch**. Registration, commissioning, offline calibration and aggregate-level monitoring remain available.

This repository contains the complete code for the data-informed simulations of the paper. It builds device populations from **15 published empirical populations** and **119 constructed mixed populations**, simulates their response to broadcast dispatch, calibrates an aggregate-response model, and measures where reduced-telemetry coordination works and where it stops working:

- **Aggregate predictability emerges before reliable controllability.** The aggregate response becomes reproducible at modest population sizes, while reliable control arrives later and depends more on the population.
- **Effective population size, not nominal fleet size, sets the scale.** $N_{\mathrm{eff}}$ measures how dependence between devices and incomplete participation reduce the statistical value of a nominal fleet. The effective margin $\Gamma$ normalizes each population by its own control threshold.
- **Averaging removes noise, not systematic error.** Heterogeneous response delays and structured local dynamics limit control even in large populations.
- **Populations evolve.** Structured availability reduces effective scale, controller drift invalidates the calibrated model, and aggregate-only recalibration restores aggregate-response fidelity under the tested perturbations.
- **Networks decide what can be delivered.** On the IEEE-33, IEEE-69 and IEEE-123 feeders, statistically controllable responses can become physically infeasible under network stress and spatial concentration.

Reliable dispatch without continuous real-time device-level telemetry therefore requires four conditions together: sufficient effective scale, compatible local responses, a valid aggregate model and a feasible network state.

## Highlights

| Result | What the code reproduces | Figures |
|---|---|---|
| Predictability precedes control | The unexplained response fraction falls approximately as $a/(N+a)$ ($a$ = 4.84 for the 15 published populations, 4.63 for the 119 mixed populations); the median prediction threshold is $N_{95}$ = 89 devices; 14 of 15 published and 104 of 119 mixed populations reach it, while reliable control is reached later and less often | Fig. 2, S1-S8 |
| Effective scale organizes control | Normalizing by the effective margin $\Gamma$ compresses the spread of the 50% control crossing from 2.3-fold to 1.4-fold (published) and from 2.5-fold to 1.7-fold (mixed) | Fig. 2g, S6-S9 |
| Systematic structure limits averaging | Price-response and random-delay mechanisms do not reach the reliability criterion within the tested range; control generally degrades before excess synchronization becomes detectable | Fig. 3, S10-S15 |
| Population evolution | Correlated and behavioural availability shrink $N_{\mathrm{eff}}$ far more than independent absence; after controller drift, the cumulative excess loss of a frozen model is 31x to 67x that of aggregate-only recalibration | Fig. 4, S16-S22 |
| Network deliverability | Under 80% distal placement only 3.1-25.8% of the requested response remains feasible; EPS keeps 90.7%, 84.5% and 89.1% of the centralized greedy reference under 50% feeder concentration on IEEE-33, IEEE-69 and IEEE-123 | Fig. 5, S33-S41 |

## Repository structure

```text
.
├── README.md                      this page
├── LICENSE                        MIT licence
├── download_data.sh               one-command download of the 15 public datasets (+ 2 auxiliary sources) with checksum verification
├── environment.yml                conda environment (Python 3.10)
├── requirements.txt               pip dependencies
├── requirements-lock.txt          exact versions of the reported runs
├── requirements-opendss.txt       optional OpenDSS extra
├── pyproject.toml                 package metadata
├── configs/
│   └── reproducibility.json       seeds, data splits, estimator and device parameters, network scenarios
├── derived_data/                  the 119 constructed mixed-population datasets, their indices and generation configuration
├── verification/                  checksums of the derived-data summaries
├── src/
│   ├── signal/                    broadcast encoding, decoding, optimization and validation
│   ├── edge/                      battery model, device constraints and local state machine
│   ├── estimation/                aggregate-response estimator and conformal calibration
│   ├── simulation/                fleet simulator, scenarios and constants
│   ├── analysis/                  statistics shared by the experiment runners
│   └── extra/
│       ├── population_experiments/        runner and configurations of the population-scale experiments (Figs. 2-4, S1-S23)
│       ├── dataset_combinations/          standalone generator of the 119 mixed populations
│       ├── dataset_experiment/            canonical adapter that runs each public dataset through the feeder simulation
│       ├── ieee33_device_day_simulation/  device-day populations, radial feeders, dispatch protocol and
│       │   └── network_experiments/       the network-deliverability experiments (Fig. 5, S24-S41)
│       └── <dataset>_ieee33_real_load/    one entry point per public dataset
├── tools/
│   ├── data/                      preprocessing, per-dataset configurations and mixed-population device pools
│   ├── protocols/                 protocol generation for mixed populations, feeder topologies and availability families
│   └── analysis/                  frozen-test R^2 and N95 readout
├── scripts/                       reproduction drivers, smoke test and derived-data checks
├── figures/                       figure scripts for Figs. 2-5 and S1-S41, style modules and table generator
├── experiments/                   synthetic-population command-line runner
└── tests/                         unit tests
```

## Installation

The reported runs used Python 3.10 on Ubuntu 22.04, CPU only.

**Conda (recommended)**

```bash
conda env create -f environment.yml
conda activate broadcast-coordination
```

**pip**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt           # or requirements-lock.txt for the exact versions of the reported runs
```

**PYTHONPATH.** All commands are run from the repository root with the root on the module path. The shell drivers in `scripts/` set it themselves; for direct `python -m` calls use

```bash
export PYTHONPATH="$PWD"
```

Alternatively, `pip install -e .` installs the `src` package in editable mode.

**Optional OpenDSS extra.** The three-phase cross-check of the IEEE-123 feeder (`ieee123_opendss_check.py`, not part of any paper figure) needs `opendssdirect.py`:

```bash
pip install -r requirements-opendss.txt
```

**Fonts.** Figures were rendered with Arial. Without it, matplotlib falls back to Liberation Sans (metric compatible), Helvetica and DejaVu Sans; any `*.ttf` placed in `figures/fonts/` is registered automatically.

## Data

### The 15 published populations

The original public datasets are **not redistributed** here: each is governed by its provider's licence, and together they exceed what a code repository should hold. Instead, the repository gives the official source of every dataset, a one-command download with checksum verification, and the complete preprocessing that turns the raw files into the canonical device-day records used by all experiments.

```bash
bash download_data.sh                                        # downloads into data/ and verifies SHA-256 / MD5 checksums
python tools/data/preprocess.py                              # raw files -> canonical device-day records
python tools/data/prepare_dataset_configs.py --all           # per-dataset simulation configurations
```

| Population (Supplementary Table S1) | Resource / signal | Resolution | Sources | Directory under `data/` | Official source |
|---|---|---|---:|---|---|
| BDG1 | building electricity | 1 h | 507 | `bdg1_building_data_genome` | [Building Data Genome 1](https://github.com/buds-lab/the-building-data-genome-project) |
| Low Carbon London | household electricity | 30 min | 5,000 | `low_carbon_london` | [London Datastore](https://data.london.gov.uk/download/vqm0d/3527bf39-d93e-4071-8451-df2ade1ea4f2/LCL-FullData.zip) |
| BDG2 | building electricity + solar | 1 h | 1,570 | `bdg2_building_data_genome` | [Building Data Genome 2](https://github.com/buds-lab/building-data-genome-project-2) |
| Danish heat meters | thermal demand | 1 h | 2,400 | `danish_smart_heat_meters` | [doi:10.5281/zenodo.6563114](https://doi.org/10.5281/zenodo.6563114) |
| Smart Grid Smart City | load + generation | 30 min | 600 | `smart_grid_smart_city` | [data.gov.au](https://data.gov.au/data/dataset/smart-grid-smart-city-customer-trial-data) |
| HEAPO | heat-pump electricity | 15 min / daily | 1,362 | `heapo_heat_pumps` | [doi:10.5281/zenodo.15056919](https://doi.org/10.5281/zenodo.15056919) |
| GoiEner | smart-meter electricity | 1 h | 5,000 | `goiener_smart_meters` | [doi:10.5281/zenodo.7362094](https://doi.org/10.5281/zenodo.7362094) |
| European LV urban-8k | LV active-power profile | 24/168 points | 5,000 | `european_lv_urban_8087` | [doi:10.17632/685vgp64sm.1](https://doi.org/10.17632/685vgp64sm.1) |
| European LV rural | LV active/reactive profile | 24/168 points | 2,731 | `european_lv_rural_2731` | [doi:10.17632/gspyzvvrhm.2](https://doi.org/10.17632/gspyzvvrhm.2) |
| European LV urban-35k | LV active/reactive profile | 24/168 points | 12,000 | `european_lv_urban_35297` | [doi:10.17632/gspyzvvrhm.2](https://doi.org/10.17632/gspyzvvrhm.2) |
| Norway AMI | MV/LV AMI profile | 1 h | 2,994 | `norway_ami_energy_distribution` | [doi:10.17632/jv3rz8k35r.1](https://doi.org/10.17632/jv3rz8k35r.1) |
| CAMSL-JP | smart-meter electricity | 30 min | 1,423 | `camsl_japan_smart_meters` | [doi:10.17632/cmpsyncmmk.1](https://doi.org/10.17632/cmpsyncmmk.1) |
| Irish CER | import/export household | 30 min | 2,904 | `irish_domestic_smart_meters` | [doi:10.6084/m9.figshare.31851922.v2](https://doi.org/10.6084/m9.figshare.31851922.v2) |
| OPSD | household multi-channel | 15 min | 11 | `opsd_household_data` | [Open Power System Data](https://data.open-power-system-data.org/household_data/opsd-household_data-2020-04-15.zip) |
| COMPLETE-EC | community load/PV/BESS/EV | 15 min | 250 | `complete_energy_community` | [doi:10.5281/zenodo.7602546](https://doi.org/10.5281/zenodo.7602546) |

Two further sources of Supplementary Table S1 are downloaded by the same script: **NextGen**, the device-day battery calibration population of the feeder simulation ([doi:10.5281/zenodo.14885589](https://doi.org/10.5281/zenodo.14885589), `data/nextgen`), and **data2**, measured EV charging sessions ([doi:10.17632/c7gg94tmvz.3](https://doi.org/10.17632/c7gg94tmvz.3), `data/data2`). Together with the 15 populations they form the 17 dataset configurations of the IEEE-69 audit (Supplementary Fig. S41).

Notes on acquisition:

- The source list, local paths and adapter parameters are in [`derived_data/configuration/source_metadata.json`](derived_data/configuration/source_metadata.json) and [`derived_data/configuration/preprocessing_config.json`](derived_data/configuration/preprocessing_config.json).
- Archives in RAR format (European LV rural and urban-35k) need `bsdtar`, `7z` or `unar`. If the Irish CER file cannot be fetched automatically, the script prints the page and file name for a manual download.
- `download_data.sh` skips files that are already present and re-verifies their checksums, so it can be re-run safely. `tools/data/par_download.py` and `tools/data/segmented_fetch.py` are optional parallel and segmented downloaders for slow links.
- Please cite the original data providers; the references are listed with Supplementary Table S1.

### The 119 constructed mixed populations (included)

The mixed populations are controlled recombinations of the 15 published sources and are **included in this repository** under [`derived_data/`](derived_data):

| Path | Content |
|---|---|
| [`derived_data/generated_data/mixed_scenarios/`](derived_data/generated_data/mixed_scenarios) | 14 multi-source populations, scenario S1-A to scenario S6-C (Supplementary Table S2), one CSV each |
| [`derived_data/generated_data/pairwise/`](derived_data/generated_data/pairwise) | 105 pairwise 50/50 populations, `P001.csv` to `P105.csv` |
| [`derived_data/index/`](derived_data/index) | pair index and seed index of every population, partition and test seed |
| [`derived_data/configuration/`](derived_data/configuration) | generation, scenario, pairwise-seed, preprocessing and source configuration |
| [`derived_data/SHA256SUMS`](derived_data/SHA256SUMS) | checksums of every file above |

Each CSV records, for every one of the 30 paired test seeds, the composition of that population and its simulated dispatch outcome: the scenario files give the nominal and actual source weights, device counts per source, fleet and algorithm seeds for ten methods under aggregate and IEEE-33 network modes; the pairwise files give the two sources, their weights and device counts and the pairwise seed under the aggregate mode. The scenario labels S1-A to S6-C are the scenario names of Supplementary Table S2 and are unrelated to the numbering of the Supplementary Figures. The device-level fleets themselves are regenerated deterministically from the public data with `python -m src.extra.dataset_combinations all` (see [`src/extra/dataset_combinations/README.md`](src/extra/dataset_combinations/README.md)).

### Source Data

Source Data for the figures are provided with the paper. [`figures/make_source_data.py`](figures/make_source_data.py) and [`figures/make_fig5.py`](figures/make_fig5.py) regenerate the underlying tables from the experiment outputs.

## Quick start

These checks run offline in about a minute and need no downloaded data.

```bash
export PYTHONPATH="$PWD"

python -m pytest tests -q -o addopts=""        # unit tests
python scripts/check_derived_data.py           # 14 + 105 mixed-population files present and well formed
python scripts/verify_derived_data.py          # rebuild summary tables and figures from derived_data/ and compare checksums
(cd derived_data && sha256sum -c SHA256SUMS)   # file-level integrity of the included data
```

`verify_derived_data.py` writes `reproduced_tables/` and `reproduced_figures/` and compares the summary tables with [`verification/release_summary_checksums.json`](verification/release_summary_checksums.json).

With the public data downloaded and preprocessed, the minimal validation cases of the population-scale experiments (all except E4) run in minutes:

```bash
bash scripts/run_smoke.sh
```

## Reproducing the paper

### How the pieces fit together

1. **Data.** `download_data.sh`, `tools/data/preprocess.py` and `tools/data/prepare_dataset_configs.py --all` (see [Data](#data)). The mixed populations additionally need their device-level fleets and pools:

   ```bash
   python -m src.extra.dataset_combinations all
   python tools/data/build_combo_pools.py --seeds 0-29 --workers 16
   ```

2. **Experiments** write to `results/<configuration>/<experiment>/`. Population-scale experiments are run by one runner with a protocol file; network experiments are Python modules.
3. **Figure inputs.** The population-figure scripts read copies of the result directories under `figures/data/<name>/`; the exact copy commands are in [`figures/README.md`](figures/README.md#figure-inputs). The network-figure scripts read `results/` directly.
4. **Figures** are written to `figures/out/` (main figures, Fig. 5 panels, S24-S41) and `figures/out/supplementary/` (S1-S23), each as PDF and PNG, together with `*_stats.json` files holding every number printed in the panels.

The tables below use three shorthands:

```bash
export PYTHONPATH="$PWD"
RUN="python -m src.extra.population_experiments.run"
CFG=src/extra/population_experiments/configs
NX=src.extra.ieee33_device_day_simulation.network_experiments
```

### Experiment index

| Experiment (paper wording) | Paper | Runner id / module | Configuration | Result directory |
|---|---|---|---|---|
| E1 population-scale experiment, 15 published populations | Result 1, Methods | `E1` | [`E1_population_scale.yaml`](src/extra/population_experiments/configs/E1_population_scale.yaml) | `results/E1_population_scale/E1_population_scale` |
| E1 on the 119 mixed populations (composition seeds 0-29) | Result 1, Note 1 | `E1` | `E1_population_scale_mixed_s00.yaml` ... `_s29.yaml` | `results/E1_population_scale_mixed_sNN/E1_population_scale` |
| E1 extended scale grid (adds N = 1, 5, 15, 30; input of Notes 12-13) | Notes 12-13 | `E1` | [`E1_population_scale_extended.yaml`](src/extra/population_experiments/configs/E1_population_scale_extended.yaml) | `results/E1_population_scale_extended/E1_population_scale` |
| E1 under three feeder topologies | Note 11 | `E1` | `E1_feeder_topology_{ieee33,ieee69,ieee123}.yaml` | `results/E1_feeder_topology/<feeder>/E1_population_scale` |
| E2 controller synchronization | Result 2, Note 7 | `E2` | [`E2_controller_synchronization.yaml`](src/extra/population_experiments/configs/E2_controller_synchronization.yaml) | `results/E2_controller_synchronization/E2_controller_synchronization` |
| Phase coherence under two surrogates | Result 2, Note 7 | `phase_coherence` | [`phase_coherence.yaml`](src/extra/population_experiments/configs/phase_coherence.yaml) | `results/phase_coherence/phase_coherence` |
| Local response mechanisms | Result 2, Notes 2 and 6 | `response_mechanisms` | [`response_mechanisms_and_capacity_concentration.yaml`](src/extra/population_experiments/configs/response_mechanisms_and_capacity_concentration.yaml) | `results/response_mechanisms_and_capacity_concentration/response_mechanisms` |
| Capacity concentration | Note 5 | `capacity_concentration` | same file | `results/response_mechanisms_and_capacity_concentration/capacity_concentration` |
| Structured availability | Result 3, Note 8 | `structured_availability` | [`structured_availability.yaml`](src/extra/population_experiments/configs/structured_availability.yaml) | `results/structured_availability/structured_availability` |
| Second availability family (published / mixed) | Note 8 | `availability_second_family` | [`availability_second_family.yaml`](src/extra/population_experiments/configs/availability_second_family.yaml), [`availability_second_family_mixed.yaml`](src/extra/population_experiments/configs/availability_second_family_mixed.yaml) | `results/availability_second_family[_mixed]/availability_second_family` |
| Long-horizon state evolution | Note 9 | `long_horizon_state` | [`long_horizon_state.yaml`](src/extra/population_experiments/configs/long_horizon_state.yaml) | `results/long_horizon_state/long_horizon_state` |
| E4 controller drift and aggregate-only recalibration (published / mixed) | Result 3, Note 10 | `E4` | [`E4_controller_drift.yaml`](src/extra/population_experiments/configs/E4_controller_drift.yaml), [`E4_controller_drift_mixed.yaml`](src/extra/population_experiments/configs/E4_controller_drift_mixed.yaml) | `results/E4_controller_drift[_mixed]/E4_controller_drift` |
| Transfer across datasets and operating domains | Note 12 | `train_pooled_model`, `transfer_across_datasets`, `transfer_alternative_splits` | module defaults | `results/transfer/` |
| Effective scale of published and mixed populations | Note 13 | `train_mixed_model`, `effective_scale_published`, `effective_scale_mixed`, `pairwise_predictability`, `pairwise_curtailment`, `mixed_reference_controllers` | module defaults | `results/effective_scale_mixtures/` |
| IEEE-69 network implementation and feeder comparison | Result 4, Notes 14 and 17 | `ieee69_network_implementation`, `ieee69_trained_vs_transferred`, `ieee33_trained_model`, `constraint_sweep*`, `feeder_comparison_*` | module defaults | `results/ieee69_network_implementation/` |
| Physical feasibility under network stress, spatial heterogeneity | Result 4, Note 15 | `network_stress_boundary`, `spatial_heterogeneity`, `spatial_heterogeneity_trained_model` | module defaults | `results/network_stress_boundary/` |
| IEEE-123 safety audit | Result 4, Note 16 | `ieee123_trained_model`, `ieee123_safety_audit` | module defaults | `results/ieee123_safety_audit/` |

Network modules are documented in [`src/extra/ieee33_device_day_simulation/network_experiments/README.md`](src/extra/ieee33_device_day_simulation/network_experiments/README.md). They need the per-dataset network baselines and the extended E1 grid first (stage `--network-inputs` below).

### Figure 2 · Population-scale transitions in aggregate predictability and reliable control

```bash
$RUN --protocol $CFG/E1_population_scale.yaml --experiments E1
bash scripts/run_mix_batches.sh 110 0 0                    # E1 on the 119 mixed populations, composition seed 0
python tools/analysis/predictability_readout.py --source results/E1_population_scale/E1_population_scale --output figures/data/predictability_readout
python tools/analysis/predictability_readout.py --source results/E1_population_scale_mixed_s00/E1_population_scale --output figures/data/predictability_readout_mixed
python figures/make_fig2.py                                # Fig. 2 (all panels)
python figures/make_fig2a.py                               # panel a as a standalone map
```

| Panel | What it shows | Experiment | Command | Script / output |
|---|---|---|---|---|
| 2a | Frozen-test aggregate predictability $R^2$ versus control-success probability $p_{\mathrm{ctrl}}$ (15 published + 119 mixed) | E1 + readout | block above | `make_fig2.py`, `make_fig2a.py` → `figures/out/Fig2.pdf`, `Fig2a.pdf` |
| 2b | Unexplained response fraction $1-R^2$ versus $N$ with $a/(N+a)$ fits | E1 + readout | block above | `make_fig2.py` → `Fig2.pdf` |
| 2c | Conditional coefficient of variation versus $N_{\mathrm{eff}}$, coupling preserved or disrupted | E1 | block above | `make_fig2.py` → `Fig2.pdf` |
| 2d | Frozen-test $R^2$ versus physical population size | E1 + readout | block above | `make_fig2.py` → `Fig2.pdf` |
| 2e | Cumulative distribution of $N_{95}$ | E1 + readout | block above | `make_fig2.py` → `Fig2.pdf` |
| 2f | $p_{\mathrm{ctrl}}$ versus $N$ with logistic fits | E1 | block above | `make_fig2.py` → `Fig2.pdf` |
| 2g | $p_{\mathrm{ctrl}}$ versus effective margin $\Gamma$ | E1 | block above | `make_fig2.py` → `Fig2.pdf` |
| 2h | Prediction and control threshold distributions, with censoring | E1 + readout | block above | `make_fig2.py` → `Fig2.pdf` |

### Figure 3 · Systematic response structure limits reliable control despite population averaging

```bash
$RUN --protocol $CFG/response_mechanisms_and_capacity_concentration.yaml --experiments response_mechanisms capacity_concentration
$RUN --protocol $CFG/E2_controller_synchronization.yaml --experiments E2
$RUN --protocol $CFG/phase_coherence.yaml --experiments phase_coherence
python figures/make_fig3.py
```

| Panel | What it shows | Experiment | Command | Script / output |
|---|---|---|---|---|
| 3a | $N^{*}_{\mathrm{eff}}$ across six local response mechanisms | response mechanisms | `$RUN --protocol $CFG/response_mechanisms_and_capacity_concentration.yaml --experiments response_mechanisms` | `make_fig3.py` → `figures/out/Fig3.pdf` |
| 3b | Fitted scaling exponent $\beta$ by mechanism | response mechanisms | as 3a | `make_fig3.py` → `Fig3.pdf` |
| 3c | Median $R^2$ for four broadcast waveforms × fixed, uniform and lognormal delays | E2 | `$RUN --protocol $CFG/E2_controller_synchronization.yaml --experiments E2` | `make_fig3.py` → `Fig3.pdf` |
| 3d | $p_{\mathrm{ctrl}}$ versus controller homogeneity $c$ | E2 | as 3c | `make_fig3.py` → `Fig3.pdf` |
| 3e | NRMSE versus excess synchronization $X_{\mathrm{sync}}$ | E2 | as 3c | `make_fig3.py` → `Fig3.pdf` |
| 3f | Factor-wise $X_{\mathrm{sync}}$ ranges | E2 | as 3c | `make_fig3.py` → `Fig3.pdf` |
| 3g | Phase coherence $H_{\mathrm{phase}}$ under two surrogates | phase coherence | `$RUN --protocol $CFG/phase_coherence.yaml --experiments phase_coherence` | `make_fig3.py` → `Fig3.pdf` |
| 3h | Response fraction and $p_{\mathrm{ctrl}}$ at the largest $N$ | response mechanisms | as 3a | `make_fig3.py` → `Fig3.pdf` |

### Figure 4 · Population evolution shifts effective scale and aggregate-model validity

```bash
$RUN --protocol $CFG/structured_availability.yaml --experiments structured_availability
$RUN --protocol $CFG/E4_controller_drift.yaml --experiments E4
python figures/make_fig4.py                                # also reads the E1 boundary of Fig. 2
```

| Panel | What it shows | Experiment | Command | Script / output |
|---|---|---|---|---|
| 4a | $N_{\mathrm{eff}}$ versus participation $p$ for five availability patterns | structured availability | `$RUN --protocol $CFG/structured_availability.yaml --experiments structured_availability` | `make_fig4.py` → `figures/out/Fig4.pdf` |
| 4b | Tracking NRMSE split into deterministic shortfall and fluctuation | structured availability | as 4a | `make_fig4.py` → `Fig4.pdf` |
| 4c | NRMSE versus effective margin $\Gamma$ using the Result 1 boundary without refitting | structured availability + E1 | as 4a and Fig. 2 | `make_fig4.py` → `Fig4.pdf` |
| 4d | Window NRMSE before and after drift for frozen, aggregate-only and full-information models | E4 | `$RUN --protocol $CFG/E4_controller_drift.yaml --experiments E4` | `make_fig4.py` → `Fig4.pdf` |
| 4e | Fraction of drift events not yet detected | E4 | as 4d | `make_fig4.py` → `Fig4.pdf` |
| 4f | Recovery time | E4 | as 4d | `make_fig4.py` → `Fig4.pdf` |
| 4g | Recovered share $G$ versus cumulative aggregate-only uplink | E4 | as 4d | `make_fig4.py` → `Fig4.pdf` |
| 4h | Cumulative excess loss | E4 | as 4d | `make_fig4.py` → `Fig4.pdf` |

### Figure 5 · Network constraints limit the physical deliverability of aggregate flexibility

Run the data, network-input and network stages of the full chain (they include every module below, in dependency order), then draw the figure:

```bash
bash scripts/run_reproduction.sh --data
bash scripts/run_reproduction.sh --network-inputs
bash scripts/run_reproduction.sh --network
python -m figures.make_fig5                                # seven panels + composite Fig5.png / Fig5.pdf
```

Methods A-I of panel a: **A** no coordination, **B** local SOC rules, **C** MPC, **D** mean-field control, **E** virtual battery, **F** packetized energy management, **G** transactive control, **H** EPS, **I** centralized greedy reference.

| Panel | What it shows | Experiment | Command (after `--network-inputs`) | Script / output |
|---|---|---|---|---|
| 5a | Network acceptance of methods A-I under matched operation, deep-feeder loading, dual bottlenecks, derating error and three placements (M0-M6) | IEEE-69 network implementation | `python -m $NX.ieee69_network_implementation --seed-count 30 --bootstrap-draws 4000`<br>`python -m $NX.ieee69_trained_vs_transferred` | `make_fig5.py` → `figures/out/Fig5a_network_acceptance.pdf` |
| 5b | Curtailment reduction relative to the centralized greedy reference, 50% feeder and 80% distal-node concentration, IEEE-33/69/123 | feeder comparison | as 5a, plus `python -m $NX.ieee33_trained_model --seed-count 30`<br>`python -m $NX.ieee123_trained_model`<br>`python -m $NX.feeder_comparison_pairwise --workers 2 --seed-count 30`<br>`python -m $NX.feeder_comparison_references --workers 2 --seed-count 30` | `make_fig5.py` → `Fig5b_spatial_retention.pdf` |
| 5c | Curtailment reduction, $R^2$ and acceptance as branch, transformer and voltage limits vary | constraint sweep | `python -m $NX.constraint_sweep --workers 6 --seed-count 30`<br>`python -m $NX.constraint_sweep_transformer`<br>`python -m $NX.constraint_sweep_transformer_summary` | `make_fig5.py` → `Fig5c_constraint_sweep.pdf` |
| 5d | Dataset-level aggregate-response $R^2$ on IEEE-33, IEEE-69 and IEEE-123 | feeder comparison | as 5b | `make_fig5.py` → `Fig5d_response_fidelity.pdf` |
| 5e | Dataset-level curtailment reduction on the three feeders | feeder comparison | as 5b | `make_fig5.py` → `Fig5e_curtailment_by_feeder.pdf` |
| 5f | Curtailment reduction under regional mixing, type zoning, spatial concentration, type-density imbalance and network coupling | spatial heterogeneity | `python -m $NX.spatial_heterogeneity`<br>`python -m $NX.spatial_heterogeneity_trained_model` | `make_fig5.py` → `Fig5f_spatial_heterogeneity.pdf` |
| 5g | Curtailment reduction under uniform, distal-feeder, distal-node and feeder concentration with reduced line capacity (IEEE-123) | IEEE-123 safety audit | `python -m $NX.ieee123_trained_model`<br>`python -m $NX.ieee123_safety_audit --model-source results/ieee123_safety_audit/trained_eps_ieee123_direct/models --workers 2 --seed-count 30 --bootstrap-draws 4000` | `make_fig5.py` → `Fig5g_ieee123_deployment.pdf`, composite `Fig5.pdf` |

### Supplementary Figures S1-S23 (population experiments)

All are drawn by one call, after the experiments listed in the table have been run and copied to `figures/data/` ([how](figures/README.md#figure-inputs)):

```bash
python figures/make_supplementary_figures.py               # writes figures/out/supplementary/FigS1_... to FigS23_...
```

<details>
<summary><b>Supplementary Figs. S1-S23: figure → experiment → command → output</b> (click to expand)</summary>

| Fig. | What it shows | Experiment | Command | Output in `figures/out/supplementary/` |
|---|---|---|---|---|
| | **Note 1 · Empirical populations and constructed mixed populations** | | | |
| S1 | Empirical populations span the population-size range used in the scale experiments | E1 (published + mixed) + readout | as Fig. 2 | `FigS1_dataset_inventory` |
| S2 | Composition of the 119 constructed mixed populations | E1 mixed | `bash scripts/run_mix_batches.sh 110 0 0` | `FigS2_mixture_composition` |
| | **Note 2 · Frozen prediction and statistical-estimator validation** | | | |
| S3 | Frozen evaluation avoids small-population optimism in aggregate predictability | E1 + readout | as Fig. 2 | `FigS3_insample_optimism` |
| S4 | Population groups and coupling arms | E1 (published + mixed) | as Fig. 2 | `FigS4_beta_exponent` |
| S5 | Random-delay responses destabilize the conditional scaling estimate | response mechanisms | `$RUN --protocol $CFG/response_mechanisms_and_capacity_concentration.yaml --experiments response_mechanisms` | `FigS5_cv_collapse_by_mechanism` |
| | **Note 3 · Effective-population coordinates and their limits** | | | |
| S6 | Empirical dependence can substantially reduce the effective fraction of a physical population | E1 (published + mixed) | as Fig. 2 | `FigS6_effective_fraction` |
| S7 | Effective-margin normalization does not reduce all forms of between-population variation | E1 mixed | `bash scripts/run_mix_batches.sh 110 0 0` | `FigS7_mixture_within_bin_variance` |
| | **Note 4 · Population-specific control transitions across the published populations** | | | |
| S8 | Population-specific control transitions across the 15 published populations | E1 | `$RUN --protocol $CFG/E1_population_scale.yaml --experiments E1` | `FigS8_pctrl_by_population` |
| | **Note 5 · Capacity concentration and the limits of correlation-based effective population size** | | | |
| S9 | Capacity concentration introduces effective-scale variation not captured by correlation alone | capacity concentration | `$RUN --protocol $CFG/response_mechanisms_and_capacity_concentration.yaml --experiments capacity_concentration` | `FigS9_capacity_concentration` |
| | **Note 6 · Temporal compatibility and local response mechanisms** | | | |
| S10 | Temporal mismatch persists as population size increases | E2 | `$RUN --protocol $CFG/E2_controller_synchronization.yaml --experiments E2` | `FigS10_waveform_gap` |
| S11 | Frequent device response does not imply reliable aggregate control | response mechanisms | as S5 | `FigS11_response_fraction` |
| | **Note 7 · Synchronization diagnostics under common broadcast forcing** | | | |
| S12 | Excess synchronization is more sensitive to broadcast waveform than to empirical coupling | E2 | as S10 | `FigS12_xsync_by_arm_and_waveform` |
| S13 | Broadcast conditioning attenuates most apparent factor effects on phase coherence | phase coherence | `$RUN --protocol $CFG/phase_coherence.yaml --experiments phase_coherence` | `FigS13_raw_vs_conditioned` |
| S14 | Residual phase-locking episodes are weak and short-lived under the tested conditions | phase coherence | as S13 | `FigS14_phase_window_metrics` |
| S15 | Control degradation generally precedes detectable excess synchronization | E2 | as S10 | `FigS15_fail_before_sync` |
| | **Note 8 · Structured availability and the time-varying effective population** | | | |
| S16 | Availability structure has little effect on directional and reserve outcomes beyond aggregate fluctuation | structured availability | `$RUN --protocol $CFG/structured_availability.yaml --experiments structured_availability` | `FigS16_availability_outcomes` |
| S17 | Correlated availability can sharply reduce effective population size | second availability family | `$RUN --protocol $CFG/availability_second_family.yaml --experiments availability_second_family` | `FigS17_availability_second_family` |
| S18 | Effective population size tracks the number of independent availability schedules across mixed populations | second availability family (published + mixed) | as S17, plus `$RUN --protocol $CFG/availability_second_family_mixed.yaml --experiments availability_second_family` | `FigS18_availability_second_family_mixed` |
| | **Note 9 · Long-horizon state evolution under repeated dispatch** | | | |
| S19 | Long-horizon closed-loop state evolution under repeated dispatch | long-horizon state | `$RUN --protocol $CFG/long_horizon_state.yaml --experiments long_horizon_state` | `FigS19_long_horizon_state` |
| | **Note 10 · Drift detection and aggregate-only recalibration** | | | |
| S20 | A drift-free calibration stream yields a low pre-drift false-alarm rate | E4 | `$RUN --protocol $CFG/E4_controller_drift.yaml --experiments E4` | `FigS20_detector_calibration` |
| S21 | Aggregate-only recalibration achieves high endpoint recovery with bounded uplink | E4 | as S20 | `FigS21_recovery_cost` |
| S22 | Drift detection and aggregate-only recovery replicate across constructed mixed populations | E4 (published + mixed) | as S20, plus `$RUN --protocol $CFG/E4_controller_drift_mixed.yaml --experiments E4` | `FigS22_drift_on_mixed_populations` |
| | **Note 11 · Population-scale control and distribution-network topology** | | | |
| S23 | The aggregate-control threshold remains stable across the tested feeder topologies | E1 mixed on the reference feeder and IEEE-33/69/123 | feeder calibration, then `bash scripts/run_topology_mix.sh` ([details](figures/README.md#supplementary-fig-s23-feeder-topologies)) | `FigS23_feeder_topology` |

</details>

### Supplementary Figures S24-S41 (network experiments)

All are drawn by one call after the `--network` stage:

```bash
python -m figures.make_supplementary_network_figures      # writes figures/out/FigS24_... to FigS41_...
```

<details>
<summary><b>Supplementary Figs. S24-S41: figure → experiment → command → output</b> (click to expand)</summary>

| Fig. | What it shows | Experiment | Command (after `--network-inputs`) | Output in `figures/out/` |
|---|---|---|---|---|
| | **Note 12 · Transferability across datasets and operating domains** | | | |
| S24 | Transfer performance across heterogeneous target datasets | transfer | `python -m $NX.train_pooled_model`<br>`python -m $NX.transfer_across_datasets` | `FigS24_transfer_heatmap` |
| S25 | Transfer robustness under alternative dataset splits | transfer | `python -m $NX.transfer_alternative_splits` | `FigS25_transfer_alternative_splits` |
| S26 | Distribution of predictive performance across physical population scales | transfer | as S24 | `FigS26_transfer_r2_by_scale` |
| | **Note 13 · Effective population size and the emergence of predictable aggregate response** | | | |
| S27 | Normalized effective scale and physical population size jointly explain aggregate predictability | effective scale, published | `python -m $NX.effective_scale_published` | `FigS27_effective_scale_published` |
| S28 | Composition of heterogeneous mixed populations used for effective-scale analysis | effective scale, mixed | `python -m $NX.train_mixed_model`<br>`python -m $NX.effective_scale_mixed` | `FigS28_mixed_composition` |
| S29 | Scaling behaviour of mixed populations in physical and normalized coordinates | effective scale, mixed | as S28 | `FigS29_mixed_scaling` |
| S30 | Distribution of predictability thresholds across pairwise population mixtures | pairwise predictability | `python -m $NX.pairwise_predictability` | `FigS30_pairwise_thresholds` |
| S31 | Population-scale convergence across heterogeneous pairwise mixtures | pairwise predictability | as S30 | `FigS31_pairwise_r2_by_scale` |
| S32 | Distribution of aggregate predictability across Gamma regimes | effective scale, published + mixed | as S27 and S28 | `FigS32_r2_by_gamma` |
| | **Note 14 · Algorithm robustness and network implementation** | | | |
| S33 | Temporal and spatial distribution of feeder constraint activation | IEEE-69 network implementation | `python -m $NX.ieee69_network_implementation --seed-count 30 --bootstrap-draws 4000` | `FigS33_ieee69_constraint_activation` |
| S34 | Performance preservation under trained and transferred network conditions | IEEE-69 trained vs transferred | `python -m $NX.ieee69_trained_vs_transferred` | `FigS34_trained_vs_transferred` |
| S35 | Joint evaluation of algorithmic effectiveness and response fidelity (IEEE-69) | IEEE-69 network implementation | as S33 | `FigS35_ieee69_effect_and_fidelity` |
| | **Note 15 · Physical feasibility boundaries under network stress** | | | |
| S36 | Physical violation probability under increasing operating stress | network stress boundary | `python -m $NX.network_stress_boundary --seed-count 30 --bootstrap-draws 2000` | `FigS36_physical_violation` |
| S37 | Algorithm robustness under line-capacity degradation | network stress boundary | as S36 | `FigS37_line_derating` |
| S38 | Performance degradation under increasing request intensity | network stress boundary | as S36 | `FigS38_request_intensity` |
| S39 | Impact of spatial concentration on network-deliverable flexibility | network stress boundary | as S36 | `FigS39_spatial_concentration` |
| | **Note 16 · Safety auditing under realistic feeder constraints** | | | |
| S40 | Safety audit of network acceptance and constraint-induced risk (IEEE-123) | IEEE-123 safety audit | `python -m $NX.ieee123_trained_model`<br>`python -m $NX.ieee123_safety_audit --model-source results/ieee123_safety_audit/trained_eps_ieee123_direct/models --workers 2 --seed-count 30 --bootstrap-draws 4000` | `FigS40_ieee123_safety_audit` |
| | **Note 17 · Component-level network safety across IEEE-69 stress conditions** | | | |
| S41 | Component-level IEEE-69 audit across 17 dataset configurations and 30 paired seeds: branch loading, minimum voltage, transformer loading and safety-layer gain of ten methods, including "E20 pooled EPS" and "E21 mixed EPS" | IEEE-69 network implementation | as S33 | `FigS41_ieee69_component_audit` |

</details>

`scripts/run_network_experiments.sh` re-runs the three audit experiments alone (`--ieee69-implementation`, `--stress-boundary`, `--ieee123-audit` or `--all`).

### Supplementary Tables

| Table | Title | Source |
|---|---|---|
| S1 | Empirical population characteristics, standardized variables and data features | compiled from the dataset documentation; the adapter parameters behind it are in [`preprocessing_config.json`](derived_data/configuration/preprocessing_config.json) |
| S2 | Mixed-Fleet Data Construction | compiled; the compositions are fixed in [`scenario_config.json`](derived_data/configuration/scenario_config.json), [`pairwise_seed_config.json`](derived_data/configuration/pairwise_seed_config.json) and [`constants.py`](src/extra/dataset_combinations/constants.py) |
| S3 | Experimental design space and simulation coverage | **generated**: `python figures/tables/make_supplementary_tables.py` → `figures/tables/Supplementary_Tables_S3_S4.docx` |
| S4 | Population evolution and perturbation protocols | **generated** by the same script |
| S5 | Control strategies, required inputs, complexity and limitations | compiled from the controller implementations in [`reference_controllers.py`](src/extra/ieee33_device_day_simulation/network_experiments/reference_controllers.py) and [`network_dispatch_protocol.py`](src/extra/ieee33_device_day_simulation/network_experiments/network_dispatch_protocol.py) |
| S6 | Physical variables, mathematical limits and questions | compiled from the Methods (analytical table, no code) |
| S7 | Environment parameters for the constrained-dispatch simulations | compiled from the literature; the implemented values are in [`configs/reproducibility.json`](configs/reproducibility.json) and [`src/extra/ieee33_device_day_simulation/configs/`](src/extra/ieee33_device_day_simulation/configs) |

### Full chain

[`scripts/run_reproduction.sh`](scripts/run_reproduction.sh) runs the raw-data-to-figure chain of Fig. 5 and Supplementary Figs. S24-S41, stage by stage or all at once. Logs are written to `logs/reproduction/<step>.log`.

| Stage | What it runs |
|---|---|
| `--check` | derived-data check, English-documentation check, byte-compilation of all code |
| `--data` | preprocessing of the downloaded datasets, generation of the 119 mixed-population fleets, per-dataset configurations |
| `--population` | every population experiment of Figs. 2-4 and S1-S22 on the 15 published populations, each with its own configuration from the [experiment index](#experiment-index) (the mixed-population batches and the feeder-topology arms are launched separately, see below) |
| `--network-inputs` | E1 on the extended scale grid, per-dataset feeder experiments and network baselines, the shared dispatch protocol |
| `--network` | all 22 network modules in dependency order (transfer, effective scale, IEEE-69, IEEE-33, IEEE-123, constraint sweeps, stress boundary, spatial heterogeneity, feeder comparison) |
| `--figures` | `figures.make_fig5` and `figures.make_supplementary_network_figures` |
| `--all` | every stage above, in order |

```bash
bash download_data.sh
bash scripts/run_reproduction.sh --all
```

The population-scale figures (Figs. 2-4, S1-S23) use the per-experiment configurations of the [experiment index](#experiment-index); [`scripts/run_main_experiments.sh`](scripts/run_main_experiments.sh) launches E1, E2, phase coherence and E4 in the background, [`scripts/run_supp.sh`](scripts/run_supp.sh) launches the long-horizon, response-mechanism and capacity-concentration experiments, [`scripts/mix_campaign.sh`](scripts/mix_campaign.sh) runs the mixed-population variants of the second availability family and E4, and [`scripts/run_all.sh`](scripts/run_all.sh) runs `download_data.sh`, prepares the dataset configurations and then runs, in the foreground, the same eight population experiments as the `--population` stage.

### Runtime and hardware

The reported runs used Python 3.10 on Ubuntu 22.04 with a 144-core CPU and 976 GB of RAM; no GPU is needed. Every runner parallelizes over populations (`execution.dataset_workers` in each configuration, `--workers` for network modules) and checkpoints per population or task, so an interrupted run resumes with the same command.

| Task | Typical wall time on the reported machine |
|---|---|
| Unit tests, derived-data verification | about a minute |
| Smoke suite (`scripts/run_smoke.sh`) | minutes |
| Download and preprocessing of the 15 datasets | hours, dominated by download speed |
| One population-scale experiment on the 15 published populations | hours |
| E1 on the 119 mixed populations, all 30 composition seeds | days |
| Network stages (`--network-inputs`, `--network`) | days |
| Any figure script, given its inputs | seconds to minutes |

Set `dataset_workers` and `--workers` to the available cores; memory scales with the number of workers.

## Configurations, seeds and data splits

Everything that determines a result is stored in version-controlled files: population-experiment protocols in [`src/extra/population_experiments/configs/`](src/extra/population_experiments/configs), feeder, device and control settings in [`src/extra/ieee33_device_day_simulation/configs/`](src/extra/ieee33_device_day_simulation/configs), and the network-half protocol in [`configs/reproducibility.json`](configs/reproducibility.json). Each run also copies its resolved protocol to `results/<configuration>/config/`.

| Item | Value | Where |
|---|---|---|
| Population-experiment seed | `random_seed: 20260720` | every file in `src/extra/population_experiments/configs/` |
| E1 split per population-scale cell | 48 operating conditions; 10 calibration and 120 held-out test repetitions; frozen prediction evaluated on every held-out realization separately | `e1` block of [`E1_population_scale.yaml`](src/extra/population_experiments/configs/E1_population_scale.yaml) |
| E1 population sizes | 50, 100, 150, 250, 400, 650, 1000, 1600, 2000, 2500, 3000 plus the largest unique fleet of each source; the extended grid adds 1, 5, 15, 30 | same file, `e1.n_candidates` |
| E2 repetitions | 10 calibration and 24 test repetitions per condition; four waveforms, three delay distributions, homogeneity 0 to 1 | `e2` block of [`E2_controller_synchronization.yaml`](src/extra/population_experiments/configs/E2_controller_synchronization.yaml) |
| E4 drift protocol | affected fractions 0.2, 0.5, 0.8; abrupt and gradual drift; 50 conditions per aggregate-sampling window | `e4` block of [`E4_controller_drift.yaml`](src/extra/population_experiments/configs/E4_controller_drift.yaml) |
| Criteria | $R^2 \ge 0.95$ for prediction; NRMSE $\le 0.10$, sign consistency $\ge 0.95$ and $p_{\mathrm{ctrl}} \ge 0.90$ for control | control criteria: `common` block of each configuration; prediction criterion: `--threshold` (default 0.95) of `tools/analysis/predictability_readout.py` |
| Profile partition seed of the mixed populations | 20260714 (train / validation / test partition of device profiles) | [`generation_config.json`](derived_data/configuration/generation_config.json) |
| Test seeds of the mixed populations | 30 per population and sampling mode; scenario seeds listed in [`scenario_seed_index.csv`](derived_data/index/scenario_seed_index.csv) | [`scenario_config.json`](derived_data/configuration/scenario_config.json) |
| Pairwise seed | `20260808 + pair_index * 100000 + seed_index` | [`pairwise_index.csv`](derived_data/index/pairwise_index.csv) |
| Network simulation seed and paired seeds | global seed 20260714; 30 paired seeds shared by all methods within a condition | [`configs/reproducibility.json`](configs/reproducibility.json) |
| Aggregate-response estimator | 70% fit / 30% conformal calibration, response-sign stratified, NN validation 10% with seed 42; calibration data never used for weight fitting | same file, `data_split.estimator` |
| Network scenarios | M0-M6 (IEEE-69), request, line-pressure and spatial-concentration sweeps, four IEEE-123 placements | same file, `scenarios` |
| Other fixed seeds | effective-scale bootstrap 20260807, pairwise predictability 20260809 | module defaults of `effective_scale_published` and `pairwise_predictability` |

Minimal validation cases for the population experiments (all except E4) are in [`src/extra/population_experiments/configs/smoke/`](src/extra/population_experiments/configs/smoke) and are run by [`scripts/run_smoke.sh`](scripts/run_smoke.sh).

## Tests

```bash
python -m pytest tests -q -o addopts=""
```

The suite (356 tests) covers the broadcast encoder, decoder and validator, the battery and constraint model, the state machine, the aggregate-response estimator, the simulator and scenarios, the statistics, the canonical dataset adapter, the constrained feeder experiment, the dispatch scale of the IEEE-33 simulation and the population metrics (effective population size, directional and network metrics, Wilson intervals, scale grid and controller homogeneity). The mixed-population generator has its own self-test:

```bash
python -m src.extra.dataset_combinations.self_test --data-root data
```

## Citation

If you use this code or the constructed mixed populations, please cite the paper:

```bibtex
@article{population_scale_dispatch_2026,
  title   = {Population scale enables reliable dispatch of distributed energy storage without real-time device-level sensing},
  journal = {Nature Communications},
  year    = {2026},
  note    = {Manuscript}
}
```

The author list, volume and DOI will be added here on publication. Please also cite the providers of the public datasets listed in [Data](#data).

## License

The code and the constructed mixed-population data in this repository are released under the [MIT License](LICENSE). The public source datasets remain under the licences of their providers.

## Contact

Questions, bug reports and reproduction problems are welcome as [GitHub issues](https://github.com/BaiCubed/-broadcast-coordination/issues). Please include the command, the configuration file and the tail of the corresponding log in `logs/`.
