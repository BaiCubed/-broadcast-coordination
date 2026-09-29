# Figures, Source Data and tables

The scripts in this directory turn experiment outputs into the figures, the Source Data tables and the generated supplementary tables of the paper. They run no simulation. Figure 1 of the paper is a schematic and is not produced here.

Run every command from the repository root with `PYTHONPATH` set to it:

```bash
export PYTHONPATH="$PWD"
```

## Scripts

| Script | Paper items | Reads | Writes |
|---|---|---|---|
| [`make_fig2.py`](make_fig2.py) | Fig. 2a-h | `figures/data/` | `figures/out/Fig2.pdf`, `Fig2.png`, `Fig2_stats.json` |
| [`make_fig2a.py`](make_fig2a.py) | Fig. 2a as a standalone transition map | `figures/data/` | `figures/out/Fig2a.pdf`, `Fig2a.png`, `Fig2a_stats.json`, `figures/out/source_data/Fig2a_transition_map_by_resource_class.csv` |
| [`make_fig3.py`](make_fig3.py) | Fig. 3a-h | `figures/data/` | `figures/out/Fig3.pdf`, `Fig3.png`, `Fig3_stats.json` |
| [`make_fig4.py`](make_fig4.py) | Fig. 4a-h | `figures/data/` | `figures/out/Fig4.pdf`, `Fig4.png`, `Fig4_stats.json` |
| [`make_fig5.py`](make_fig5.py) | Fig. 5a-g | `results/` | `figures/out/Fig5a_network_acceptance` ... `Fig5g_ieee123_deployment` (PDF and PNG), composite `figures/out/Fig5.png` and `Fig5.pdf`, panel tables in `figures/out/source_data/` |
| [`make_supplementary_figures.py`](make_supplementary_figures.py) | Supplementary Figs. S1-S23 | `figures/data/` | `figures/out/supplementary/FigS1_...` to `FigS23_...` (PDF and PNG), `supplementary_stats.json` |
| [`make_supplementary_network_figures.py`](make_supplementary_network_figures.py) | Supplementary Figs. S24-S41 | `results/` | `figures/out/FigS24_...` to `FigS41_...` (PNG and PDF) |
| [`make_source_data.py`](make_source_data.py) | Source Data tables of Figs. 2-4 | `figures/data/` | `figures/out/source_data/Fig2*.csv`, `Fig3*.csv`, `Fig4*.csv` |
| [`tables/make_supplementary_tables.py`](tables/make_supplementary_tables.py) | Supplementary Tables S3 and S4 | nothing (the table content is in the script) | `figures/tables/Supplementary_Tables_S3_S4.docx` (needs `python-docx`) |

```bash
python figures/make_fig2.py
python figures/make_fig2a.py
python figures/make_fig3.py
python figures/make_fig4.py
python figures/make_supplementary_figures.py
python figures/make_source_data.py
python -m figures.make_fig5
python -m figures.make_supplementary_network_figures
python figures/tables/make_supplementary_tables.py
```

Every population-figure script also writes a `*_stats.json` holding the numbers shown in its panels, so a value quoted in the text can be checked without opening the PDF.

## Supplementary figure files

| Fig. | File stem | Fig. | File stem |
|---|---|---|---|
| S1 | `supplementary/FigS1_dataset_inventory` | S22 | `supplementary/FigS22_drift_on_mixed_populations` |
| S2 | `supplementary/FigS2_mixture_composition` | S23 | `supplementary/FigS23_feeder_topology` |
| S3 | `supplementary/FigS3_insample_optimism` | S24 | `FigS24_transfer_heatmap` |
| S4 | `supplementary/FigS4_beta_exponent` | S25 | `FigS25_transfer_alternative_splits` |
| S5 | `supplementary/FigS5_cv_collapse_by_mechanism` | S26 | `FigS26_transfer_r2_by_scale` |
| S6 | `supplementary/FigS6_effective_fraction` | S27 | `FigS27_effective_scale_published` |
| S7 | `supplementary/FigS7_mixture_within_bin_variance` | S28 | `FigS28_mixed_composition` |
| S8 | `supplementary/FigS8_pctrl_by_population` | S29 | `FigS29_mixed_scaling` |
| S9 | `supplementary/FigS9_capacity_concentration` | S30 | `FigS30_pairwise_thresholds` |
| S10 | `supplementary/FigS10_waveform_gap` | S31 | `FigS31_pairwise_r2_by_scale` |
| S11 | `supplementary/FigS11_response_fraction` | S32 | `FigS32_r2_by_gamma` |
| S12 | `supplementary/FigS12_xsync_by_arm_and_waveform` | S33 | `FigS33_ieee69_constraint_activation` |
| S13 | `supplementary/FigS13_raw_vs_conditioned` | S34 | `FigS34_trained_vs_transferred` |
| S14 | `supplementary/FigS14_phase_window_metrics` | S35 | `FigS35_ieee69_effect_and_fidelity` |
| S15 | `supplementary/FigS15_fail_before_sync` | S36 | `FigS36_physical_violation` |
| S16 | `supplementary/FigS16_availability_outcomes` | S37 | `FigS37_line_derating` |
| S17 | `supplementary/FigS17_availability_second_family` | S38 | `FigS38_request_intensity` |
| S18 | `supplementary/FigS18_availability_second_family_mixed` | S39 | `FigS39_spatial_concentration` |
| S19 | `supplementary/FigS19_long_horizon_state` | S40 | `FigS40_ieee123_safety_audit` |
| S20 | `supplementary/FigS20_detector_calibration` | S41 | `FigS41_ieee69_component_audit` |
| S21 | `supplementary/FigS21_recovery_cost` | | |

All stems are relative to `figures/out/`. S2 and S7 are drawn by one function (`figS2_S7`) and saved as two files.

## Figure inputs

The population-figure scripts (`make_fig2.py`, `make_fig2a.py`, `make_fig3.py`, `make_fig4.py`, `make_supplementary_figures.py`, `make_source_data.py`) read `figures/data/<name>/`. Each directory is a copy of one runner output directory:

| `figures/data/<name>` | Experiment | Copy from |
|---|---|---|
| `E1_population_scale` | E1, 15 published populations | `results/E1_population_scale/E1_population_scale` |
| `E1_population_scale_mixed` | E1, 119 mixed populations, composition seed 0 | `results/E1_population_scale_mixed_s00/E1_population_scale` |
| `predictability_readout` | frozen-test $R^2$ and $N_{95}$ readout of E1 | written directly by `tools/analysis/predictability_readout.py` (below) |
| `predictability_readout_mixed` | the same readout for the mixed populations | written directly by `tools/analysis/predictability_readout.py` (below) |
| `E2_controller_synchronization` | E2 | `results/E2_controller_synchronization/E2_controller_synchronization` |
| `phase_coherence` | phase coherence | `results/phase_coherence/phase_coherence` |
| `response_mechanisms` | local response mechanisms | `results/response_mechanisms_and_capacity_concentration/response_mechanisms` |
| `capacity_concentration` | capacity concentration | `results/response_mechanisms_and_capacity_concentration/capacity_concentration` |
| `structured_availability` | structured availability | `results/structured_availability/structured_availability` |
| `availability_second_family` | second availability family, published | `results/availability_second_family/availability_second_family` |
| `availability_second_family_mixed` | second availability family, mixed | `results/availability_second_family_mixed/availability_second_family` |
| `long_horizon_state` | long-horizon state evolution | `results/long_horizon_state/long_horizon_state` |
| `E4_controller_drift` | E4 | `results/E4_controller_drift/E4_controller_drift` |
| `E4_controller_drift_mixed` | E4 on the mixed populations | `results/E4_controller_drift_mixed/E4_controller_drift` |
| `E1_feeder_topology/{ieee33,ieee69,ieee123}` | E1 mixed under the IEEE feeders | `results/E1_feeder_topology/<feeder>/E1_population_scale` |
| `E1_feeder_topology/published` | E1 mixed on the reference feeder | `results/E1_population_scale_mixed_s00/E1_population_scale` |
| `dataset_metadata.csv` | population inventory, one row per published population: `dataset, short, resource_type, unique_sources, unique_sources_note` | shipped in this repository (the population inventory of Supplementary Table S1) |
| `E1_population_scale_mixed/mixture_composition.csv` | nominal source weights of each mixed population: `combo, source, weight` (335 rows for the 119 populations) | shipped in this repository (derived from `derived_data/configuration/scenario_config.json` and `derived_data/index/pairwise_index.csv`) |

```bash
mkdir -p figures/data/E1_feeder_topology
cp -r results/E1_population_scale/E1_population_scale                                 figures/data/E1_population_scale
cp -r results/E1_population_scale_mixed_s00/E1_population_scale/.                     figures/data/E1_population_scale_mixed/   # directory already holds the shipped mixture_composition.csv
cp -r results/E2_controller_synchronization/E2_controller_synchronization             figures/data/E2_controller_synchronization
cp -r results/phase_coherence/phase_coherence                                         figures/data/phase_coherence
cp -r results/response_mechanisms_and_capacity_concentration/response_mechanisms      figures/data/response_mechanisms
cp -r results/response_mechanisms_and_capacity_concentration/capacity_concentration   figures/data/capacity_concentration
cp -r results/structured_availability/structured_availability                         figures/data/structured_availability
cp -r results/availability_second_family/availability_second_family                   figures/data/availability_second_family
cp -r results/availability_second_family_mixed/availability_second_family             figures/data/availability_second_family_mixed
cp -r results/long_horizon_state/long_horizon_state                                   figures/data/long_horizon_state
cp -r results/E4_controller_drift/E4_controller_drift                                 figures/data/E4_controller_drift
cp -r results/E4_controller_drift_mixed/E4_controller_drift                           figures/data/E4_controller_drift_mixed
for f in ieee33 ieee69 ieee123; do
  cp -r results/E1_feeder_topology/$f/E1_population_scale figures/data/E1_feeder_topology/$f
done
cp -r results/E1_population_scale_mixed_s00/E1_population_scale figures/data/E1_feeder_topology/published

python tools/analysis/predictability_readout.py \
  --source results/E1_population_scale/E1_population_scale --output figures/data/predictability_readout
python tools/analysis/predictability_readout.py \
  --source results/E1_population_scale_mixed_s00/E1_population_scale --output figures/data/predictability_readout_mixed
```

Which script needs which input:

| Script | `figures/data/` inputs |
|---|---|
| `make_fig2.py` | `E1_population_scale`, `E1_population_scale_mixed`, `predictability_readout`, `predictability_readout_mixed`, `dataset_metadata.csv` |
| `make_fig2a.py` | `predictability_readout`, `predictability_readout_mixed` |
| `make_fig3.py` | `response_mechanisms`, `E2_controller_synchronization`, `phase_coherence` |
| `make_fig4.py` | `structured_availability`, `E1_population_scale`, `E4_controller_drift` |
| `make_supplementary_figures.py` | all directories in the table above |
| `make_source_data.py` | `E1_population_scale`, `E1_population_scale_mixed`, `predictability_readout`, `predictability_readout_mixed`, `response_mechanisms`, `E2_controller_synchronization`, `phase_coherence`, `structured_availability`, `E4_controller_drift` |

The network-figure scripts (`make_fig5.py`, `make_supplementary_network_figures.py`) read the network result trees under `results/` directly: `results/ieee69_network_implementation/`, `results/network_stress_boundary/`, `results/ieee123_safety_audit/`, `results/transfer/`, `results/effective_scale_mixtures/` and the per-dataset network baselines `results/<dataset>_ieee33_real_load/network_baseline/`. They are filled by the `--network-inputs` and `--network` stages of `scripts/run_reproduction.sh`; the module behind each panel is listed in the main [README](../README.md#figure-5--network-constraints-limit-the-physical-deliverability-of-aggregate-flexibility).

## Supplementary Fig. S23: feeder topologies

S23 compares the E1 control threshold of the 119 mixed populations on the reference feeder and on the IEEE-33, IEEE-69 and IEEE-123 feeders. The IEEE runs use a per-population capacity calibration that must exist before the runner starts:

```bash
python -c "import yaml; print('\n'.join(yaml.safe_load(open('src/extra/population_experiments/configs/E1_feeder_topology_ieee33.yaml'))['e1']['datasets']))" \
  | xargs -P 16 -I{} python tools/protocols/calibrate_topology.py {}
python tools/protocols/merge_calib.py                      # writes results/_topology_calibration.json
bash scripts/run_topology_mix.sh                           # E1 on IEEE-33, IEEE-69 and IEEE-123
```

`run_topology_mix.sh` regenerates the three protocols with `tools/protocols/make_feeder_topology_protocols.py` and selects the feeder through the environment variables `NC_TOPOLOGY` and `NC_NETWORK_FEEDBACK`.

## Style

| Module | Role |
|---|---|
| [`ncstyle.py`](ncstyle.py) | colours, font sizes and panel helpers of Figs. 2-4 and S1-S23; registers any `*.ttf` in `figures/fonts/` |
| [`nckeys.py`](nckeys.py) | short dataset names and dataset-to-resource-class assignment |
| [`network_style.py`](network_style.py) | style, dataset labels and panel helpers of Fig. 5 and S24-S41 |

Panel geometry is set in each script's `main()` (or `build()`) through `ax_mm(fig, x, y, w, h)`, in millimetres on a 183 mm double-column canvas. Panel functions are named after the printed panel letters (`panel_a` ... `panel_h`; `panel_a_network_acceptance` ... `panel_g_ieee123_deployment` in `make_fig5.py`).

Figures were rendered with Arial. Without it, matplotlib falls back to Liberation Sans, Helvetica and DejaVu Sans; the metric-compatible Liberation Sans reproduces the published layout.

## Supplementary tables

`tables/make_supplementary_tables.py` writes Supplementary Tables S3 (experimental design space and simulation coverage) and S4 (population evolution and perturbation protocols) to `figures/tables/Supplementary_Tables_S3_S4.docx`. Supplementary Tables S1, S2, S5, S6 and S7 are compiled from the dataset documentation, the generator configuration, the controller implementations, the Methods and the literature; the main [README](../README.md#supplementary-tables) points to the files that hold their values.
