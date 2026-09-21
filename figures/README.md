# Figure and table generation

These scripts turn the experiment outputs into the figures, the source data and
the supplementary tables of the paper. They run no simulation: they read the CSV
and JSON files written by the experiment runner and write to `figures/out/`.

Figure 1 of the paper is a schematic of the architecture and the experimental
framework and is not produced here.

## Main figures

| Figure | Script | Output |
|---|---|---|
| Figure 2, population-scale transitions in aggregate predictability and reliable control | `make_fig1.py` | `out/Fig1.pdf`, `out/Fig1.png`, `out/Fig1_stats.json` |
| Figure 2a as a standalone panel, the predictability-controllability transition map | `make_fig_transition.py` | `out/Fig_transition_map.*` |
| Figure 3, systematic response structure limits reliable control despite population averaging | `make_fig2_new.py` | `out/Fig2_New.*` |
| Figure 4, population evolution shifts effective scale and aggregate-model validity | `make_fig3.py` | `out/Fig3.*` |
| Supplementary Fig. S1, the population inventory, standalone rendering | `make_fig_coverage.py` | `out/Fig_dataset_coverage.*` |
| Supplementary Figs. S1-S22 | `make_appendix.py` | `out/appendix/AppFig1..AppFig22` |
| Source data for Figures 2 and 3 | `make_source_data.py` | `out/source_data/*.csv` |
| Source data for Figures 3 and 4 | `make_source_data_new.py` | `out/source_data/*.csv` |

Figure 5, network constraints limit the physical deliverability of aggregate
flexibility, and Supplementary Figs. S23-S40 are produced by the
network-deliverability scripts under
`src/extra/ieee33_device_day_simulation/figures/`, which run their own
simulations; see the main `README.md`.

```bash
python figures/make_fig1.py
python figures/make_fig2_new.py
python figures/make_fig3.py
python figures/make_fig_transition.py
python figures/make_fig_coverage.py
python figures/make_appendix.py
python figures/make_source_data.py
python figures/make_source_data_new.py
```

Every script also writes a `*_stats.json` next to the figure holding the numbers
that appear in the panels, so a quoted value can be checked without opening the
PDF.

Colours and font sizes are defined once in `ncstyle.py`; the dataset to
resource-class assignment is in `nckeys.py`; panel geometry is set in each
script's `main()` through `ax_mm(fig, x, y, w, h)`, in millimetres on a 183 mm
double-column canvas.

The figures were rendered with Arial. If Arial is not installed, matplotlib falls
back to Liberation Sans, Helvetica and then DejaVu Sans; installing the
metric-compatible Liberation Sans family reproduces the published metrics.
`ncstyle.py` also registers every `*.ttf` found in `figures/fonts/`, so a local
font copy can be dropped there.

## Supplementary figures S1-S22

`make_appendix.py` writes the files under `out/appendix/`. The mapping to the
supplementary numbering is:

| Supplementary figure | File written |
|---|---|
| S1, empirical populations span the population-size range | `AppFig2_dataset_inventory` |
| S2, composition of the 119 constructed mixed populations | `AppFig10_mixture_population` (composition panel) |
| S3, frozen evaluation avoids small-population optimism | `AppFig12_insample_optimism` |
| S4, population groups and coupling arms | `AppFig1_beta_exponent` |
| S5, random-delay responses destabilize the conditional scaling estimate | `AppFig3_cv_collapse_by_logic` |
| S6, empirical dependence reduces the effective fraction | `AppFig11_effective_fraction` |
| S7, effective-margin normalization does not remove all between-population variation | `AppFig10_mixture_population` (variance panel) |
| S8, population-specific control transitions | `AppFig22_pctrl_by_fleet` |
| S9, capacity concentration introduces effective-scale variation | `AppFig21_dominant_capacity` |
| S10, temporal mismatch persists as population size increases | `AppFig4_waveform_gap` |
| S11, frequent device response does not imply reliable aggregate control | `AppFig13_response_fraction` |
| S12, excess synchronization is more sensitive to waveform than to coupling | `AppFig5_xsync_by_arm_and_waveform` |
| S13, broadcast conditioning attenuates apparent factor effects | `AppFig6_raw_vs_conditioned` |
| S14, residual phase-locking episodes are weak and short-lived | `AppFig7_phase_window_metrics` |
| S15, control degradation precedes detectable excess synchronization | `AppFig14_fail_before_sync` |
| S16, availability structure has little effect on directional and reserve outcomes | `AppFig8_availability_outcomes` |
| S17, correlated availability sharply reduces effective population size | `AppFig15_absence_structures_second_family` |
| S18, effective population size tracks the number of independent schedules | `AppFig16_absence_structures_mixtures` |
| S19, long-horizon closed-loop state evolution | `AppFig20_long_horizon_soc` |
| S19, a drift-free calibration stream yields a low pre-drift false-alarm rate | `AppFig18_detector_calibration` |
| S20, aggregate-only recalibration achieves high endpoint recovery | `AppFig9_recovery_cost` |
| S21, drift detection and recovery replicate across mixed populations | `AppFig17_drift_on_mixtures` |
| S22, the aggregate-control threshold is stable across feeder topologies | `AppFig19_feeder_topology` |

The supplement currently gives the number S19 to two different figures, the
long-horizon state evolution and the detector calibration; both files are listed
above in the order in which they appear.

## Figure inputs

`figures/data/<name>/` holds the experiment output each script reads. The
directories are copies of the result trees written by the experiment runner. The
`<name>` is the internal run name of the experiment and is kept so that a result
directory can be matched to its configuration:

| `figures/data/` | Experiment | Configuration | Result directory |
|---|---|---|---|
| `e1` | scale boundary, the 15 published populations (Fig. 2) | `e1_full_v2.yaml` | `results/e1_full_v2/E1_scale_boundary_new` |
| `e1_mix` | scale boundary, the 119 mixed populations (Fig. 2) | `e1_mix_s00.yaml` | `results/e1_mix_s00/E1_scale_boundary_new` |
| `e2` | controller synchronization (Fig. 3c-f) | `e2_full_v2.yaml` | `results/e2_full_v2/E2_controller_synchronization_new` |
| `e3` | phase coherence (Fig. 3g) | `e3_full_v2.yaml` | `results/e3_full_v2/E3_phase_coherence_new` |
| `e4` | controller drift and aggregate-only recalibration (Fig. 4d-h) | `e4_v2.yaml` | `results/e4_v2/E4_controller_drift_new` |
| `e4_mix` | the same on the mixed populations (S21) | `e4_mix_s00.yaml` | `results/e4_mix_s00/E4_controller_drift_new` |
| `e6` | availability structure and participation (Fig. 4a-c, S16) | `protocol_e6_full.yaml` | `results/e6_full/E6_behaviour_availability_new` |
| `e9` | long-horizon closed-loop state (S19) | `protocol_e9_phase1.yaml` | `results/e9_phase1/E9_long_horizon_soc_new` |
| `e13`, `e13_full` | local control and response mechanisms (Fig. 3a-b, S5, S11) | `protocol_supp_full.yaml` | `results/supp_full/E13_control_logic_new` |
| `e14` | capacity concentration (S9) | `protocol_supp_full.yaml` | `results/supp_full/E14_dominant_capacity_new` |
| `e15` | availability structures, second family (S17) | `protocol_e15_full.yaml` | `results/e15_full/E15_availability_cases_new` |
| `e15_mix` | the same on the mixed populations (S18) | `e15_mix_s00.yaml` | `results/e15_mix_s00/E15_availability_cases_new` |
| `topo/{ieee33,ieee69,ieee123,published}` | the same scale experiment under three feeder topologies (S22) | `e1_mix_s00_<topology>_fb.yaml` | `results/e1_mix_s00_<topology>_fb/E1_scale_boundary_new` |
| `r2`, `r2_mix` | the R^2 and N95 readout of the scale experiment (Fig. 2a-e) | `tools/analysis/r2_paper_exact.py` | `results/r2_paper`, `results/r2_mix_s00` |
| `dataset_metadata.csv` | the population inventory | `nckeys.py` | - |

Which script needs which directory:

| Script | Inputs |
|---|---|
| `make_fig1.py` | `e1`, `e1_mix`, `r2`, `r2_mix` |
| `make_fig2_new.py` | `e2`, `e3`, `e13_full` |
| `make_fig3.py` | `e1`, `e4`, `e6` |
| `make_fig_transition.py` | `r2`, `r2_mix` |
| `make_fig_coverage.py` | `e1_mix`, `r2` |
| `make_appendix.py` | `e1`, `e1_mix`, `e2`, `e3`, `e4`, `e4_mix`, `e6`, `e9`, `e13_full`, `e14`, `e15`, `e15_mix`, `r2`, `topo` |
| `make_source_data.py` | `e1`, `e1_mix`, `e2`, `e3`, `e13`, `r2`, `r2_mix` |
| `make_source_data_new.py` | `e1`, `e2`, `e3`, `e4`, `e6`, `e13_full` |

`figures/data/` is not part of the code release: the figure inputs are deposited
with the paper as Source Data. Copy them into `figures/data/` under the directory
names above, or re-create them by running the experiments listed in the table and
copying each result directory.

## Supplementary tables

`tables/make_supp_tables_2-4.py` and `tables/make_table_only.py` write the
supplementary tables as `.docx` next to the script; they need `python-docx`.
