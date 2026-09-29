# Network-deliverability experiments

These modules produce the data of Fig. 5 and Supplementary Figs. S24-S41 (Result 4 and Supplementary Notes 12-17). They map aggregate dispatch onto the IEEE-33, IEEE-69 and IEEE-123 feeders and ask how much of a statistically controllable response survives branch-loading, voltage, transformer and spatial-placement constraints. Each module runs its own simulations, writes CSV and JSON data and PNG/PDF figures under `results/`, checkpoints per dataset or task, and resumes with the same command after an interruption.

Run from the repository root with `PYTHONPATH` set to it:

```bash
export PYTHONPATH="$PWD"
NX=src.extra.ieee33_device_day_simulation.network_experiments
python -m $NX.ieee69_network_implementation --seed-count 30 --bootstrap-draws 4000
```

The complete, ordered chain (including the per-dataset network baselines these modules need) is `bash scripts/run_reproduction.sh --network-inputs` followed by `--network`; the figures are drawn by `figures/make_fig5.py` and `figures/make_supplementary_network_figures.py`.

## Modules

| Module | Paper | Result directory |
|---|---|---|
| **Note 12 · Transferability across datasets and operating domains** | | |
| `train_pooled_model.py` | pooled cross-dataset EPS model ("E20 pooled EPS"); S24 | `results/transfer/` |
| `transfer_across_datasets.py` | in-domain, zero-shot and target-calibrated transfer; S24, S26 | `results/transfer/across_datasets/` |
| `transfer_alternative_splits.py` | alternative dataset splits; S25 | `results/transfer/rotations/` |
| **Note 13 · Effective population size and the emergence of predictable aggregate response** | | |
| `train_mixed_model.py` | mixed-population EPS model ("E21 mixed EPS") on the 14 multi-source scenarios | `results/effective_scale_mixtures/` |
| `train_mixed_model_source_unique.py` | the same model with source-unique sampling | `results/effective_scale_mixtures/unique/` |
| `effective_scale_published.py` | $R^2$ against normalized effective scale $\Gamma$ and $N$, published populations; S27, S32 | `results/effective_scale_mixtures/gamma_universal_boundary/` |
| `effective_scale_mixed.py` | the same for the mixed populations; S28, S29, S32 | `results/effective_scale_mixtures/gamma_mixed_boundary/` |
| `pairwise_predictability.py` | $R^2$ and $N_{95}$ of the 105 pairwise populations; S30, S31 | `results/effective_scale_mixtures/pairwise_r2/` |
| `pairwise_curtailment.py` | curtailment reduction of the 105 pairwise populations | `results/effective_scale_mixtures/pairwise_curtailment/` |
| `mixed_reference_controllers.py` | reference controllers on the mixed populations | `results/effective_scale_mixtures/curtailment_baseline_supplement/` |
| **Note 14 · Algorithm robustness and network implementation** (and Note 17) | | |
| `ieee69_network_implementation.py` | methods A-I under M0-M6 on IEEE-33 and IEEE-69; Fig. 5a, S33, S35, S41 | `results/ieee69_network_implementation/` |
| `ieee69_trained_vs_transferred.py` | EPS trained on IEEE-69 conditions versus transferred; S34, Fig. 5a-b, d-e | `.../trained_eps_ieee69_direct/` |
| `ieee33_trained_model.py` | EPS trained on IEEE-33 conditions; Fig. 5b, d-e | `.../trained_eps_ieee33_direct/` |
| `constraint_sweep.py` | one constraint varied at a time (branch loading, voltage bounds); Fig. 5c | `.../constraint_sweep/` |
| `constraint_sweep_transformer.py`, `constraint_sweep_transformer_summary.py` | transformer-capacity sweep and its summary; Fig. 5c | `.../constraint_sweep_transformer/` |
| `feeder_comparison_pairwise.py`, `feeder_comparison_references.py` | EPS and reference controllers on the three feeders; Fig. 5b, d-e | `.../feeder_comparison/` |
| **Note 15 · Physical feasibility boundaries under network stress** | | |
| `network_stress_boundary.py` | continuous request, line-capacity and spatial-concentration sweeps on IEEE-69; S36-S39 | `results/network_stress_boundary/` |
| `spatial_heterogeneity.py`, `spatial_heterogeneity_trained_model.py` | regional mixing, type zoning, spatial concentration, type-density imbalance and network coupling; Fig. 5f | `results/network_stress_boundary/`, `.../direct_training/` |
| **Note 16 · Safety auditing under realistic feeder constraints** | | |
| `ieee123_trained_model.py` | EPS trained on IEEE-123 conditions; Fig. 5b, d-e, g | `results/ieee123_safety_audit/trained_eps_ieee123_direct/` |
| `ieee123_safety_audit.py` | four IEEE-123 placements; Fig. 5g, S40 | `results/ieee123_safety_audit/` |
| **Shared libraries and optional checks** | | |
| `network_dispatch_protocol.py` | shared dispatch protocol, fleet construction and metrics | `results/<dataset>_ieee33_real_load/network_baseline/` |
| `reference_controllers.py` | implementations of the reference controllers | - |
| `ieee123_opendss_check.py` | optional three-phase OpenDSS cross-check of the IEEE-123 equivalent (needs `requirements-opendss.txt` and the official `IEEE123Master.dss`); not part of a paper figure | `results/ieee123_safety_audit/data/` |

## Coordination and reference methods

All methods see the same devices, availability, external energy input, random disturbances and network state within one condition and paired seed (Supplementary Table S5). Letters are those of Fig. 5a.

| Fig. 5a | Method | Online complexity | Operating principle |
|---|---|---|---|
| A | No coordination | O(0) | no control signal |
| B | Local SOC rules | O(1) per device | each device clips the request to its own SOC, availability and time-of-day range |
| C | MPC | O(HN) | receding-horizon allocation over a 12-step surplus and availability forecast; needs states and forecasts |
| D | Mean-field control | O(N) | one participation probability from the predicted population distribution over SOC bins |
| E | Virtual battery | O(N) | aggregate energy and power envelope mapped back to devices by fixed weights |
| F | Packetized energy management | O(N log N) | indivisible fixed-power packets, accepted or rejected whole |
| G | Transactive control | O(N log N) | device bids cleared at a discrete price |
| H | EPS | O(1) | one broadcast command and one aggregate feedback scalar; no per-device state upload |
| I | Centralized greedy reference | O(N) | observes all device and feeder states; a reference implementation, not a strict upper bound or a deployable schedule |

Supplementary Fig. S41 and Note 17 report ten rows: methods A-G and I plus two EPS variants, **"E20 pooled EPS"** (the pooled cross-dataset model of `train_pooled_model.py`, key `eps_e20_pooled`) and **"E21 mixed EPS"** (the mixed-population model of `train_mixed_model.py`, key `eps_e21_mixed`). In Fig. 5, method H is the EPS model trained on the target feeder conditions (key `eps_ieee69_fused`).

## Network conditions

**IEEE-69 conditions M0-M6** (Fig. 5a, Supplementary Table S3, Note 17). M0-M3 change the network state; M4-M6 change only where the devices sit. Parameters are in `STRESS_MODES` of `ieee69_network_implementation.py` and in [`configs/reproducibility.json`](../../../../configs/reproducibility.json).

| Condition | Meaning |
|---|---|
| M0 | matched operating point |
| M1 | deep-feeder high load |
| M2 | dual bottleneck with reverse flow |
| M3 | line derating combined with forecast error |
| M4 | uniform device placement |
| M5 | 50% of the devices on a distal feeder |
| M6 | 80% of the devices on a distal node |

**Continuous stress sweeps** (`network_stress_boundary.py`, S36-S39) vary one axis at a time on IEEE-69: request intensity 0.0-1.5, line-capacity stress 0.0-0.4 (capacity multiplied by one minus the stress) and spatial concentration 0.0-0.8. A boundary is declared after two consecutive points whose bootstrap upper bound of the effect falls below zero (or below -2 percentage points) or whose lower bound of physical risk exceeds 5%.

**IEEE-123 placements** (`ieee123_safety_audit.py`, Fig. 5g, S40). The keys below are configuration keys of the module and are unrelated to the numbering of the Supplementary Figures:

| Key | Placement |
|---|---|
| `S0_uniform` | uniform deployment |
| `S1_feeder_50` | 50% of the devices on a distal feeder |
| `S2_node_80` | 80% of the devices on a distal node |
| `S3_feeder_50_derated` | 50% distal-feeder concentration with line capacity reduced to 70% |

**Spatial heterogeneity** (`spatial_heterogeneity.py`, Fig. 5f): regionally mixed placement, type zoning, 25%, 50% and 75% density concentration, joint type-density imbalance, and adverse or favourable network coupling.

## Metrics

**Curtailment reduction.** The positive surplus at each step is the external energy input minus the local load. A requested charging response must pass both the battery and the network constraints, and only energy that is safely absorbed counts:

```text
curtailment reduction (%) = 100 * (uncontrolled curtailment - remaining curtailment) / uncontrolled curtailment
```

Absorption is matched inside the electrical partitions of the radial feeder: transfer within a partition is allowed, but surplus on one distal branch is never credited to charging on another. In Fig. 5b the reduction is expressed relative to the centralized greedy reference (reference = 100%); values above 100% can occur because the reference is not a strict upper bound.

**Network acceptance.** The fraction of a method's own requested response that remains feasible after network screening; 100% means the whole request is executable. Because methods request different magnitudes, acceptance is compared within a method across network conditions.

**Violation-free request fraction.** The share of dispatch steps in which the raw request, before the safety layer, causes no branch overload, voltage excursion or transformer overload. The violation-free execution fraction applies the same check after the safety layer; the **safety-layer gain** of Supplementary Fig. S41d compares the two.

**Added violation share.** Violations that a request adds relative to the no-coordination case, so that violations caused by the base load alone are not attributed to control.

**Retention.** The curtailment reduction on IEEE-69 divided by the same quantity on IEEE-33; spatial retention divides the value under M5 or M6 by the value under M4 on the same feeder.

**Aggregate-response $R^2$ and NRMSE** compare the target surplus power with the absorbed power at every five-minute step.

Voltages are in per unit with an admissible band of 0.95-1.05 p.u.; a branch or transformer loading of 1.0 is the limit; losses follow the resistive-loss approximation of LinDistFlow.

## Scope

The network layer is a balanced single-phase LinDistFlow model of standard IEEE test feeders, with thermal limits derived once from the standard bus loads. The IEEE-123 case is a single-phase structural equivalent of the official feeder that keeps its connectivity, line lengths and aggregated loads. Three-phase imbalance, protection, harmonics, communication faults and controller hardware are outside the scope of these experiments.
