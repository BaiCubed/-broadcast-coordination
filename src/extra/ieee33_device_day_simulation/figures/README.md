# Network-deliverability experiments

These scripts produce Figure 5 and Supplementary Figs. S23-S40. They run their
own simulations and compare the broadcast controller with eight reference
controllers under explicit feeder constraints, writing their data as CSV and
JSON and their figures as PNG and PDF into a result directory.

`plot_figures.py`, `legacy_compat.py` and `data2_figure_labels.py` are the figure
helpers of the device-day simulation and are imported by the experiment
packages; the remaining modules are the experiments themselves.

## Controllers

All controllers see the same devices, availability, external energy input,
random disturbances and network conditions within one condition and seed.

| Controller | Online complexity | Description |
|---|---|---|
| No coordination | O(0) | no control signal is sent |
| Local SOC rules | O(1) per device | each device acts on its own SOC, availability and time-of-day thresholds |
| MPC | O(HN) | receding-horizon water filling over a 12-step surplus and availability forecast |
| Mean-field control | O(N) | one participation probability from the predicted population distribution over SOC bins |
| Virtual battery | O(N) | an aggregate energy and power envelope mapped back to devices by fixed weights |
| Packetized energy management | O(N log N) | indivisible fixed-power packets, accepted whole |
| Transactive control | O(N log N) | bids ordered and cleared at a discrete price |
| Broadcast controller | O(1) | one broadcast signal and one aggregate feedback scalar; no per-device state upload |
| Centralized greedy bound | O(N) | a device-layer absorption bound, not a network-executable schedule |

## Stress and placement scenarios

The IEEE-69 experiments sweep seven scenarios. M0-M3 change the network stress
and M4-M6 change only the spatial deployment of the devices:

| Scenario | Meaning |
|---|---|
| M0 | matched operating point |
| M1 | deep-feeder high load |
| M2 | dual bottleneck with reverse flow |
| M3 | line derating combined with forecast error |
| M4 | uniform device placement |
| M5 | 50% of the devices on a distal feeder |
| M6 | 80% of the devices on a distal bus |

The continuous-stress experiment scans three independent axes instead: broadcast
request intensity `q` (0.0-1.5), line-capacity stress `lambda_line` (0.0-0.4,
capacity multiplied by `1 - lambda_line`) and spatial concentration `kappa`
(0.0-0.8).

## Metrics

**Deliverable curtailment reduction.** The positive surplus at each step is the
external energy input minus the local load. A requested charging response still
has to pass the battery and the network constraints; only energy that is safely
absorbed counts:

```text
100 * (uncontrolled curtailment - remaining curtailment) / uncontrolled curtailment
```

Curtailment absorption is matched inside the electrical partitions of the radial
network: transfer within a partition is allowed, surplus of one distal branch is
never credited to charging on another.

**Network acceptance.** The accepted response after the network check divided by
the requested response. 100% means the whole request is executable.

**Violation-free request fraction.** The share of dispatch steps in which the raw
request, before the safety layer, causes no branch overload, voltage excursion or
transformer overload. The violation-free execution fraction is the same check
after the safety layer.

**Added violation share.** The violations that the request adds relative to the
no-coordination baseline, so that violations caused by the base load alone are
not attributed to control.

**Retention.** The deliverable curtailment reduction on IEEE-69 divided by the
same quantity on IEEE-33. Spatial retention divides M5 or M6 by M4 on the same
network.

**R^2 and NRMSE** compare the target surplus power and the absorbed power at
every five-minute step.

Voltages are per unit with an admissible band of 0.95-1.05 p.u.; a branch or
transformer loading of 1.0 is the limit; losses are in kW from the resistive loss
approximation of LinDistFlow.

## Scope

The network layer is a balanced single-phase LinDistFlow on standard IEEE test
feeders, with frozen thermal limits derived once from the standard bus loads.
Three-phase imbalance, full non-linear AC power flow, protection, harmonics,
communication faults and controller hardware are out of scope. The IEEE-123 case
is a structural equivalent of the official OpenDSS feeder that keeps its
connectivity, line lengths and aggregated loads, not a multi-phase solution.

## Running

From the repository root, with `PYTHONPATH` set to it:

```bash
python -m src.extra.ieee33_device_day_simulation.figures.run_e20_transfer
python -m src.extra.ieee33_device_day_simulation.figures.run_e21_curtailment_baseline_supplement
python -m src.extra.ieee33_device_day_simulation.figures.run_e22_ieee69_complexity
bash scripts/run_network_audits.sh --all
```

Every experiment checkpoints per dataset and skips work that is already
complete, so an interrupted run continues with the same command.
