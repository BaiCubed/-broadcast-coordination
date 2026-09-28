# Real Snapshot Network Ablation

This experiment records the real-data ablation needed to separate temporal
co-movement from IEEE33 network constraints. It uses the existing
`OriginalEPSAdapter`, real NextGen device-day records, real PV/load/SOC and
the existing battery constraints. No new battery or control model is added.

The suite writes data only. Plotting is intentionally postponed until the
three data groups have been inspected together.

After inspection, the new composite Figure 6 can be generated independently;
it does not touch any earlier figure output.

## Groups

1. `independent_real_snapshot_no_network`: each resource samples an
   independent real within-day profile point at every experiment step;
   network feedback is disabled.
2. `time_aligned_real_snapshot_no_network`: all resources use the same real
   profile time index; IEEE33 is evaluated only as a diagnostic layer and does
   not change accepted dispatch.
3. `time_aligned_real_snapshot_network`: the same time-aligned real inputs,
   with the existing IEEE33 outer network feedback active.

All groups use the same deterministic device-day sampling, EPS schedule,
resource counts, validation protocol and N grid. Figure 3A's threshold
protocol uses eight training and eight evaluation repetitions; the N-scaling
repeat count remains the value in the shared experiment configuration.
Artificial `zone_correlation` shocks are disabled. The three groups differ
only in real profile alignment and network feedback.

## Run

```bash
python -m src.extra.ieee33_device_day_simulation.experiments.snapshot.run_experiment
```

Outputs are written below
`results/ieee33_real_snapshot_simulation/snapshot/`. Each group contains
`data/`, `estimation/`, and protocol metadata. No `Figs/` directory is made
by this runner.
