# IEEE33 Real Snapshot Experiment

This is an independent experiment directory. It does not overwrite the
existing `weak_correlation` or `network_stress` outputs.

All traces use the same snapshot protocol:

- real NextGen device-day PV, load, battery parameters, and initial SOC;
- one real 5-minute profile index per observation;
- fixed EPS input for each profile index across independent fleet replicates;
- device battery state restored before every profile index;
- original Bernoulli response, battery constraints, and optional IEEE33 network
  feedback are retained.

Run one mode:

```bash
python -m src.extra.ieee33_device_day_simulation.experiments.ieee33_real_snapshot.run_experiment \
  --mode weak_correlation
```

```bash
python -m src.extra.ieee33_device_day_simulation.experiments.ieee33_real_snapshot.run_experiment \
  --mode network_stress
```

Outputs are written to:

```text
results/ieee33_real_snapshot_simulation/<mode>/
```

`data/n_threshold.json` records the fixed-snapshot multi-replication protocol
used by Figure 3A. The five paper-compatible figures are written to `Figs/`
and `paper_figures/` by the normal plotting step.
