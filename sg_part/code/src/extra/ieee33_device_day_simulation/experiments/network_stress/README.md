# Network-Stress IEEE 33 Experiment

This mode uses the same protocol as `weak_correlation`, but feeder, voltage,
and transformer limits feed back through the configured proportional dispatch
scale. It is the active network-constraint comparison.

Run from the repository root:

```bash
python -m src.extra.ieee33_device_day_simulation.experiments.network_stress.run_experiment
python -m src.extra.ieee33_device_day_simulation.figures.plot_figures \
  --results-root results/ieee33_device_day_simulation/network_stress
```
