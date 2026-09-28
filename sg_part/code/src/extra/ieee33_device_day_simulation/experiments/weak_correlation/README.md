# Weak-Correlation IEEE 33 Experiment

This mode keeps the IEEE 33 topology as a diagnostic layer. It records feeder,
voltage, and transformer metrics, but network limits do not modify EPS power.
It uses the same device-day sample, seeds, scenarios, validation protocol, and
N-scaling protocol as `network_stress`.

Run from the repository root:

```bash
python -m src.extra.ieee33_device_day_simulation.experiments.weak_correlation.run_experiment
python -m src.extra.ieee33_device_day_simulation.figures.plot_figures \
  --results-root results/ieee33_device_day_simulation/weak_correlation
```
