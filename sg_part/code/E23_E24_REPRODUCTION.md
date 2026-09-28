# E23-E24 reproduction notes

E23 evaluates the IEEE-69 pressure and spatial-heterogeneity boundaries using direct IEEE-69 models. E24 trains direct IEEE-123 structural-equivalent models and audits effectiveness and network safety across the four placement and capacity scenarios. Run E23 before E24 because E24 uses the same canonical device-day protocol but a distinct network configuration.

```bash
python -m src.extra.ieee33_device_day_simulation.figures.run_e23_ieee69_relative_boundary
python -m src.extra.ieee33_device_day_simulation.figures.run_e23_spatial_heterogeneity
python -m src.extra.ieee33_device_day_simulation.figures.run_e24_ieee123_direct_training
python -m src.extra.ieee33_device_day_simulation.figures.run_e24_ieee123_audit \
  --model-source results/E24/trained_eps_ieee123_direct/models
```

The optional multiphase audit requires the official IEEE-123 OpenDSS case and the package in `requirements-e24-multiphase.txt`. The E24 audit must receive the IEEE-123 model directory explicitly; the default frozen-model path can otherwise select an IEEE-69 model regime.
