# european lv rural 2731 ieee33 real load

This module adapts the named public dataset to the IEEE-33 device-day simulation protocol.

Run from the release `code/` directory:

```bash
python -m src.extra.european_lv_rural_2731_ieee33_real_load.run
```

The adapter reads the local dataset directory described in `data/configuration/preprocessing_config.json` and writes deterministic results under `results/european_lv_rural_2731_ieee33_real_load/`. The shared network, estimator, seed, split and output contracts are documented in `code/README.md` and `code/config/reproducibility.json`. Dataset-specific field mappings and licensing information are recorded in the corresponding `data/<dataset>/README.md`.
