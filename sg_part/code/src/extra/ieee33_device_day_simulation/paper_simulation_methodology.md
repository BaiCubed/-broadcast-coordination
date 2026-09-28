# Simulation methodology

This document records the reproducible IEEE-33 device-day simulation protocol. The authoritative numeric values are stored in `code/config/reproducibility.json`; experiment-specific overrides are stored in each E20-E24 manifest.

The simulation uses 288 five-minute steps per day, deterministic composition and test seeds, response-sign-stratified fitting/calibration, direction-specific quantile neural networks, explicit network constraints, and separate result-to-figure builders. Run the ordered chain in the repository-level README.
