# Broadcast coordination public-release materials

This directory contains the code and machine-readable metadata needed to reconstruct the derived datasets used in the paper. It does not redistribute any third-party source data, experiment results, model files, or supplementary tables.

The reconstruction chain is:

`original public datasets` -> `canonical preprocessing already documented in the source-data inventory` -> `scenario composition (S1-A--S6-C, corresponding to S1A--S6C)` -> `pairwise 50/50 mixing where applicable` -> `derived fleet manifests` -> `paper experiments`.

The official source repositories and records are listed in `configuration/source_metadata.json`. The source files must be obtained from those official locations and placed under the paths recorded in the configuration. Their licenses and citation requirements remain those of the original providers.

## Contents

- `generation/preprocess.py`: canonical-cache builder for the final public source adapters;
- `generation/src/extra/dataset_experiment/canonical_adapter.py` and `generation/src/extra/ieee33_device_day_simulation/population/device_day_loader.py`: the preprocessing implementation used by the composition pipeline;
- `generation/dataset_combinations/`: the standalone generator used by the project, copied from `src/extra/dataset_combinations/` without changing its logic;
- `configuration/scenario_config.json`: all 14 scenario definitions, including S1-A--S6-C weights, zone placement, partition seeds, test composition regimes, and allocation rules;
- `configuration/pairwise_seed_config.json`: pairwise seed formula and canonical dataset order;
- `configuration/generation_config.json`: runtime and output-layout contract;
- `configuration/source_metadata.json`: source paths and official public locations;
- `index/scenario_seed_index.csv`: every scenario partition and test seed for both sampling modes;
- `index/pairwise_index.csv`: all 105 unordered pair IDs, source mapping, equal weights, and seed formula;
- `DATA_AVAILABILITY.md`: the manuscript-ready Data availability paragraph.

## Rebuild

From the repository root, after installing `numpy` and `pandas` and obtaining the source datasets:

First build the canonical caches expected by the composition code:

```bash
PYTHONPATH=generation \
python generation/preprocess.py \
  --data-root data
```

The preprocessing configuration is in `configuration/preprocessing_config.json`; it records source paths, unit conversions, 288-point resampling, measured versus load-shape input mapping, and explicit battery-parameter fallbacks. The original source data remain outside this release.

```bash
PYTHONPATH=generation \
python -m dataset_combinations all \
  --data-root data \
  --output-root derived_data \
  --sampling-mode both \
  --force
```

To regenerate one scenario and one test replicate:

```bash
PYTHONPATH=generation \
python -m dataset_combinations scenarios \
  --data-root data \
  --output-root derived_data \
  --scenario S6-C \
  --partition test \
  --seed-index 0 \
  --sampling-mode both \
  --force
```

To regenerate one pairwise composition:

```bash
PYTHONPATH=generation \
python -m dataset_combinations pairwise \
  --data-root data \
  --output-root derived_data \
  --pair-id P001 \
  --seed-index 0 \
  --sampling-mode both \
  --force
```

The generator writes CSV fleet manifests and metadata. It does not copy the source time series into the release. `unique` sampling uses source profiles without replacement and may produce a smaller feasible fleet when source capacity is limiting. `non_unique` uses a fixed fleet of 5,000 logical devices and permits profile bootstrap only when required. Scenario test seeds 0--9 use nominal weights, 10--19 shift the dominant source by +15 percentage points, and 20--29 shift it by -15 percentage points; these are the implemented regimes, not a new composition rule.

The public release is tied to the GitHub tag `v1.0.0-public` in `https://github.com/songge00/broadcast-coordination-public-release`. A DOI is recorded in `DATA_AVAILABILITY.md` only after a real Zenodo archive has been created.
