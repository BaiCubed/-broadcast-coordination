# Mixed-population generator

This standalone module builds the **119 constructed mixed populations** of the paper (Supplementary Note 1, Supplementary Table S2, Supplementary Fig. S2): the 14 multi-source scenarios, scenario S1-A to scenario S6-C, and the 105 pairwise 50/50 mixtures `P001` to `P105` of the 15 published populations. The mixed populations are controlled recombinations used as composition stress tests; they are not independent empirical datasets.

The generator reads only the canonical device-day records of the 15 public datasets under `data/` (produced by `tools/data/preprocess.py`). It imports no experiment module, reads no experiment result and needs no trained model.

## Relation to `derived_data/`

The repository ships the constructed mixed-population datasets under [`derived_data/`](../../../derived_data):

| Path | Content |
|---|---|
| `derived_data/generated_data/mixed_scenarios/<scenario>.csv` | one file per multi-source scenario (S1-A ... S6-C) |
| `derived_data/generated_data/pairwise/P001.csv` ... `P105.csv` | one file per pairwise mixture |
| `derived_data/index/scenario_seed_index.csv` | train, validation and 30 test seeds of every scenario and sampling mode |
| `derived_data/index/pairwise_index.csv` | source pair, weights and seed formula of every pairwise mixture |
| `derived_data/configuration/*.json` | generation, scenario, pairwise-seed, preprocessing and source configuration |

These files record the composition, seeds and simulated dispatch outcome of every population. This module regenerates the underlying device-level fleets from the public data with exactly the same seeds, source order and weights, which are fixed in [`constants.py`](constants.py) and mirrored in `derived_data/configuration/`.

## Sampling modes

Every mixture is generated in two sampling modes:

- `non_unique`: a fixed fleet of 5,000 logical devices; a real source may be drawn more than once when its pool is too small;
- `unique`: real sources are drawn without replacement, so each real source appears at most once in a fleet.

## Output

Each fleet is written as a CSV manifest (`devices.csv`) plus metadata under

```text
data/dataset_combinations/<sampling-mode>/ieee33/<scenario>/{train,validation,test/seed_00..seed_29}/
data/dataset_combinations/<sampling-mode>/ieee33/pairwise/<pair-id>/seed_00..seed_29/
```

Each row carries the canonical source file, the source row index or locator, the dataset, the real source id, the day, the device parameters and the IEEE-33 bus. The 288-point time series stay in the canonical source records and are not copied.

## Running

From the repository root, with `PYTHONPATH` set to it:

```bash
python -m src.extra.dataset_combinations all --sampling-mode both --force
```

One scenario, one partition and one test seed:

```bash
python -m src.extra.dataset_combinations scenarios \
  --scenario S1-A --partition test --seed-index 0 --sampling-mode both --force
```

One pair:

```bash
python -m src.extra.dataset_combinations pairwise \
  --pair-id P001 --seed-index 0 --sampling-mode both --force
```

`--data-root` (default `data`) and `--output-root` (default `data/dataset_combinations`) change the input and output locations. Existing fleets are skipped unless `--force` is given.

The population-scale experiments use device pools built from these fleets:

```bash
python tools/data/build_combo_pools.py --seeds 0-29 --workers 16
```

## Fixed seeds

- profile train / validation / test partition seed: `20260714`;
- scenario seeds: one train seed, one validation seed and 30 test seeds per scenario, listed in `derived_data/index/scenario_seed_index.csv`;
- pairwise seed: `20260808 + pair_index * 100000 + seed_index`;
- the dataset order, the numbering of the 105 pairs and the IEEE-33 zone map are fixed in `constants.py`.

With the same source data and dependency versions, the same command reproduces the twelve core columns of every manifest exactly. The additional `source_path`, `source_row_index` and `source_locator` columns record provenance only and do not affect any device parameter.

## Self-test

```bash
python -m src.extra.dataset_combinations.self_test --data-root data
```

The self-test regenerates the unique and non-unique fleets of scenario S1-A and pair P001 and checks the source-unique constraint. With `--reference-root data/dataset_combinations` it also compares the twelve core columns row by row against previously generated manifests.

## Requirements

Python 3.10 or newer, NumPy and pandas ([`requirements.txt`](requirements.txt)), and the canonical records of the 15 datasets under `data/`.

## Packaging

```bash
bash src/extra/dataset_combinations/package.sh
```

writes `dist/mixed_dataset_generator.tar.gz` and its `.sha256`. The archive holds only the source of this module, this documentation and its dependency list; no data, results, models or caches.
