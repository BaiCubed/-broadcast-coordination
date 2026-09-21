# Mixture fleet generator

This module builds the synthetic mixture fleets used wherever the paper reports
results on mixed populations: 14 predefined scenarios and 105 pairwise 50/50
combinations, 119 mixtures in total. Each mixture is generated in two sampling
modes:

- `non_unique` - a fixed 5000 logical devices; a real source may be drawn more
  than once when the pool is too small;
- `unique` - real sources are drawn without replacement, so each real source
  appears at most once inside a fleet.

The module reads nothing but the 15 standardised datasets under `data/`. It does
not import any experiment module, does not read experiment results and does not
depend on a trained model.

The output of a mixture is a CSV fleet manifest. Each row carries the canonical
source file, the source row index or locator, the dataset, the real source id,
the day, the device parameters and the IEEE-33 bus. The full 288-point time
series stays in the source dataset and is never copied into an NPZ.

## Running

```bash
python -m src.extra.dataset_combinations all --sampling-mode both --force
```

One scenario:

```bash
python -m src.extra.dataset_combinations scenarios \
  --scenario S1-A --partition test --seed-index 0 \
  --sampling-mode both --force
```

One pair:

```bash
python -m src.extra.dataset_combinations pairwise \
  --pair-id P001 --seed-index 0 --sampling-mode both --force
```

## Requirements

- Python 3.10 or newer
- NumPy
- pandas
- the 15 datasets under `data/`

No experiment result directory and no experiment module has to be present.

## Self-test

```bash
python -m src.extra.dataset_combinations.self_test \
  --data-root data \
  --reference-root data/dataset_combinations
```

The self-test regenerates the unique and non-unique fleets of S1-A and P001,
checks the source-unique constraint, and compares the twelve core columns row by
row against the reference manifests. Without `--reference-root` it reads no
existing combination output at all:

```bash
python -m src.extra.dataset_combinations.self_test --data-root data
```

## Fixed seeds

- profile train/validation/test partition seed: `20260714`;
- the train seed, the validation seed and the 30 test seeds of the predefined
  scenarios are fixed in `constants.py`;
- pairwise seed: `20260808 + pair_index * 100000 + seed_index`;
- the dataset order, the numbering of the 105 pairs and the IEEE-33 partition
  map are all fixed in `constants.py`.

The same source data, the same dependency versions and the same command
reproduce the twelve core columns exactly. The additional `source_path`,
`source_row_index` and `source_locator` columns only record source provenance
and do not change any device parameter.

## Packaging

```bash
bash src/extra/dataset_combinations/package.sh
```

writes

```text
dist/mixed_dataset_generator.tar.gz
dist/mixed_dataset_generator.tar.gz.sha256
```

The archive contains only the standalone source of
`src/extra/dataset_combinations/`, its documentation and its dependency
declaration: no data, results, models, CSV, NPZ or Python caches.
