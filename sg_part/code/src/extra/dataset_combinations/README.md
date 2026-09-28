# Dataset composition generator

This module generates 14 predefined mixed scenarios and 105 50/50 pairwise combinations. Both `non_unique` fixed-size fleets and `unique` without-replacement fleets are supported. The generator reads only canonical data under the configured data root and never reads model checkpoints or experiment results.

```bash
python -m src.extra.dataset_combinations all --sampling-mode both
python -m src.extra.dataset_combinations scenarios --scenario S1-A --partition test --seed-index 0 --sampling-mode both
```

The partition, seed and output contracts are defined by the JSON configuration files under `data/configuration/`.
