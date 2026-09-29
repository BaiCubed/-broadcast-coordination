# Experiment and plot ownership audit

The release separates three relationships:

1. Experiment-owned entrypoints and experiment-owned source files.
2. Shared numerical foundations, which are declared explicitly and are not
   silently counted as experiment files.
3. Read-only result/model artifacts passed between experiments.

E23 may consume frozen E22 result/model products. E24 may consume frozen E22
and E23 products. These are artifact edges, not shared implementation files.

Every file under `plots/leaf/` owns one logical figure. PNG and PDF files are
format variants of one figure. Files under `plots/composites/` and the
registered builders in `PLOT_REGISTRY.json` assemble leaf outputs, Appendix
collections, supplementary tables, presentations, or reports. They are not
treated as one script per leaf plot.

Run:

```bash
python scripts/audit_ownership.py
```

The command writes `EXPERIMENT_OWNERSHIP_AUDIT.csv`,
`PLOT_LEAF_OWNERSHIP.csv`, `PLOT_COMPOSITE_AUDIT.csv`, and
`OWNERSHIP_AUDIT_SUMMARY.json`. A passing audit requires unique experiment
entrypoints, no owned-file collisions, one owner for every leaf output, and
all registered composite builders to exist.
