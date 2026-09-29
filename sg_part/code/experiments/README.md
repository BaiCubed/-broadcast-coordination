# Experiment ownership

Each experiment has one directory and one public entrypoint. The entrypoints
delegate to the validated implementation under `src/extra`; they do not share
experiment-specific state. Result files are explicit data products. E23 may
read the frozen E22 model/result products, and E24 may read the E22/E23 result
products, as recorded in `audit/EXPERIMENT_OWNERSHIP.json`.
