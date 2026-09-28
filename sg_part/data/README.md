# Data component

This directory contains the derived release data required to inspect and validate the composition protocol. It does not redistribute third-party raw archives. Download raw files from the original providers, comply with their licenses, and place them under the local directories specified by [`configuration/preprocessing_config.json`](configuration/preprocessing_config.json).

## Rebuild derived data

From `sg_part/code`:

```bash
python generation/preprocess.py --data-root ../data
python -m src.extra.dataset_combinations all \
  --data-root ../data --output-root ../data/dataset_combinations
```

The preprocessing cache format is a canonical 288-point device-day record with source, day, device, zone and bus identifiers. The composition generator writes fixed train/validation partitions and 30 deterministic test seeds for each mixed scenario and pairwise combination. `configuration/generation_config.json`, `configuration/preprocessing_config.json`, `configuration/scenario_config.json` and `configuration/pairwise_seed_config.json` are the protocol records.

## Official source catalog

The source URLs below are the original dataset locations. They are included for provenance and acquisition only; the release package does not depend on the URL of any comparison repository.

| Dataset | Original source |
|---|---|
| Building Data Genome 1 | https://github.com/buds-lab/the-building-data-genome-project |
| Building Data Genome 2 | https://github.com/buds-lab/building-data-genome-project-2 |
| Low Carbon London | https://data.london.gov.uk/download/vqm0d/3527bf39-d93e-4071-8451-df2ade1ea4f2/LCL-FullData.zip |
| Danish smart heat meters | https://doi.org/10.5281/zenodo.6563114 |
| Smart Grid Smart City | https://data.gov.au/data/dataset/smart-grid-smart-city-customer-trial-data |
| HEAPO heat pumps | https://doi.org/10.5281/zenodo.15056919 |
| GoiEner smart meters | https://doi.org/10.5281/zenodo.7362094 |
| European LV Urban 8087 | https://doi.org/10.17632/685vgp64sm.1 |
| European LV Rural 2731 and Urban 35297 | https://doi.org/10.17632/gspyzvvrhm.2 |
| Norway AMI Energy Distribution | https://doi.org/10.17632/jv3rz8k35r.1 |
| CAMSL Japan smart meters | https://doi.org/10.17632/cmpsyncmmk.1 |
| Irish domestic smart meters | https://doi.org/10.6084/m9.figshare.31851922.v2 |
| OPSD household data | https://data.open-power-system-data.org/household_data/opsd-household-data-2020-04-15.zip |
| Complete Energy Community | https://doi.org/10.5281/zenodo.7602546 |
| NextGen device days | https://doi.org/10.5281/zenodo.14885589 |
| data2 charging sessions | https://doi.org/10.17632/c7gg94tmvz.3 |

The complete machine-readable catalog, local paths, conversion notes and checksums are in `configuration/source_metadata.json` and `SHA256SUMS`.
