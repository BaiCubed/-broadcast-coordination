<div align="center">

<h1>Datasets</h1>

<p><b>The public datasets behind the paper and the 119 mixed populations constructed from them</b></p>

<p>
<a href="#public-datasets"><img alt="Public datasets" src="https://img.shields.io/badge/Public%20datasets-15%20populations%20%2B%202%20reference-3A7BD5?style=for-the-badge"></a>
<a href="#mixed-populations"><img alt="Mixed populations" src="https://img.shields.io/badge/Mixed%20populations-119-7B4FC4?style=for-the-badge"></a>
<a href="#integrity-checks"><img alt="Integrity checks" src="https://img.shields.io/badge/Integrity-SHA--256-2E9E6A?style=for-the-badge"></a>
</p>

</div>

---

This directory holds all data of the study, organised in two parts:

| Folder | What it contains | In this repository |
|---|---|---|
| [`public_datasets/`](public_datasets) | the $\color{#1F6FEB}{\textbf{15 public datasets}}$ used as empirical populations, plus 2 reference datasets (NextGen and data2): official sources, a one-command download with checksum verification, and the preprocessing configuration | acquisition tools and metadata; the raw files are downloaded from the providers |
| [`mixed_populations/`](mixed_populations) | the $\color{#8250DF}{\textbf{119 mixed populations}}$: 14 multi-source scenarios and 105 pairwise mixtures, with their indices, generation configuration and checksums | **included in full** (about 29 MB) |

```text
datasets/
├── README.md                          this file
├── public_datasets/
│   ├── download.sh                    download of all public datasets with checksum verification
│   ├── source_metadata.json           official source and local path of every dataset
│   └── preprocessing_config.json      raw-to-canonical conversion of every dataset
└── mixed_populations/
    ├── mixed_scenarios/               14 multi-source populations, S1-A.csv ... S6-C.csv
    ├── pairwise/                      105 pairwise populations, P001.csv ... P105.csv
    ├── index/                         pair index and seed index
    ├── configuration/                 generation, scenario and pairwise-seed configuration
    ├── summary_checksums.json         checksums of the summary tables rebuilt from the CSVs
    └── SHA256SUMS                     checksums of every file in this folder
```

All commands below are run from the repository root.

<a name="public-datasets"></a>

## <img src="../assets/icons/data.svg" width="28" align="top" alt=""> $\color{#1F6FEB}{\textbf{Public datasets}}$

> [!NOTE]
> The original datasets are published by their providers under their own licences and are therefore obtained from the official sources listed below rather than copied into this repository; the download script fetches every file from its provider and verifies it against a recorded SHA-256 or MD5 checksum, so the data used here can be reproduced byte for byte.

### How to obtain them

```bash
bash datasets/public_datasets/download.sh              # 1. download and verify every dataset
python tools/data/preprocess.py                        # 2. raw files -> canonical device-day records
python tools/data/prepare_dataset_configs.py --all     # 3. per-dataset simulation configurations
```

1. **Download.** `download.sh` writes each dataset to `data/<directory>/raw/` at the repository root (the directory names are in the table below; `data/` is not tracked by git). Files that are already present are re-verified and skipped, so the script can be re-run safely.
2. **Preprocess.** `tools/data/preprocess.py` converts every raw dataset into canonical device-day records of 288 five-minute steps, written to `data/<directory>/processed/`. Unit conversions, resampling, the mapping of measured and load-shape inputs and the battery-parameter fallbacks of each dataset are recorded in [`preprocessing_config.json`](public_datasets/preprocessing_config.json).
3. **Configure.** `tools/data/prepare_dataset_configs.py --all` writes one simulation configuration per dataset and runs a hardware preflight.

> [!TIP]
> Archives in RAR format (European LV rural and urban-35k) need `bsdtar`, `7z` or `unar`. If the Irish CER file cannot be fetched automatically, the script prints the page and file name for a manual download. On slow links, `tools/data/par_download.py` (parallel, resumable) and `tools/data/segmented_fetch.py` (byte-range segments) can fetch the same files.

### The 15 datasets

| Dataset (Supplementary Table S1) | Resource / signal | Resolution | Sources used | Directory under `data/` | Official source |
|---|---|---|---:|---|---|
| BDG1 | building electricity | 1 h | 507 | `bdg1_building_data_genome` | [Building Data Genome 1](https://github.com/buds-lab/the-building-data-genome-project) |
| Low Carbon London | household electricity | 30 min | 5,000 | `low_carbon_london` | [London Datastore](https://data.london.gov.uk/download/vqm0d/3527bf39-d93e-4071-8451-df2ade1ea4f2/LCL-FullData.zip) |
| BDG2 | building electricity + solar | 1 h | 1,570 | `bdg2_building_data_genome` | [Building Data Genome 2](https://github.com/buds-lab/building-data-genome-project-2) |
| Danish heat meters | thermal demand | 1 h | 2,400 | `danish_smart_heat_meters` | [doi:10.5281/zenodo.6563114](https://doi.org/10.5281/zenodo.6563114) |
| Smart Grid Smart City | load + generation | 30 min | 600 | `smart_grid_smart_city` | [data.gov.au](https://data.gov.au/data/dataset/smart-grid-smart-city-customer-trial-data) |
| HEAPO | heat-pump electricity | 15 min / daily | 1,362 | `heapo_heat_pumps` | [doi:10.5281/zenodo.15056919](https://doi.org/10.5281/zenodo.15056919) |
| GoiEner | smart-meter electricity | 1 h | 5,000 | `goiener_smart_meters` | [doi:10.5281/zenodo.7362094](https://doi.org/10.5281/zenodo.7362094) |
| European LV urban-8k | LV active-power profile | 24/168 points | 5,000 | `european_lv_urban_8087` | [doi:10.17632/685vgp64sm.1](https://doi.org/10.17632/685vgp64sm.1) |
| European LV rural | LV active/reactive profile | 24/168 points | 2,731 | `european_lv_rural_2731` | [doi:10.17632/gspyzvvrhm.2](https://doi.org/10.17632/gspyzvvrhm.2) |
| European LV urban-35k | LV active/reactive profile | 24/168 points | 12,000 | `european_lv_urban_35297` | [doi:10.17632/gspyzvvrhm.2](https://doi.org/10.17632/gspyzvvrhm.2) |
| Norway AMI | MV/LV AMI profile | 1 h | 2,994 | `norway_ami_energy_distribution` | [doi:10.17632/jv3rz8k35r.1](https://doi.org/10.17632/jv3rz8k35r.1) |
| CAMSL-JP | smart-meter electricity | 30 min | 1,423 | `camsl_japan_smart_meters` | [doi:10.17632/cmpsyncmmk.1](https://doi.org/10.17632/cmpsyncmmk.1) |
| Irish CER | import/export household | 30 min | 2,904 | `irish_domestic_smart_meters` | [doi:10.6084/m9.figshare.31851922.v2](https://doi.org/10.6084/m9.figshare.31851922.v2) |
| OPSD | household multi-channel | 15 min | 11 | `opsd_household_data` | [Open Power System Data](https://data.open-power-system-data.org/household_data/opsd-household_data-2020-04-15.zip) |
| COMPLETE-EC | community load/PV/BESS/EV | 15 min | 250 | `complete_energy_community` | [doi:10.5281/zenodo.7602546](https://doi.org/10.5281/zenodo.7602546) |

### The 2 reference datasets

Supplementary Table S1 lists two further datasets that are not used as populations in the scale experiments but serve as reference data. They are downloaded by the same script:

| Dataset | Role in the study | Directory under `data/` | Official source |
|---|---|---|---|
| NextGen | device-day battery calibration population: battery response trajectories, state variables and power limits for the feeder simulation (100 household batteries, 3,000 device-days, 5 min) | `nextgen` | [doi:10.5281/zenodo.14885589](https://doi.org/10.5281/zenodo.14885589) |
| data2 | measured electric-vehicle charging sessions at 5 min resolution | `data2` | [doi:10.17632/c7gg94tmvz.3](https://doi.org/10.17632/c7gg94tmvz.3) |

Together with the 15 datasets they form the 17 dataset configurations of the IEEE-69 network-safety audit (Supplementary Fig. S41).

### Files in `public_datasets/`

| File | Content |
|---|---|
| [`download.sh`](public_datasets/download.sh) | URL, destination and checksum of every file of every dataset |
| [`source_metadata.json`](public_datasets/source_metadata.json) | official source, local path and reader type of each dataset |
| [`preprocessing_config.json`](public_datasets/preprocessing_config.json) | input files, caches, unit conversions and resampling of each dataset |

### Licences and citation

Each dataset remains under the licence of its provider; please respect those terms and cite the original data publications, which are listed with Supplementary Table S1 of the paper.

<a name="mixed-populations"></a>

## <img src="../assets/icons/structure.svg" width="28" align="top" alt=""> $\color{#8250DF}{\textbf{Mixed populations}}$

The $\color{#8250DF}{\textbf{119 mixed populations}}$ are controlled recombinations of the 15 public datasets that mix resource types, regions and data-collection designs. They consist of

- **14 multi-source scenarios**, `S1-A` to `S6-C` (Supplementary Table S2), each drawing devices from 3 to 15 datasets with fixed nominal weights, and
- **105 pairwise mixtures**, `P001` to `P105`, one for every pair of the 15 datasets, with equal 50/50 weights.

> [!IMPORTANT]
> The scenario codes `S1-A` to `S6-C` are the scenario names of Supplementary Table S2. They are unrelated to the numbering of the Supplementary Figures.

### The 14 multi-source scenarios

| Scenario | Name | Datasets | Nominal composition |
|---|---|---:|---|
| `S1-A` | Similar residential | 3 | CAMSL-JP 33.3%, Irish CER 33.3%, Low Carbon London 33.3% |
| `S1-B` | Building-heat pump-residential | 3 | BDG2 33.3%, HEAPO 33.3%, Smart Grid Smart City 33.3% |
| `S2-A` | LV network and AMI | 5 | European LV urban-35k 40%, European LV urban-8k 25%, European LV rural 15%, GoiEner 10%, Norway AMI 10% |
| `S2-B` | Building-residential-thermal-DER | 5 | BDG2 35%, Low Carbon London 25%, Danish heat meters 15%, HEAPO 15%, COMPLETE-EC 10% |
| `S3-A` | Ten-source electricity mix | 10 | Low Carbon London 18%, Smart Grid Smart City 14%, CAMSL-JP 12%, European LV urban-35k 10%, Irish CER 10%, European LV urban-8k 8%, GoiEner 8%, Norway AMI 8%, European LV rural 7%, HEAPO 5% |
| `S3-B` | Ten-source multi-sector mix | 10 | BDG2 15%, Norway AMI 12%, BDG1 10%, COMPLETE-EC 10%, Danish heat meters 10%, HEAPO 10%, Low Carbon London 10%, Smart Grid Smart City 10%, European LV rural 8%, OPSD 5% |
| `S4-A` | All datasets equally weighted | 15 | BDG1 6.7%, BDG2 6.7%, CAMSL-JP 6.7%, COMPLETE-EC 6.7%, Danish heat meters 6.7%, European LV rural 6.7%, European LV urban-35k 6.7%, European LV urban-8k 6.7%, GoiEner 6.7%, HEAPO 6.7%, Irish CER 6.7%, Low Carbon London 6.7%, Norway AMI 6.7%, OPSD 6.7%, Smart Grid Smart City 6.7% |
| `S4-B` | All dataset types equally weighted | 15 | COMPLETE-EC 15%, BDG1 10%, BDG2 10%, Danish heat meters 10%, HEAPO 10%, European LV rural 5%, European LV urban-35k 5%, European LV urban-8k 5%, Norway AMI 5%, OPSD 5%, CAMSL-JP 4%, GoiEner 4%, Irish CER 4%, Low Carbon London 4%, Smart Grid Smart City 4% |
| `S5-A` | Residential-dominant long tail | 15 | CAMSL-JP 13%, GoiEner 13%, Irish CER 13%, Low Carbon London 13%, Smart Grid Smart City 13%, European LV rural 5%, European LV urban-35k 5%, European LV urban-8k 5%, Norway AMI 5%, COMPLETE-EC 3%, BDG1 2.5%, BDG2 2.5%, Danish heat meters 2.5%, HEAPO 2.5%, OPSD 2% |
| `S5-B` | Building-thermal-dominant long tail | 15 | BDG1 20%, BDG2 20%, Danish heat meters 15%, HEAPO 15%, CAMSL-JP 3%, COMPLETE-EC 3%, GoiEner 3%, Irish CER 3%, Low Carbon London 3%, Smart Grid Smart City 3%, European LV rural 2.5%, European LV urban-35k 2.5%, European LV urban-8k 2.5%, Norway AMI 2.5%, OPSD 2% |
| `S5-C` | Network-DER-dominant long tail | 15 | COMPLETE-EC 20%, European LV rural 11.2%, European LV urban-35k 11.2%, European LV urban-8k 11.2%, Norway AMI 11.2%, OPSD 5%, CAMSL-JP 4%, GoiEner 4%, Irish CER 4%, Low Carbon London 4%, Smart Grid Smart City 4%, BDG1 2.5%, BDG2 2.5%, Danish heat meters 2.5%, HEAPO 2.5% |
| `S6-A` | Spatially clustered LV networks | 4 | European LV urban-35k 35%, European LV urban-8k 25%, European LV rural 20%, Norway AMI 20% |
| `S6-B` | Spatial cross-sector congestion | 5 | BDG2 35%, HEAPO 20%, Low Carbon London 20%, Danish heat meters 15%, Smart Grid Smart City 10% |
| `S6-C` | Spatial DER and bidirectional mix | 5 | COMPLETE-EC 30%, Smart Grid Smart City 30%, Irish CER 20%, Norway AMI 15%, OPSD 5% |

Percentages are rounded; the exact weights are in [`scenario_config.json`](mixed_populations/configuration/scenario_config.json). Test seeds 0-9 use the nominal weights, seeds 10-19 raise the dominant source by 15 percentage points and seeds 20-29 lower it by 15 percentage points.

### What each file contains

| Path | Rows | Content |
|---|---:|---|
| [`mixed_scenarios/S*.csv`](mixed_populations/mixed_scenarios) | 600 per file | 30 paired test seeds x 10 dispatch methods x 2 network modes (aggregate and IEEE-33). Each row gives the composition of the population for that seed (`nominal_weights_json`, `actual_weights_json`, `device_counts_json`, `fleet_size`), the fleet and method seeds, and the simulated dispatch outcome (curtailment before and after, absorbed energy, reduction in %, availability, network scale and violations, state-of-charge violations, charging and discharge energy) |
| [`pairwise/P*.csv`](mixed_populations/pairwise) | 30 per file | 30 paired test seeds under the aggregate mode. Each row gives the two datasets, their weights and device counts, the seed, profile reuse, and the same dispatch outcome |
| [`index/pairwise_index.csv`](mixed_populations/index/pairwise_index.csv) | 105 | pair id, the two datasets, their weights and the seed formula `20260808 + pair_index * 100000 + seed_index` |
| [`index/scenario_seed_index.csv`](mixed_populations/index/scenario_seed_index.csv) | | seed of every scenario, sampling mode and partition (train, validation, test seeds 0-29) |
| [`configuration/`](mixed_populations/configuration) | | `generation_config.json` (partition seed 20260714, fleet size, time steps), `scenario_config.json` (weights, seeds, placement), `pairwise_seed_config.json` |

### How they were generated

The device-level fleets behind every CSV are regenerated deterministically from the preprocessed public datasets:

```bash
python -m src.extra.dataset_combinations all --data-root data --sampling-mode both
```

Device profiles are split into train, validation and test partitions before any experiment runs (partition seed 20260714), and every test population is drawn with its own recorded seed. Two sampling modes are generated: `unique`, which draws each source profile at most once, and `non_unique`, which fixes the fleet at 5,000 logical devices and reuses profiles only when a source is too small. The generator is documented in [`src/extra/dataset_combinations/README.md`](../src/extra/dataset_combinations/README.md).

<a name="integrity-checks"></a>

## <img src="../assets/icons/tests.svg" width="28" align="top" alt=""> Integrity checks

These run offline in seconds:

```bash
(cd datasets/mixed_populations && sha256sum -c SHA256SUMS)    # every file of the mixed populations
python scripts/check_mixed_populations.py                     # 14 + 105 files present and well formed
python scripts/verify_mixed_populations.py                    # rebuild the summary tables and compare with summary_checksums.json
```

## <img src="../assets/icons/license.svg" width="28" align="top" alt=""> Licence

The constructed mixed populations are released under the [MIT License](../LICENSE) of this repository. The public datasets remain under the licences of their providers.
