# Additional source-data archives

The two source datasets used by the project's real-data validation code are distributed as CSV archives attached to the `v1.0.0-public` GitHub release. They are not committed to the Git history because the archives are large. See `manifest.json` for the exact checksums and URLs.

| Dataset | Release asset | Contents | Source and licence |
|---|---|---|---|
| NextGen household batteries | [`nextgen-household-batteries.tar.gz`](https://github.com/songge00/broadcast-coordination-public-release/releases/download/v1.0.0-public/nextgen-household-batteries.tar.gz) | 100 original household CSV files | Zenodo record `10.5281/zenodo.14885589`, CC BY 4.0 |
| data2 EV charging | [`data2-ev-charging-csv.zip`](https://github.com/songge00/broadcast-coordination-public-release/releases/download/v1.0.0-public/data2-ev-charging-csv.zip) | CSV exports of both source workbooks, all 16 sheets and 2,938,319 records | Mendeley Data `10.17632/c7gg94tmvz.3`, CC BY 4.0 |

The data2 archive preserves the original column names and sheet separation; the project adapter uses `processed_data_longer_than_30` when that configuration is selected.

The other public source datasets remain external and are listed in `configuration/source_metadata.json`; they are not redistributed here.
