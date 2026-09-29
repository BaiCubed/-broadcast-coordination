#!/usr/bin/env bash
set -e
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "error: neither python3 nor python found; some official APIs cannot be queried."
    exit 1
  fi
fi

mkdir -p data/raw data/processed

hash_value() {
  algo="$1"
  file="$2"
  case "$algo" in
    sha256)
      if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$file" | awk '{print $1}'
      else
        shasum -a 256 "$file" | awk '{print $1}'
      fi
      ;;
    md5)
      if command -v md5sum >/dev/null 2>&1; then
        md5sum "$file" | awk '{print $1}'
      else
        md5 -q "$file"
      fi
      ;;
    none|"")
      echo ""
      ;;
    *)
      echo "error: unsupported checksum algorithm $algo"
      exit 1
      ;;
  esac
}

verify_checksum() {
  dataset="$1"
  file="$2"
  algo="${3:-none}"
  expected="${4:-}"
  if [ -z "$expected" ] || [ "$algo" = "none" ]; then
    return 0
  fi
  actual="$(hash_value "$algo" "$file")"
  if [ "$actual" != "$expected" ]; then
    echo "error: checksum mismatch for dataset $dataset: $file"
    echo "expected $algo: $expected"
    echo "actual   $algo: $actual"
    exit 1
  fi
  echo "Dataset $dataset checksum OK ($algo)"
}

fetch_url() {
  url="$1"
  tmp="$2"
  if command -v wget >/dev/null 2>&1; then
    wget --progress=bar:force --tries=3 --timeout=30 -O "$tmp" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 --connect-timeout 30 --progress-bar -o "$tmp" "$url"
  else
    echo "error: neither wget nor curl found."
    return 1
  fi
}

download_file() {
  dataset="$1"
  url="$2"
  dest="$3"
  algo="${4:-none}"
  checksum="${5:-}"
  mkdir -p "$(dirname "$dest")"
  if [ -s "$dest" ]; then
    echo "Dataset $dataset already exists, skipping: $dest"
    verify_checksum "$dataset" "$dest" "$algo" "$checksum"
    return 0
  fi
  echo "Downloading Dataset $dataset: $dest"
  tmp="${dest}.part"
  rm -f "$tmp"
  fetch_url "$url" "$tmp"
  mv "$tmp" "$dest"
  verify_checksum "$dataset" "$dest" "$algo" "$checksum"
}

download_file_optional() {
  dataset="$1"
  url="$2"
  dest="$3"
  algo="${4:-none}"
  checksum="${5:-}"
  mkdir -p "$(dirname "$dest")"
  if [ -s "$dest" ]; then
    echo "Dataset $dataset already exists, skipping: $dest"
    verify_checksum "$dataset" "$dest" "$algo" "$checksum"
    return 0
  fi
  echo "Trying optional download for Dataset $dataset: $dest"
  tmp="${dest}.part"
  rm -f "$tmp"
  set +e
  fetch_url "$url" "$tmp"
  status="$?"
  set -e
  if [ "$status" -ne 0 ]; then
    rm -f "$tmp"
    return 1
  fi
  mv "$tmp" "$dest"
  verify_checksum "$dataset" "$dest" "$algo" "$checksum"
}

extract_zip() {
  dataset="$1"
  archive="$2"
  dest="$3"
  marker="$4"
  if [ -e "$marker" ]; then
    echo "Dataset $dataset already extracted, skipping: $marker"
    return 0
  fi
  if ! command -v unzip >/dev/null 2>&1; then
    echo "error: unzip is required to extract $dataset."
    exit 1
  fi
  echo "Extracting Dataset $dataset: $archive -> $dest"
  mkdir -p "$dest"
  unzip -q -n "$archive" -d "$dest"
}

extract_rar() {
  dataset="$1"
  archive="$2"
  dest="$3"
  marker="$4"
  if [ -e "$marker" ]; then
    echo "Dataset $dataset already extracted, skipping: $marker"
    return 0
  fi
  echo "Extracting Dataset $dataset: $archive -> $dest"
  mkdir -p "$dest"
  if command -v bsdtar >/dev/null 2>&1; then
    bsdtar -xf "$archive" -C "$dest"
  elif command -v 7z >/dev/null 2>&1; then
    7z x -y "-o$dest" "$archive"
  elif command -v unar >/dev/null 2>&1; then
    unar -quiet -force-overwrite -output-directory "$dest" "$archive"
  else
    echo "$dataset ships a RAR5 archive; bsdtar, 7z or unar is required to extract it."
    echo "install one of them and re-run, or extract $archive into $dest manually."
  fi
}

download_zenodo_record_files() {
  dataset="$1"
  record_id="$2"
  dest_dir="$3"
  mkdir -p "$dest_dir"
  manifest="$(mktemp)"
  "$PYTHON_BIN" - "$record_id" > "$manifest" <<'PY'
import json
import sys
import urllib.request

record_id = sys.argv[1]
url = f"https://zenodo.org/api/records/{record_id}"
request = urllib.request.Request(url, headers={"User-Agent": "broadcast-coordination-data-script"})
with urllib.request.urlopen(request, timeout=60) as response:
    data = json.load(response)
for item in data.get("files", []):
    key = item["key"]
    checksum = item.get("checksum", "")
    link = item.get("links", {}).get("self") or item.get("links", {}).get("download")
    if not link:
        continue
    print(f"{key}\t{checksum}\t{link}")
PY
  while IFS="$(printf '\t')" read -r key checksum url; do
    algo="none"
    value=""
    if [ -n "$checksum" ] && echo "$checksum" | grep -q ':'; then
      algo="${checksum%%:*}"
      value="${checksum#*:}"
    fi
    download_file "$dataset" "$url" "$dest_dir/$key" "$algo" "$value"
  done < "$manifest"
  rm -f "$manifest"
}

echo "== 1. Building Data Genome 1 =="
download_file "bdg1_building_data_genome" \
  "https://raw.githubusercontent.com/buds-lab/the-building-data-genome-project/master/data/processed/temp_open_utc_complete.csv" \
  "data/bdg1_building_data_genome/raw/temp_open_utc_complete.csv" \
  sha256 "39881f4253367b59db9c94b7a88defa0c165c373c54193771807e8ed50c2aece"
download_file "bdg1_building_data_genome" \
  "https://raw.githubusercontent.com/buds-lab/the-building-data-genome-project/master/data/raw/meta_open.csv" \
  "data/bdg1_building_data_genome/raw/meta_open.csv" \
  sha256 "3032035004ea8e16bd5def27ee3f11e2e9b40262c8926ca779fb87d6c2da84e1"

echo "== 2. Low Carbon London =="
download_file "low_carbon_london" \
  "https://data.london.gov.uk/download/vqm0d/3527bf39-d93e-4071-8451-df2ade1ea4f2/LCL-FullData.zip" \
  "data/low_carbon_london/raw/LCL-FullData.zip" \
  sha256 "68a35598cc8e70898a7651c482216fb096a0a6910c2d180c616632dfc7a9b014"

echo "== 3. Building Data Genome 2 =="
BDG2_BASE="https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/master"
download_file "bdg2_building_data_genome" "$BDG2_BASE/data/meters/cleaned/electricity_cleaned.csv" "data/bdg2_building_data_genome/raw/electricity_cleaned.csv" sha256 "b6ffc9b4dfcefe5c753594730a08ae822b0d50fec6815abb8f185591e6c630a3"
download_file "bdg2_building_data_genome" "$BDG2_BASE/data/meters/cleaned/chilledwater_cleaned.csv" "data/bdg2_building_data_genome/raw/chilledwater_cleaned.csv" sha256 "8211aaf210379af50cbf7af87579d12414c3500d3e3f4dea725761c93a172bcd"
download_file "bdg2_building_data_genome" "$BDG2_BASE/data/meters/cleaned/steam_cleaned.csv" "data/bdg2_building_data_genome/raw/steam_cleaned.csv" sha256 "ea5956c49ed1d6cc1b611a752b5a9dc3bcd4d45ddcd674e53b3c41661a6d3b9b"
download_file "bdg2_building_data_genome" "$BDG2_BASE/data/meters/cleaned/hotwater_cleaned.csv" "data/bdg2_building_data_genome/raw/hotwater_cleaned.csv" sha256 "9714f1e803c88f7efe294480e8a32c61317e84edc65ad55ea7e9fc3282e80fce"
download_file "bdg2_building_data_genome" "$BDG2_BASE/data/meters/cleaned/gas_cleaned.csv" "data/bdg2_building_data_genome/raw/gas_cleaned.csv" sha256 "b3b059d32e8a16a92fb274e2e90483fac9b7797e42a9e07840ab9df130266405"
download_file "bdg2_building_data_genome" "$BDG2_BASE/data/meters/cleaned/water_cleaned.csv" "data/bdg2_building_data_genome/raw/water_cleaned.csv" sha256 "cf3474e7d3ca89b7e04674ef80ed6d6248473f5ccd523b1624ad73625d5fb220"
download_file "bdg2_building_data_genome" "$BDG2_BASE/data/meters/cleaned/irrigation_cleaned.csv" "data/bdg2_building_data_genome/raw/irrigation_cleaned.csv" sha256 "dc36305ba5b9a0e4b908b3e481d89ce6fc21b276e0e2a9df565f74e1b6c8e773"
download_file "bdg2_building_data_genome" "$BDG2_BASE/data/meters/cleaned/solar_cleaned.csv" "data/bdg2_building_data_genome/raw/solar_cleaned.csv" sha256 "4a419a993b084d41f7ef7014261350db889eb7bec6cf657c0dae446ad6717181"
download_file "bdg2_building_data_genome" "$BDG2_BASE/data/metadata/metadata.csv" "data/bdg2_building_data_genome/raw/metadata.csv" sha256 "992d0b29f24f96ad4332bc4dbb534b7bdd7dd2689aad093f94e93068ecddca02"

echo "== 4. Danish smart heat meters =="
download_file "danish_smart_heat_meters" \
  "https://zenodo.org/api/records/6563114/files/3_years_3021_smart_heat_meters_residential_denmark.zip/content" \
  "data/danish_smart_heat_meters/raw/3_years_3021_smart_heat_meters_residential_denmark.zip" \
  md5 "e32a9216f2d3243e7cf9f6ca5ffad685"

echo "== 5. Smart Grid Smart City =="
download_file "smart_grid_smart_city" "https://data.gov.au/data/dataset/4e21dea3-9b87-4610-94c7-15a8a77907ef/resource/b71eb954-196a-4901-82fd-69b17f88521e/download/cdintervalreadingallnoquotes.csv.7z" "data/smart_grid_smart_city/raw/cdintervalreadingallnoquotes.csv.7z"
download_file "smart_grid_smart_city" "https://data.gov.au/data/dataset/4e21dea3-9b87-4610-94c7-15a8a77907ef/resource/0404c872-8a83-40e6-9c04-88dfec125aee/download/sgsc-ct_customer-household-data-revised.csv" "data/smart_grid_smart_city/raw/sgsc_customer_household_data_revised.csv"
download_file "smart_grid_smart_city" "https://data.gov.au/data/dataset/4e21dea3-9b87-4610-94c7-15a8a77907ef/resource/c7901c40-28e2-4601-bd3b-38156428671a/download/sgsc-cthanplug-readings.7z" "data/smart_grid_smart_city/raw/sgsc-cthanplug-readings.7z"
download_file "smart_grid_smart_city" "https://data.gov.au/data/dataset/4e21dea3-9b87-4610-94c7-15a8a77907ef/resource/e45517ea-189a-4d61-b02d-bd7d85b0f156/download/sgsc-ctpeak-events.csv" "data/smart_grid_smart_city/raw/sgsc-ctpeak-events.csv"
download_file "smart_grid_smart_city" "https://data.gov.au/data/dataset/4e21dea3-9b87-4610-94c7-15a8a77907ef/resource/2ea70e4e-da2a-42d8-94dc-c12b801d45b7/download/sgsc-ctpeak-event-response.csv" "data/smart_grid_smart_city/raw/sgsc-ctpeak-event-response.csv"
download_file "smart_grid_smart_city" "https://data.gov.au/data/dataset/4e21dea3-9b87-4610-94c7-15a8a77907ef/resource/55be730d-0d3f-477b-8f47-d3fd1e89fbca/download/sgsc-ctoffers-and-acceptances.csv" "data/smart_grid_smart_city/raw/sgsc-ctoffers-and-acceptances.csv"
download_file "smart_grid_smart_city" "https://data.gov.au/data/dataset/4e21dea3-9b87-4610-94c7-15a8a77907ef/resource/52e630d3-725c-4854-81bf-5d3a4f3c0385/download/sgsc-ct_datagov_revised-data-dictionary.xlsx" "data/smart_grid_smart_city/raw/sgsc-ct_datagov_revised-data-dictionary.xlsx"

echo "== 6. HEAPO heat pumps =="
download_file "heapo_heat_pumps" \
  "https://zenodo.org/api/records/15056919/files/heapo_data.zip/content" \
  "data/heapo_heat_pumps/raw/heapo_data.zip" \
  md5 "2b92b0d52487e1a733c35ee59860450d"

echo "== 7. GoiEner smart meters =="
download_file "goiener_smart_meters" "https://zenodo.org/api/records/7362094/files/metadata.csv/content" "data/goiener_smart_meters/raw/metadata.csv" md5 "1876a8fe4f0c22080e9c7936e1b0cfde"
download_file "goiener_smart_meters" "https://zenodo.org/api/records/7362094/files/imp-post.tzst/content" "data/goiener_smart_meters/raw/imp-post.tzst" md5 "1ff85d89fee75da33756c6b7bdbf7a7e"
download_file "goiener_smart_meters" "https://zenodo.org/api/records/7362094/files/imp-in.tzst/content" "data/goiener_smart_meters/raw/imp-in.tzst" md5 "dc593839ad812c0dc2d854894097bc13"
download_file "goiener_smart_meters" "https://zenodo.org/api/records/7362094/files/imp-pre.tzst/content" "data/goiener_smart_meters/raw/imp-pre.tzst" md5 "77d549a1dd9fec438505930750e712db"
download_file "goiener_smart_meters" "https://zenodo.org/api/records/7362094/files/raw.tzst/content" "data/goiener_smart_meters/raw/raw.tzst" md5 "6af6b9f6baddf82b7a2a663186e1e1e4"

echo "== 8. European LV Urban 8087 =="
download_file "european_lv_urban_8087" \
  "https://data.mendeley.com/public-files/datasets/685vgp64sm/files/6fc11feb-0a40-4b31-8948-7f1ca4fc3636/file_downloaded" \
  "data/european_lv_urban_8087/raw/Sim_files_190128_OK_V0.zip" \
  sha256 "4ac9986becb6447450e7cc8a7a236146faf7b595ab42d56f7c3eb2c1c8fb9967"

echo "== 9. European LV Rural 2731 =="
download_file "european_lv_rural_2731" \
  "https://data.mendeley.com/public-files/datasets/gspyzvvrhm/files/4c5ab724-47b9-4ea1-9339-934d1095b06d/file_downloaded" \
  "data/european_lv_rural_2731/raw/Rural.rar" \
  sha256 "c3791f177abc2bd9d66d481b96ec1c20bc8ba9ff309885dc35f1d28cdc6bb016"
extract_rar "european_lv_rural_2731" "data/european_lv_rural_2731/raw/Rural.rar" "data/european_lv_rural_2731/processed" "data/european_lv_rural_2731/processed/PQ_csv"

echo "== 10. European LV Urban 35297 =="
download_file "european_lv_urban_35297" \
  "https://data.mendeley.com/public-files/datasets/gspyzvvrhm/files/3ed18f16-16f9-49ab-8bdd-4f5fbc74f2fa/file_downloaded" \
  "data/european_lv_urban_35297/raw/Urban.rar" \
  sha256 "0341f0ec347093b4bf19567d40ac1aef1b155954b9b19894b749dbfa0b256ad7"
extract_rar "european_lv_urban_35297" "data/european_lv_urban_35297/raw/Urban.rar" "data/european_lv_urban_35297/processed" "data/european_lv_urban_35297/processed/PQ_csv"

echo "== 11. data2 EV charging =="
download_file "data2" \
  "https://data.mendeley.com/public-files/datasets/c7gg94tmvz/files/662879c2-cd93-4fad-81c3-c2c1e5412c9e/file_downloaded" \
  "data/data2/processed_data.xlsx" \
  sha256 "e4dff6c568a0031188e964ddf63abdcfaefff15f3f0406638b7dd7b180f01ff1"
download_file "data2" \
  "https://data.mendeley.com/public-files/datasets/c7gg94tmvz/files/0bd6856e-41e5-45a4-ab75-e9f1273dff85/file_downloaded" \
  "data/data2/processed_data_longer_than_30.xlsx" \
  sha256 "8d7c676bbf2042a4cc8124d12aa5652b6a335bc43e3d4d984e819a0078e08c70"
download_file "data2" \
  "https://data.mendeley.com/public-files/datasets/c7gg94tmvz/files/77179ce8-2b20-465e-888b-901e8acbfc55/file_downloaded" \
  "data/data2/Readme.txt" \
  sha256 "0a535074de55e4d10592266fbbce588132d37f60ba5192b6b6812e0b128115b2"

echo "== 12. NextGen household batteries =="
download_zenodo_record_files "nextgen" "14885589" "data/nextgen"

echo "== 13. Norway AMI Energy Distribution =="
download_file "norway_ami_energy_distribution" \
  "https://data.mendeley.com/public-api/zip/jv3rz8k35r/download/1" \
  "data/norway_ami_energy_distribution/norway_ami_energy_distribution_v1.zip"
extract_zip "norway_ami_energy_distribution" \
  "data/norway_ami_energy_distribution/norway_ami_energy_distribution_v1.zip" \
  "data/norway_ami_energy_distribution/raw" \
  "data/norway_ami_energy_distribution/raw/Energy distribution models with AMI smart meter sensor dataset/data/ami"

echo "== 14. CAMSL Japan smart meters =="
download_file "camsl_japan_smart_meters" \
  "https://data.mendeley.com/public-files/datasets/cmpsyncmmk/files/1083e761-92db-4828-8b07-58f74eccb4d4/file_downloaded" \
  "data/camsl_japan_smart_meters/raw/public.zip" \
  sha256 "435a2a5c5b601b0d783b95542c6f3199e2b0f169b367ceea5269813403c5b2a8"
extract_zip "camsl_japan_smart_meters" \
  "data/camsl_japan_smart_meters/raw/public.zip" \
  "data/camsl_japan_smart_meters/extracted" \
  "data/camsl_japan_smart_meters/extracted/public/consumption_data/consumption_data"

if [ -f "data/camsl_japan_smart_meters/extracted/public/consumption_data.zip" ] && \
   [ ! -d "data/camsl_japan_smart_meters/extracted/public/consumption_data/consumption_data" ]; then
  mkdir -p "data/camsl_japan_smart_meters/extracted/public/consumption_data"
  unzip -q -n "data/camsl_japan_smart_meters/extracted/public/consumption_data.zip" \
    -d "data/camsl_japan_smart_meters/extracted/public/consumption_data"
fi

echo "== 15. Irish domestic smart meters =="
if ! download_file_optional "irish_domestic_smart_meters" \
  "https://ndownloader.figshare.com/files/63098560" \
  "data/irish_domestic_smart_meters/SMData2.0.zip" \
  sha256 "d8b26bf3990d8605f1255c5b5e4d28af08b5c42b0e549d1eb0d8d52c5987e86c"; then
  echo "download the Irish domestic smart meters manually:"
  echo "  page: https://figshare.com/articles/dataset/31851922/2"
  echo "  file: https://ndownloader.figshare.com/files/63098560"
  echo "  save as: data/irish_domestic_smart_meters/SMData2.0.zip"
fi
if [ -s "data/irish_domestic_smart_meters/SMData2.0.zip" ]; then
  extract_zip "irish_domestic_smart_meters" \
    "data/irish_domestic_smart_meters/SMData2.0.zip" \
    "data/irish_domestic_smart_meters/raw" \
    "data/irish_domestic_smart_meters/raw/SM Data 2.0"
fi

echo "== 16. OPSD Household Data =="
download_file "opsd_household_data" \
  "https://data.open-power-system-data.org/household_data/opsd-household_data-2020-04-15.zip" \
  "data/opsd_household_data/opsd-household-data-2020-04-15.zip" \
  sha256 "8b27465beb20e4cd1eab579e3967a28213e592a07a30ec31c511a68d48844e23"
extract_zip "opsd_household_data" \
  "data/opsd_household_data/opsd-household-data-2020-04-15.zip" \
  "data/opsd_household_data/raw" \
  "data/opsd_household_data/raw/opsd-household_data-2020-04-15/household_data_15min_singleindex.csv"

echo "== 17. Complete Energy Community =="
download_file "complete_energy_community" \
  "https://zenodo.org/api/records/7602546/files/EC_EV_dataset.xlsx/content" \
  "data/complete_energy_community/EC_EV_dataset.xlsx" \
  md5 "a4184b98906dbca2169f66d8c6842b0e"
download_file "complete_energy_community" \
  "https://zenodo.org/api/records/7602546/files/General%20description.docx/content" \
  "data/complete_energy_community/General description.docx" \
  md5 "e28608414152fcdad3a21c6e4bfe3a36"

echo "== final path check =="
"$PYTHON_BIN" - <<'PY'
from pathlib import Path

checks = {
    "bdg1_building_data_genome": Path("data/bdg1_building_data_genome/raw/temp_open_utc_complete.csv"),
    "low_carbon_london": Path("data/low_carbon_london/raw/LCL-FullData.zip"),
    "bdg2_building_data_genome": Path("data/bdg2_building_data_genome/raw/electricity_cleaned.csv"),
    "danish_smart_heat_meters": Path("data/danish_smart_heat_meters/raw/3_years_3021_smart_heat_meters_residential_denmark.zip"),
    "smart_grid_smart_city": Path("data/smart_grid_smart_city/raw/cdintervalreadingallnoquotes.csv.7z"),
    "heapo_heat_pumps": Path("data/heapo_heat_pumps/raw/heapo_data.zip"),
    "goiener_smart_meters": Path("data/goiener_smart_meters/raw/imp-post.tzst"),
    "european_lv_urban_8087": Path("data/european_lv_urban_8087/raw/Sim_files_190128_OK_V0.zip"),
    "european_lv_rural_2731": Path("data/european_lv_rural_2731/processed/PQ_csv"),
    "european_lv_urban_35297": Path("data/european_lv_urban_35297/processed/PQ_csv"),
    "data2": Path("data/data2/processed_data_longer_than_30.xlsx"),
    "nextgen": Path("data/nextgen"),
    "norway_ami_energy_distribution": Path("data/norway_ami_energy_distribution/raw/Energy distribution models with AMI smart meter sensor dataset/data/ami"),
    "camsl_japan_smart_meters": Path("data/camsl_japan_smart_meters/extracted/public/consumption_data/consumption_data"),
    "irish_domestic_smart_meters": Path("data/irish_domestic_smart_meters/raw/SM Data 2.0"),
    "opsd_household_data": Path("data/opsd_household_data/raw/opsd-household_data-2020-04-15/household_data_15min_singleindex.csv"),
    "complete_energy_community": Path("data/complete_energy_community/EC_EV_dataset.xlsx"),
}
missing = []
for name, path in checks.items():
    ok = path.exists()
    if path.is_dir():
        ok = ok and any(path.iterdir())
    print(f"{name}: {'OK' if ok else 'MISSING'} -> {path}")
    if not ok:
        missing.append(name)
if missing:
    print("warning: the following datasets still need manual handling or a re-download: " + ", ".join(missing))
else:
    print("all paths required by the project are present.")
PY

echo "done. Next: python -m src.extra.dataset_experiment.run --dataset smart_grid_smart_city --preflight-only"
