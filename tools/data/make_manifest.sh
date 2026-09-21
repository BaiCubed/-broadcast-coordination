#!/usr/bin/env bash
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${NC_ROOT:-$(cd "$HERE/../.." && pwd)}"
PYTHON_BIN="$(command -v python3)"
download_file(){ printf "%s\t%s\t%s\t%s\t%s\n" "$1" "$2" "$3" "${4:-none}" "${5:-}"; }
download_file_optional(){ printf "%s\t%s\t%s\t%s\t%s\n" "$1" "$2" "$3" "${4:-none}" "${5:-}"; return 0; }
extract_zip(){ :; }
extract_rar(){ :; }
download_zenodo_record_files(){
  "$PYTHON_BIN" - "$1" "$2" "$3" <<"PY"
import json,sys,urllib.request
ds,rec,dest=sys.argv[1:4]
r=urllib.request.Request(f"https://zenodo.org/api/records/{rec}",headers={"User-Agent":"bc-data"})
d=json.load(urllib.request.urlopen(r,timeout=60))
for it in d.get("files",[]):
    k=it["key"]; c=it.get("checksum","") or ""
    link=it.get("links",{}).get("self") or it.get("links",{}).get("download")
    if not link: continue
    algo,val=(c.split(":",1) if ":" in c else ("none",""))
    print(f"{ds}\t{link}\t{dest}/{k}\t{algo}\t{val}")
PY
}
echo(){ :; }
source "$HERE/_calls.sh"
