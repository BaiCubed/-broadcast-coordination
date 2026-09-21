#!/usr/bin/env python3
from __future__ import annotations
import concurrent.futures as cf
import hashlib, os, subprocess, sys, threading
from pathlib import Path

ROOT = Path(os.environ.get("NC_ROOT", Path(__file__).resolve().parents[2]))
MANIFEST = Path(os.environ.get("NC_MANIFEST", ROOT / "data" / "manifest.tsv"))
WORKERS = int(os.environ.get("WORKERS", "8"))
lock = threading.Lock()

def say(msg: str) -> None:
    with lock:
        print(msg, flush=True)

def digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()

def fetch(entry):
    dataset, url, dest, algo, expected = entry
    target = ROOT / dest
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        if algo in {"md5", "sha256"} and expected:
            if digest(target, algo) == expected:
                say(f"SKIP  {dataset:32s} {dest}")
                return (dataset, dest, "skip", "")
            say(f"REDO  {dataset:32s} {dest} (checksum mismatch)")
            target.unlink()
        else:
            say(f"SKIP  {dataset:32s} {dest}")
            return (dataset, dest, "skip", "")
    part = target.with_suffix(target.suffix + ".part")
    cmd = ["wget", "-c", "-q", "--tries=5", "--timeout=60", "--waitretry=10",
           "-O", str(part), url]
    say(f"GET   {dataset:32s} {dest}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not part.exists() or part.stat().st_size == 0:
        say(f"FAIL  {dataset:32s} {dest} rc={proc.returncode} {proc.stderr.strip()[:200]}")
        return (dataset, dest, "fail", proc.stderr.strip()[:300])
    if algo in {"md5", "sha256"} and expected:
        actual = digest(part, algo)
        if actual != expected:
            say(f"BAD   {dataset:32s} {dest} {algo} expected={expected} actual={actual}")
            return (dataset, dest, "checksum_mismatch", actual)
    part.rename(target)
    say(f"OK    {dataset:32s} {dest} ({target.stat().st_size/1e6:.1f} MB)")
    return (dataset, dest, "ok", "")

def main() -> int:
    entries = []
    for line in MANIFEST.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        while len(parts) < 5:
            parts.append("")
        entries.append(tuple(parts[:5]))
    order = {"goiener_smart_meters": 0, "smart_grid_smart_city": 1, "low_carbon_london": 2,
             "norway_ami_energy_distribution": 3, "heapo_heat_pumps": 4, "bdg2_building_data_genome": 5}
    entries.sort(key=lambda e: order.get(e[0], 9))
    results = []
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for result in pool.map(fetch, entries):
            results.append(result)
    bad = [r for r in results if r[2] not in {"ok", "skip"}]
    say("=" * 70)
    say(f"TOTAL={len(results)} ok={sum(r[2]=='ok' for r in results)} "
        f"skip={sum(r[2]=='skip' for r in results)} problem={len(bad)}")
    for r in bad:
        say(f"PROBLEM {r[0]} {r[1]} {r[2]} {r[3]}")
    return 0
if __name__ == "__main__":
    sys.exit(main())
