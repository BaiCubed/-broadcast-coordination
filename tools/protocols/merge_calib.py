import json
from pathlib import Path
root = Path("results")
merged = {}
parts = sorted(root.glob("_topology_calibration.*.json"))
for part in parts:
    merged.update(json.loads(part.read_text(encoding="utf-8")))
(root / "_topology_calibration.json").write_text(
    json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
print("parts", len(parts), "keys", len(merged))
rows = {}
for k, v in merged.items():
    rows.setdefault(v["dataset"], {})[v["topology"]] = v["network_capacity_multiplier"]
print(f"{'dataset':<32}{'ieee33':>12}{'ieee69':>12}{'ieee123':>12}")
for ds in sorted(rows):
    r = rows[ds]
    print(f"{ds:<32}{r.get('ieee33',float('nan')):>12.3f}{r.get('ieee69',float('nan')):>12.3f}{r.get('ieee123',float('nan')):>12.3f}")
