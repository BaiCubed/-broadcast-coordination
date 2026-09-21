import sys, yaml
from pathlib import Path
cfg = Path("src/extra/nc_excel_experiments/configs")
def derive(base: str, name: str, root: str, workers: int | None = None):
    payload = yaml.safe_load((cfg / base).read_text(encoding="utf-8"))
    payload["results_root"] = root
    if workers:
        payload["execution"]["dataset_workers"] = workers
    out = cfg / name
    out.write_text(yaml.safe_dump(payload, sort_keys=True, allow_unicode=True), encoding="utf-8")
    print("wrote", out, "->", root)

for topo in ("ieee33", "ieee69", "ieee123"):
    derive("e1_smoke.yaml", f"e1_idcheck_{topo}.yaml", f"results/e1_idcheck_{topo}")
derive("e1_smoke.yaml", "e1_idcheck_published.yaml", "results/e1_idcheck_published")
for topo in ("ieee33", "ieee69", "ieee123"):
    derive("e1_full_v2.yaml", f"e1_topo_{topo}_fb.yaml", f"results/e1_topo_{topo}_fb", 15)
