import yaml
from pathlib import Path
cfg = Path("src/extra/nc_excel_experiments/configs")
def derive(base, name, root, workers=None):
    p = yaml.safe_load((cfg / base).read_text(encoding="utf-8"))
    p["results_root"] = root
    if workers: p["execution"]["dataset_workers"] = workers
    (cfg / name).write_text(yaml.safe_dump(p, sort_keys=True, allow_unicode=True), encoding="utf-8")
    print("wrote", name, "->", root)
for topo in ("ieee33", "ieee69", "ieee123"):
    derive("e1_smallN.yaml", f"e1_smallN_{topo}_fb.yaml", f"results/e1_smallN_{topo}_fb", 15)
    derive("e1_full_v2.yaml", f"e1_topo_{topo}_bypass.yaml", f"results/e1_topo_{topo}_bypass", 15)
    derive("e1_smallN.yaml", f"e1_smallN_{topo}_bypass.yaml", f"results/e1_smallN_{topo}_bypass", 15)
