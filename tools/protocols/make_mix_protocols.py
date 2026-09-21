import yaml
from pathlib import Path
cfg = Path("src/extra/nc_excel_experiments/configs")
base = yaml.safe_load((cfg / "e1_mix_s00.yaml").read_text(encoding="utf-8"))
for topo in ("ieee33", "ieee69", "ieee123"):
    p = dict(base)
    p["results_root"] = f"results/e1_mix_s00_{topo}_fb"
    p["execution"] = dict(base["execution"]); p["execution"]["dataset_workers"] = 40
    out = cfg / f"e1_mix_s00_{topo}_fb.yaml"
    out.write_text(yaml.safe_dump(p, sort_keys=True, allow_unicode=True), encoding="utf-8")
    print("wrote", out.name, "->", p["results_root"])
