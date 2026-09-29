import yaml
from pathlib import Path
cfg = Path("src/extra/population_experiments/configs")
base = yaml.safe_load((cfg / "E1_population_scale_mixed_s00.yaml").read_text(encoding="utf-8"))
for topo in ("ieee33", "ieee69", "ieee123"):
    p = dict(base)
    p["results_root"] = f"results/E1_feeder_topology/{topo}"
    p["execution"] = dict(base["execution"]); p["execution"]["dataset_workers"] = 40
    out = cfg / f"E1_feeder_topology_{topo}.yaml"
    out.write_text(yaml.safe_dump(p, sort_keys=True, allow_unicode=True), encoding="utf-8")
    print("wrote", out.name, "->", p["results_root"])
