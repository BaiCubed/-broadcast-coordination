import pandas as pd, json
out = {}
for t in ("ieee33", "ieee69", "ieee123"):
    d = pd.read_csv(f"results/r2_{t}_fb/n95_table_paper.csv")
    r = d.dropna(subset=["N95"])
    out[t] = {
        "median_N95": round(float(r["N95"].median()), 1),
        "min_N95": round(float(r["N95"].min()), 1),
        "max_N95": round(float(r["N95"].max()), 1),
        "reached": f"{len(r)}/{len(d)}",
    }
print(json.dumps(out, indent=2))
print()
print(json.dumps(json.load(open("results/figs_topo/figure_E1_topology_compare_stats.json"))["IEEE-33"], indent=2, ensure_ascii=False))
