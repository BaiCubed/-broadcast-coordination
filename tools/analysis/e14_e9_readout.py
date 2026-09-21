import csv, json, math, os, statistics as st

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "supp")


def jload(*p):
    with open(os.path.join(D, *p)) as fh:
        return json.load(fh)


def fnum(r, k):
    v = r.get(k, "")
    if v in ("", "None", "nan", "NaN"):
        return None
    try:
        x = float(v)
    except ValueError:
        return None
    return None if math.isnan(x) else x


ac = jload("E14", "axis_collapse.json")
by = ac["by_dataset"] if "by_dataset" in ac else ac.get("axis_collapse", {}).get("by_dataset")
print("=" * 88)
print("E14 (1) axis collapse, pooled over all capacity configurations: the only comparable axis readout")
print("=" * 88)
allr = [d["variance_ratio_capacity_over_N"] for d in by]
matched = [d["variance_ratio_capacity_over_N"] for d in by if d.get("comparison_is_matched")]
unmatched = [d["dataset"] for d in by if not d.get("comparison_is_matched")]
coll = sum(1 for d in by if d.get("collapses"))
print(f"datasets: {len(by)}; collapses (ratio<1) {coll}/{len(by)}; matched {len(matched)}/{len(by)}")
print(f"median ratio over all 15 = {st.median(allr):.4f}   range [{min(allr):.4f}, {max(allr):.4f}]")
print(f"median over the 12 matched = {st.median(matched):.4f}   range [{min(matched):.4f}, {max(matched):.4f}]")
print(f"the 3 unmatched cases, where the guard forbids reporting a ratio: {', '.join(unmatched)}")
print("  -> for these three the physical N axis has zero within-bin spread, so the comparison is not like for like")

li = jload("E14", "lindeberg_summary.json")
tbl = li["by_injection"] if "by_injection" in li else li.get("lindeberg_summary", {}).get("by_injection")
print()
print("=" * 88)
print("E14 (2) per injection level: beta on the physical N axis only; the per-level capacity-axis beta fails the guard and is not reported")
print("=" * 88)
print(f"{'injection level':<20}{'beta_vs_N':>11}{'beta_decoup':>13}{'max share':>12}{'A_ctrl':>9}")
order = ["count=0,ratio=1", "count=1,ratio=10", "count=1,ratio=100",
         "count=1,ratio=1000", "count=5,ratio=100", "count=20,ratio=100"]
for k in order:
    v = tbl[k]
    print(f"{k:<20}{v['beta_vs_N_median']:>11.4f}{v['beta_decoupled_vs_N_median']:>11.4f}"
          f"{v['lindeberg_share_max_median']:>18.4f}{v['A_ctrl_median']:>9.4f}")

print()
print("=" * 88)
print("E9 long-horizon closed-loop SOC (30 days)")
print("=" * 88)
sd = jload("E9", "soc_drift_summary.json")
print(json.dumps(sd, ensure_ascii=False, indent=2)[:2600])
