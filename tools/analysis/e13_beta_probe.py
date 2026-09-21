import csv, json, math, os, statistics as st

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "supp")
LOGICS = ["open_loop", "soc_threshold", "probabilistic", "price_response",
          "random_delay", "hybrid"]


def read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def fnum(row, key):
    v = row.get(key, "")
    if v in ("", "None", "nan", "NaN"):
        return None
    try:
        x = float(v)
    except ValueError:
        return None
    return None if math.isnan(x) else x


def slope(xs, ys):
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len(pts) < 2:
        return None
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    den = sum((p[0] - mx) ** 2 for p in pts)
    if den == 0:
        return None
    return sum((p[0] - mx) * (p[1] - my) for p in pts) / den


summ = read_csv(os.path.join(D, "E13", "summary.csv"))
dsum = read_csv(os.path.join(D, "E13", "dataset_summary.csv"))
datasets = sorted({r["dataset"] for r in dsum})

print("=" * 78)
print("1. is the N grid identical across control rules")
print("=" * 78)
bad_grid = []
for ds in datasets:
    grids = {r["logic"]: r["N_values"] for r in dsum if r["dataset"] == ds}
    uniq = set(grids.values())
    if len(uniq) != 1:
        bad_grid.append((ds, grids))
print(f"datasets whose N grid differs across the 6 arms: {len(bad_grid)}")
for ds, g in bad_grid:
    print("  ", ds, g)
ex = [r for r in dsum if r["dataset"] == datasets[0]][0]
print(f"  example: {datasets[0]} -> N = {ex['N_values']}")

print()
print("=" * 78)
print("2. refitted beta vs the tabulated beta_data_coupled (data_coupled arm)")
print("=" * 78)
print(f"{'logic':<16}{'tabulated':>10}{'refitted':>10}{'max_diff':>12}{'CV(minN)':>11}{'CV(maxN)':>11}{'CV_drop':>9}")
refit = {lg: [] for lg in LOGICS}
cv_lo = {lg: [] for lg in LOGICS}
cv_hi = {lg: [] for lg in LOGICS}
half = {lg: ([], []) for lg in LOGICS}
for lg in LOGICS:
    diffs = []
    tabled = []
    for ds in datasets:
        rows = sorted(
            [r for r in summ
             if r["dataset"] == ds and r["logic"] == lg and r["arm"] == "data_coupled"],
            key=lambda r: float(r["N"]),
        )
        xs = [fnum(r, "N") for r in rows]
        ys = [fnum(r, "condition_cv") for r in rows]
        pair = [(x, y) for x, y in zip(xs, ys) if x and y]
        if len(pair) < 2:
            continue
        xs = [p[0] for p in pair]
        ys = [p[1] for p in pair]
        s = slope(xs, ys)
        refit[lg].append(s)
        cv_lo[lg].append(ys[0])
        cv_hi[lg].append(ys[-1])
        mid = len(xs) // 2 + 1
        s1, s2 = slope(xs[:mid], ys[:mid]), slope(xs[mid - 1:], ys[mid - 1:])
        if s1 is not None:
            half[lg][0].append(s1)
        if s2 is not None:
            half[lg][1].append(s2)
        t = [fnum(r, "beta_data_coupled") for r in dsum
             if r["dataset"] == ds and r["logic"] == lg]
        if t and t[0] is not None:
            tabled.append(t[0])
            diffs.append(abs(t[0] - s))
    drop = [h / l for l, h in zip(cv_lo[lg], cv_hi[lg]) if l]
    print(f"{lg:<16}{st.median(tabled):>10.4f}{st.median(refit[lg]):>10.4f}"
          f"{max(diffs):>12.2e}{st.median(cv_lo[lg]):>11.4f}"
          f"{st.median(cv_hi[lg]):>11.4f}{st.median(drop):>9.3f}")

print()
print("=" * 78)
print("3. curvature: slope of the lower half vs the upper half (equal => a genuinely steeper power law)")
print("=" * 78)
print(f"{'logic':<16}{'lower_half':>12}{'upper_half':>12}{'full_range':>11}{'upper-lower':>9}")
for lg in LOGICS:
    a, b = st.median(half[lg][0]), st.median(half[lg][1])
    print(f"{lg:<16}{a:>12.4f}{b:>12.4f}{st.median(refit[lg]):>11.4f}{b - a:>9.4f}")

print()
print("=" * 78)
print("4. consistency between the bootstrap CI of beta and the point estimate")
print("=" * 78)
print(f"{'logic':<16}{'beta':>10}{'CI_low':>12}{'CI_high':>12}{'CI_covers':>12}")
for lg in LOGICS:
    bs, los, his, inside = [], [], [], 0
    tot = 0
    for r in dsum:
        if r["logic"] != lg:
            continue
        b = fnum(r, "beta_data_coupled")
        lo = fnum(r, "beta_data_coupled_ci_lower")
        hi = fnum(r, "beta_data_coupled_ci_upper")
        if None in (b, lo, hi):
            continue
        tot += 1
        bs.append(b)
        los.append(lo)
        his.append(hi)
        if lo <= b <= hi:
            inside += 1
    print(f"{lg:<16}{st.median(bs):>10.4f}{st.median(los):>12.4f}"
          f"{st.median(his):>12.4f}{f'{inside}/{tot}':>12}")

print()
print("=" * 78)
print("5. beta of the decoupled arm (synthetic control): it should stay at -1/2")
print("=" * 78)
print(f"{'logic':<16}{'beta_decoupled':>20}{'beta_coupled':>20}")
for lg in LOGICS:
    d = [fnum(r, "beta_decoupled") for r in dsum if r["logic"] == lg]
    c = [fnum(r, "beta_data_coupled") for r in dsum if r["logic"] == lg]
    d = [x for x in d if x is not None]
    c = [x for x in c if x is not None]
    print(f"{lg:<16}{st.median(d):>20.4f}{st.median(c):>20.4f}")
