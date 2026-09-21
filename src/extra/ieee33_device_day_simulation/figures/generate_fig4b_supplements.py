from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


FIG4_NAME = "fig4_robustness_generalization.png"
OUT_NAME = "fig4b_curtailment_scaling_supplement.png"
MANIFEST_NAME = "fig4b_supplement_manifest.json"
LEGACY_NSCALING = Path("legacy_compat/result2_curtailment_3000dev/data/n_scaling_curtailment.json")
DIRECT_NSCALING = Path("data/n_scaling_curtailment.json")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = data.get("results", data.get("scenarios", []))
    if isinstance(rows, dict):
        rows = [{"name": key, **value} for key, value in rows.items()]
    return sorted(list(rows), key=lambda row: float(row["N"]))


def _valid_source(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        rows = _rows(_read_json(path))
    except Exception:
        return False
    return bool(rows) and all("N" in row for row in rows)


def _direct_source(root: Path) -> Path | None:
    for candidate in (root / LEGACY_NSCALING, root / DIRECT_NSCALING):
        if _valid_source(candidate):
            return candidate
    return None


def _source_candidates(root: Path) -> list[Path]:
    candidates: list[Path] = []
    direct = _direct_source(root)
    if direct is not None:
        candidates.append(direct)

    for base in (root, root.parent, root.parent.parent if root.parent != root else root.parent):
        if not base.exists():
            continue
        for path in base.rglob("n_scaling_curtailment.json"):
            if path not in candidates and _valid_source(path):
                candidates.append(path)

    return candidates


def _common_prefix_score(root: Path, candidate: Path) -> int:
    root_parts = root.resolve().parts
    cand_parts = candidate.resolve().parts
    score = 0
    for left, right in zip(root_parts, cand_parts):
        if left != right:
            break
        score += 1
    if root in candidate.parents:
        score += 100
    if root.parent in candidate.parents:
        score += 20
    return score


def _choose_source(root: Path) -> Path | None:
    candidates = _source_candidates(root)
    if not candidates:
        return None
    return max(candidates, key=lambda path: (_common_prefix_score(root, path), -len(path.parts)))


def _dataset_label(root: Path, results_root: Path) -> str:
    try:
        relative = root.relative_to(results_root)
    except ValueError:
        relative = root
    parts = relative.parts
    if not parts:
        return root.name
    if len(parts) >= 2 and parts[0] in {"experiments", "real_data_simulation"}:
        return "/".join(parts[:2])
    return "/".join(parts[: min(3, len(parts))])


def plot_fig4b(source: Path, output: Path, *, title: str) -> None:
    data = _read_json(source)
    rows = _rows(data)
    n = np.asarray([float(row["N"]) for row in rows], dtype=float)
    mean = np.asarray(
        [float(row.get("reduction_pct_mean", row.get("reduction_pct", np.nan))) for row in rows],
        dtype=float,
    )
    lo = np.asarray(
        [float(row.get("reduction_pct_ci95_lo", np.nan)) for row in rows],
        dtype=float,
    )
    hi = np.asarray(
        [float(row.get("reduction_pct_ci95_hi", np.nan)) for row in rows],
        dtype=float,
    )
    std = np.asarray(
        [float(row.get("reduction_pct_std", np.nan)) for row in rows],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(6.0, 4.2), constrained_layout=True)
    ax.fill_between(n, 0, mean, color="#6ec6bd", alpha=0.18)
    valid_ci = np.isfinite(lo) & np.isfinite(hi)
    if np.any(valid_ci):
        ax.fill_between(n[valid_ci], lo[valid_ci], hi[valid_ci], color="#069c8f", alpha=0.16, label="95% interval")
    elif np.any(np.isfinite(std)):
        lower = np.clip(mean - std, 0, None)
        upper = np.clip(mean + std, 0, 100)
        valid_std = np.isfinite(std)
        ax.fill_between(n[valid_std], lower[valid_std], upper[valid_std], color="#069c8f", alpha=0.16, label="±1 SD")

    ax.plot(n, mean, "o-", color="#069c8f", linewidth=2.4, markerfacecolor="white", markeredgewidth=2)
    for x_value, y_value in zip(n, mean):
        ax.text(x_value, min(y_value + 3.0, 104.0), f"{y_value:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xscale("log")
    ax.set_xticks(n)
    ax.set_xticklabels([str(int(value)) if value < 1000 else f"{value / 1000:.0f}k" for value in n], rotation=35, ha="right")
    ax.set_ylim(0, 108)
    ax.set_xlabel("Number of devices (N)")
    ax.set_ylabel("Curtailment reduction (%)")
    ax.set_title(title)
    ax.grid(alpha=0.25, axis="y")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8, loc="lower right")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_all(results_root: Path) -> dict[str, Any]:
    fig4_paths = sorted(results_root.rglob(f"Figs/{FIG4_NAME}"))
    generated: list[dict[str, str]] = []
    missing: list[str] = []

    for fig4_path in fig4_paths:
        result_root = fig4_path.parent.parent
        source = _choose_source(result_root)
        if source is None:
            missing.append(str(result_root))
            continue
        output = fig4_path.parent / OUT_NAME
        plot_fig4b(source, output, title=f"Figure 4B supplement: {_dataset_label(result_root, results_root)}")
        paper_fig4 = result_root / "paper_figures" / FIG4_NAME
        paper_output = None
        if paper_fig4.exists():
            paper_output = paper_fig4.parent / OUT_NAME
            paper_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output, paper_output)
        generated.append(
            {
                "result_root": str(result_root),
                "fig4": str(fig4_path),
                "fig4b_supplement": str(output),
                "paper_fig4b_supplement": str(paper_output) if paper_output else "",
                "source": str(source),
            }
        )

    manifest = {
        "figure": "Figure 4B supplement",
        "output_name": OUT_NAME,
        "generated_count": len(generated),
        "missing_count": len(missing),
        "generated": generated,
        "missing": missing,
    }
    manifest_path = results_root / MANIFEST_NAME
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate standalone Figure 4B supplements beside every existing Figure 4.")
    parser.add_argument("--results-root", default="results")
    args = parser.parse_args()
    manifest = generate_all(Path(args.results_root))
    print(f"generated={manifest['generated_count']} missing={manifest['missing_count']}")
    print(Path(args.results_root) / MANIFEST_NAME)


if __name__ == "__main__":
    main()
