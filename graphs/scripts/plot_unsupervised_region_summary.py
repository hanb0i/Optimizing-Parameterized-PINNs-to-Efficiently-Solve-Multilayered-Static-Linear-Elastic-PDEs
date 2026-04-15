from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _common import DATA_DIR, apply_ieee_style, save_figure, watermark_placeholder


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    apply_ieee_style()
    path = DATA_DIR / "unsupervised_region_comparison.csv"
    rows = _load_rows(path)
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    if not rows:
        watermark_placeholder(ax, "MISSING\nunsupervised-region results")
        ax.set_axis_off()
        out_paths = save_figure(fig, "fig_unsupervised_region_summary_placeholder")
    else:
        labels = [f"{r['case_id']}\n{r['spatial_region'].replace('_', ' ')}" for r in rows]
        vals = np.array([float(r["mae_pct"]) for r in rows], dtype=float)
        colors = ["0.35" if "unsupervised" in r["spatial_region"] or "unsupervised" in r["parameter_region"] else "0.65" for r in rows]
        ax.bar(np.arange(len(rows)), vals, color=colors)
        ax.axhline(5.0, color="0.2", linestyle="--", linewidth=1.0)
        ax.set_xticks(np.arange(len(rows)))
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=6.5)
        ax.set_ylabel("MAE (% of FEM max)")
        ax.set_title("Supervised vs Unsupervised Evaluation Regions")
        out_paths = save_figure(fig, "fig_unsupervised_region_summary")
    plt.close(fig)
    print("Wrote:")
    for p in out_paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
