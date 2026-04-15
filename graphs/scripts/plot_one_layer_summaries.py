from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _common import DATA_DIR, apply_ieee_style, save_figure, watermark_placeholder


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _plot_generalization() -> list[Path]:
    rows = _rows(DATA_DIR / "one_layer_random_generalization.csv")
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    if not rows:
        watermark_placeholder(ax, "MISSING\none-layer generalization")
        ax.set_axis_off()
        out = save_figure(fig, "fig_one_layer_generalization_placeholder")
    else:
        labels = [r["case_id"].replace("one_layer_random_", "r") for r in rows]
        top = np.array([float(r["top_uz_mae_pct"]) for r in rows], dtype=float)
        vol = np.array([float(r["volume_mae_pct"]) for r in rows], dtype=float)
        x = np.arange(len(rows))
        width = 0.38
        ax.bar(x - width / 2, top, width, label="top $u_z$")
        ax.bar(x + width / 2, vol, width, label="volume")
        ax.axhline(5.0, color="0.2", linestyle="--", linewidth=1.0, label="5% target")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("MAE (% of FEM max)")
        ax.set_title("One-Layer Random-Interior Generalization")
        ax.legend()
        out = save_figure(fig, "fig_one_layer_generalization")
    plt.close(fig)
    return out


def _plot_ablation() -> list[Path]:
    rows = _rows(DATA_DIR / "one_layer_ablation_results.csv")
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    if not rows:
        watermark_placeholder(ax, "MISSING\none-layer ablation")
        ax.set_axis_off()
        out = save_figure(fig, "fig_one_layer_ablation_placeholder")
    else:
        cell_text = []
        for row in rows:
            mean = row.get("mean_top_uz_mae_pct") or row.get("status", "")
            worst = row.get("worst_top_uz_mae_pct") or row.get("status", "")
            try:
                mean = f"{float(mean):.2f}"
                worst = f"{float(worst):.2f}"
            except Exception:
                pass
            cell_text.append([row["variant"], row["status"], mean, worst])
        ax.axis("off")
        table = ax.table(
            cellText=cell_text,
            colLabels=["Variant", "Status", "Mean Top MAE", "Worst Top MAE"],
            loc="center",
            cellLoc="left",
            colLoc="left",
            colWidths=[0.38, 0.28, 0.17, 0.17],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.5)
        table.scale(1.0, 1.25)
        for (r, _c), cell in table.get_celld().items():
            cell.set_linewidth(0.5)
            if r == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor("0.95")
        ax.set_title("One-Layer Ablation / Applicability")
        out = save_figure(fig, "fig_one_layer_ablation")
    plt.close(fig)
    return out


def _plot_unsupervised() -> list[Path]:
    rows = _rows(DATA_DIR / "one_layer_unsupervised_region_comparison.csv")
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    if not rows:
        watermark_placeholder(ax, "MISSING\none-layer unsupervised")
        ax.set_axis_off()
        out = save_figure(fig, "fig_one_layer_unsupervised_placeholder")
    else:
        labels = [f"{r['case_id']}\n{r['spatial_region']}" for r in rows]
        vals = np.array([float(r["mae_pct"]) for r in rows], dtype=float)
        ax.bar(np.arange(len(rows)), vals, color="0.45")
        ax.axhline(5.0, color="0.2", linestyle="--", linewidth=1.0)
        ax.set_xticks(np.arange(len(rows)))
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=6.5)
        ax.set_ylabel("MAE (% of FEM max)")
        ax.set_title("One-Layer FEM-Unsupervised Regions")
        out = save_figure(fig, "fig_one_layer_unsupervised")
    plt.close(fig)
    return out


def _plot_efficiency() -> list[Path]:
    rows = _rows(DATA_DIR / "one_layer_efficiency_timing.csv")
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    if not rows:
        watermark_placeholder(ax, "MISSING\none-layer timing")
        ax.set_axis_off()
        out = save_figure(fig, "fig_one_layer_efficiency_placeholder")
    else:
        fem = np.array([float(r["fem_seconds"]) for r in rows], dtype=float)
        pinn = np.array([float(r["pinn_eval_seconds"]) for r in rows], dtype=float)
        ax.boxplot([fem, pinn], tick_labels=["FEM solve", "PINN eval"], showfliers=True)
        ax.set_yscale("log")
        ax.set_ylabel("seconds per case")
        ax.set_title("One-Layer Per-Case Cost")
        out = save_figure(fig, "fig_one_layer_efficiency")
    plt.close(fig)
    return out


def main() -> None:
    apply_ieee_style()
    paths = []
    paths.extend(_plot_generalization())
    paths.extend(_plot_ablation())
    paths.extend(_plot_unsupervised())
    paths.extend(_plot_efficiency())
    print("Wrote:")
    for path in paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
