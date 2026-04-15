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
    path = DATA_DIR / "random_interior_generalization.csv"
    rows = _load_rows(path)
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    if not rows:
        watermark_placeholder(ax, "MISSING\nrandom interior results")
        ax.set_axis_off()
        out_paths = save_figure(fig, "fig_random_interior_generalization_placeholder")
    else:
        case_ids = [r["case_id"].replace("random_interior_", "r") for r in rows]
        top = np.array([float(r["top_uz_mae_pct"]) for r in rows])
        vol = np.array([float(r["volume_mae_pct"]) for r in rows])
        x = np.arange(len(rows))
        width = 0.38
        ax.bar(x - width / 2, top, width, label="top $u_z$")
        ax.bar(x + width / 2, vol, width, label="volume")
        ax.axhline(5.0, color="0.2", linestyle="--", linewidth=1.0, label="5% target")
        ax.set_xticks(x)
        ax.set_xticklabels(case_ids, rotation=45, ha="right")
        ax.set_ylabel("MAE (% of FEM max)")
        ax.set_title("Random-Interior Generalization")
        ax.legend()
        out_paths = save_figure(fig, "fig_random_interior_generalization")
    plt.close(fig)
    print("Wrote:")
    for p in out_paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
