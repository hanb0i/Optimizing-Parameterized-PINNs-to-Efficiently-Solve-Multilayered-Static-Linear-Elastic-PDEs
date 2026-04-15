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
    path = DATA_DIR / "efficiency_timing.csv"
    rows = _load_rows(path)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    if not rows:
        watermark_placeholder(ax, "MISSING\nefficiency timing")
        ax.set_axis_off()
        out_paths = save_figure(fig, "fig_efficiency_timing_placeholder")
    else:
        fem = np.array([float(r["fem_seconds"]) for r in rows])
        pinn = np.array([float(r["pinn_eval_seconds"]) for r in rows])
        ax.boxplot([fem, pinn], tick_labels=["FEM solve", "PINN eval"], showfliers=True)
        ax.set_yscale("log")
        ax.set_ylabel("seconds per case")
        ax.set_title("Per-Case Cost After Training")
        out_paths = save_figure(fig, "fig_efficiency_timing")
    plt.close(fig)
    print("Wrote:")
    for p in out_paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
