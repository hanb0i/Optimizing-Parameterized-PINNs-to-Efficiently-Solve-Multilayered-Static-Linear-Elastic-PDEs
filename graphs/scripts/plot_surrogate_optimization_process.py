from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import DATA_DIR, apply_ieee_style, save_figure, watermark_placeholder


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    apply_ieee_style()
    candidates_path = DATA_DIR / "surrogate_optimization_mean_patch_candidates.csv"
    confirmation_path = DATA_DIR / "surrogate_optimization_mean_patch_confirmation.csv"
    candidate_rows = _load_rows(candidates_path)
    confirmation_rows = _load_rows(confirmation_path)

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2))
    ax0, ax1, ax2 = axes

    if not candidate_rows or not confirmation_rows:
        for ax in axes:
            watermark_placeholder(ax, "MISSING\noptimization process")
            ax.set_axis_off()
        plt.tight_layout()
        out_paths = save_figure(fig, "fig_surrogate_optimization_process")
    else:
        candidate_scores = np.array([float(row["mean_patch_abs"]) for row in candidate_rows], dtype=float)
        candidate_scores.sort()
        ranks = np.arange(1, len(candidate_scores) + 1)
        top_k = len(confirmation_rows)

        ax0.plot(ranks, candidate_scores, color="0.30", linewidth=1.4)
        ax0.scatter(
            ranks[:top_k],
            candidate_scores[:top_k],
            color="#c44e52",
            s=20,
            zorder=3,
            label=f"Top {top_k}",
        )
        ax0.set_xlabel("Screened candidate rank")
        ax0.set_ylabel("Mean load-patch |$u_z$|")
        ax0.set_title("1. PINN Screening")
        ax0.legend(frameon=False, loc="upper left")

        top_ranks = np.array([int(row["rank"]) for row in confirmation_rows], dtype=int)
        pinn_obj = np.array([float(row["pinn_mean_patch_abs"]) for row in confirmation_rows], dtype=float)
        fem_obj = np.array([float(row["fem_mean_patch_abs"]) for row in confirmation_rows], dtype=float)
        x = np.arange(len(top_ranks))
        width = 0.36
        ax1.bar(x - width / 2, pinn_obj, width, color="#4c72b0", label="PINN")
        ax1.bar(x + width / 2, fem_obj, width, color="#55a868", label="FEM")
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"#{rank}" for rank in top_ranks])
        ax1.set_xlabel("Top-ranked candidate")
        ax1.set_ylabel("Mean load-patch |$u_z$|")
        ax1.set_title("2. Objective Confirmation")
        ax1.legend(frameon=False, loc="upper left")

        mae = np.array([float(row["top_uz_mae_pct"]) for row in confirmation_rows], dtype=float)
        bars = ax2.bar(x, mae, color="#8172b3")
        ax2.axhline(5.0, color="0.25", linestyle="--", linewidth=1.0, label="5%")
        for idx, bar in enumerate(bars):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.08,
                f"{mae[idx]:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"#{rank}" for rank in top_ranks])
        ax2.set_xlabel("Top-ranked candidate")
        ax2.set_ylabel("top $u_z$ MAE (%)")
        ax2.set_title("3. PINN vs FEM Error")
        ax2.legend(frameon=False, loc="upper left")

        plt.tight_layout()
        out_paths = save_figure(fig, "fig_surrogate_optimization_process")

    plt.close(fig)
    print("Wrote:")
    for path in out_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
