from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-csv", default=str(DATA_DIR / "surrogate_optimization_mean_patch_candidates.csv"))
    parser.add_argument("--confirmation-csv", default=str(DATA_DIR / "surrogate_optimization_mean_patch_confirmation.csv"))
    parser.add_argument("--objective-column", default="mean_patch_abs")
    parser.add_argument("--pinn-objective-column", default="pinn_mean_patch_abs")
    parser.add_argument("--fem-objective-column", default="fem_mean_patch_abs")
    parser.add_argument("--out-stem", default="fig_surrogate_optimization_mean_patch")
    parser.add_argument("--screening-title", default="PINN Screening Over 500 Designs (Mean Patch)")
    parser.add_argument("--confirmation-title", default="FEM Confirmation of Top Designs (Mean Patch)")
    parser.add_argument("--y-label", default="Mean load-patch |$u_z$|")
    args = parser.parse_args()

    apply_ieee_style()
    candidates_path = Path(args.candidates_csv)
    confirmation_path = Path(args.confirmation_csv)
    candidate_rows = _load_rows(candidates_path)
    confirmation_rows = _load_rows(confirmation_path)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax0, ax1 = axes

    if not candidate_rows or not confirmation_rows:
        for ax in axes:
            watermark_placeholder(ax, "MISSING\nsurrogate optimization")
            ax.set_axis_off()
        plt.tight_layout()
        out_paths = save_figure(fig, "fig_surrogate_optimization_placeholder")
    else:
        candidate_scores = np.array([float(row[args.objective_column]) for row in candidate_rows], dtype=float)
        candidate_scores.sort()
        ranks = np.arange(1, len(candidate_scores) + 1)
        ax0.plot(ranks, candidate_scores, color="0.25", linewidth=1.4)
        top_k = len(confirmation_rows)
        ax0.scatter(
            ranks[:top_k],
            candidate_scores[:top_k],
            color="#c44e52",
            s=18,
            zorder=3,
            label=f"Top {top_k} screened",
        )
        ax0.set_xlabel("Candidate rank after PINN screening")
        ax0.set_ylabel(args.y_label)
        ax0.set_title(args.screening_title)
        ax0.legend(frameon=False, loc="upper left")

        top_ranks = np.array([int(row["rank"]) for row in confirmation_rows], dtype=int)
        pinn = np.array([float(row[args.pinn_objective_column]) for row in confirmation_rows], dtype=float)
        fem = np.array([float(row[args.fem_objective_column]) for row in confirmation_rows], dtype=float)
        rel_gap = np.array([float(row["rel_gap_peak_downward_pct"]) for row in confirmation_rows], dtype=float)

        width = 0.36
        x = np.arange(len(top_ranks))
        ax1.bar(x - width / 2, pinn, width, label="PINN", color="#4c72b0")
        ax1.bar(x + width / 2, fem, width, label="FEM", color="#55a868")
        for idx, gap in enumerate(rel_gap):
            ax1.text(x[idx], max(pinn[idx], fem[idx]) * 1.02, f"{gap:.1f}%", ha="center", va="bottom", fontsize=7)
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"#{rank}" for rank in top_ranks])
        ax1.set_xlabel("Top-ranked candidate")
        ax1.set_ylabel(args.y_label)
        ax1.set_title(args.confirmation_title)
        ax1.legend(frameon=False, loc="upper left")

        plt.tight_layout()
        out_paths = save_figure(fig, args.out_stem)

    plt.close(fig)
    print("Wrote:")
    for path in out_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
