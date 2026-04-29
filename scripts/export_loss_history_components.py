"""Export existing loss_history.npy artifacts to component-wise CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


LOSS_FIELDS = [
    "total",
    "pde",
    "bc_sides",
    "free_top",
    "free_bot",
    "load",
    "energy",
    "impact_invariance",
    "impact_contact",
    "friction_coulomb",
    "friction_stick",
    "interface_u",
    "data",
    "fem_mae",
    "fem_max_err",
]


def export_loss_history(input_path: Path, output_path: Path) -> None:
    history = np.load(input_path, allow_pickle=True).item()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "step", *LOSS_FIELDS])
        writer.writeheader()
        for stage in ("adam", "lbfgs"):
            stage_history = history.get(stage, {})
            n_steps = len(stage_history.get("total", []))
            for idx in range(n_steps):
                row = {"stage": stage, "step": idx}
                for field in LOSS_FIELDS:
                    values = stage_history.get(field, [])
                    row[field] = values[idx] if idx < len(values) else 0.0
                writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="loss_history.npy paths")
    args = parser.parse_args()

    for input_arg in args.inputs:
        input_path = Path(input_arg)
        output_path = input_path.with_name("loss_history_components.csv")
        export_loss_history(input_path, output_path)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
