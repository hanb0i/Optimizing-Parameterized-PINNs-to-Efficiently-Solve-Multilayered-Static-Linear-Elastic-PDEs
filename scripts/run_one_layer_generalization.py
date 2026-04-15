"""Evaluate one-layer pure-physics PINN generalization on random interior cases."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from one_layer_experiment_utils import (
    GRAPHS_DATA_DIR,
    ensure_output_dirs,
    evaluate_case_grid,
    load_pinn,
    random_interior_cases,
    rows_to_csv,
    select_device,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--n-cases", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260415)
    parser.add_argument("--ne-x", type=int, default=8)
    parser.add_argument("--ne-y", type=int, default=8)
    parser.add_argument("--ne-z", type=int, default=4)
    parser.add_argument("--out-csv", default=str(GRAPHS_DATA_DIR / "one_layer_random_generalization.csv"))
    parser.add_argument("--out-summary", default=str(GRAPHS_DATA_DIR / "one_layer_random_generalization_summary.json"))
    args = parser.parse_args()

    ensure_output_dirs()
    device = select_device()
    pinn, model_path = load_pinn(device, args.model_path)

    rows = []
    for case in random_interior_cases(args.n_cases, args.seed):
        result = evaluate_case_grid(pinn, device, case, args.ne_x, args.ne_y, args.ne_z)
        rows.append(
            {
                "case_id": case.case_id,
                "E": f"{case.E:.8g}",
                "thickness": f"{case.thickness:.8g}",
                "top_uz_mae_pct": f"{result['top_uz_mae_pct']:.6f}",
                "top_uz_max_pct": f"{result['top_uz_max_pct']:.6f}",
                "volume_mae_pct": f"{result['volume_mae_pct']:.6f}",
                "volume_max_pct": f"{result['volume_max_pct']:.6f}",
                "peak_fem_uz": f"{result['peak_fem_uz']:.10g}",
                "peak_pinn_uz": f"{result['peak_pinn_uz']:.10g}",
                "fem_seconds": f"{result['fem_seconds']:.6f}",
                "pinn_eval_seconds": f"{result['pinn_eval_seconds']:.6f}",
                "n_eval_points": str(result["n_eval_points"]),
            }
        )
        print(f"{case.case_id}: top MAE={result['top_uz_mae_pct']:.2f}% volume MAE={result['volume_mae_pct']:.2f}%")

    fieldnames = [
        "case_id",
        "E",
        "thickness",
        "top_uz_mae_pct",
        "top_uz_max_pct",
        "volume_mae_pct",
        "volume_max_pct",
        "peak_fem_uz",
        "peak_pinn_uz",
        "fem_seconds",
        "pinn_eval_seconds",
        "n_eval_points",
    ]
    rows_to_csv(Path(args.out_csv), fieldnames, rows)

    top = np.array([float(r["top_uz_mae_pct"]) for r in rows], dtype=float)
    vol = np.array([float(r["volume_mae_pct"]) for r in rows], dtype=float)
    summary = {
        "model_path": str(model_path),
        "seed": int(args.seed),
        "n_cases": int(args.n_cases),
        "mesh": {"ne_x": args.ne_x, "ne_y": args.ne_y, "ne_z": args.ne_z},
        "top_uz_mae_pct_mean": float(top.mean()) if len(top) else None,
        "top_uz_mae_pct_worst": float(top.max()) if len(top) else None,
        "volume_mae_pct_mean": float(vol.mean()) if len(vol) else None,
        "volume_mae_pct_worst": float(vol.max()) if len(vol) else None,
        "all_top_cases_below_5_pct": bool(np.all(top < 5.0)) if len(top) else False,
        "all_volume_cases_below_5_pct": bool(np.all(vol < 5.0)) if len(vol) else False,
    }
    write_json(Path(args.out_summary), summary)
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_summary}")


if __name__ == "__main__":
    main()
