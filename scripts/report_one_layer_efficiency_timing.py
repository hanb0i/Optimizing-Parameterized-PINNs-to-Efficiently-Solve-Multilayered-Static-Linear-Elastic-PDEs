"""Generate one-layer timing and sparse-vs-fuller supervision accounting."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from one_layer_experiment_utils import (
    GRAPHS_DATA_DIR,
    ONE_LAYER_DIR,
    OneLayerCase,
    config,
    ensure_output_dirs,
    evaluate_case_grid,
    load_pinn,
    rows_to_csv,
    select_device,
    write_json,
)


TRAIN_TIME_RE = re.compile(r"SOAP Pretraining Complete\. Total Time:\s*(?P<sec>[0-9.]+)s")


def _training_time_from_artifacts(model_path: Path) -> dict:
    run_dir = model_path.parent
    timing_path = run_dir / "training_timing.json"
    if timing_path.exists():
        return json.loads(timing_path.read_text())
    loss_path = run_dir / "loss_history.npy"
    if loss_path.exists():
        return {"source": str(loss_path), "note": "Legacy checkpoint has loss history but no wall-clock timing JSON."}
    return {"note": "No one-time training timing artifact found for this checkpoint."}


def _supervision_accounting(ne_x: int, ne_y: int, ne_z: int) -> dict:
    data_e = list(getattr(config, "DATA_E_VALUES", [1.0, 5.0, 10.0]))
    data_t = list(getattr(config, "DATA_THICKNESS_VALUES", [0.05, 0.10, 0.15]))
    n_cases = len(data_e) * len(data_t)
    # one-layer/data.py calls fem_solver.solve_fem without overriding mesh resolution,
    # so the supervision generator uses fem_solver's default 30x30x10 mesh.
    supervision_ne_x, supervision_ne_y, supervision_ne_z = 30, 30, 10
    full_nodes_per_case = (supervision_ne_x + 1) * (supervision_ne_y + 1) * (supervision_ne_z + 1)
    sparse_points = int(getattr(config, "N_DATA_POINTS", 9000))
    return {
        "supervision_cases": n_cases,
        "mesh_nodes_per_case": full_nodes_per_case,
        "supervision_mesh": {"ne_x": supervision_ne_x, "ne_y": supervision_ne_y, "ne_z": supervision_ne_z},
        "sparse_supervision_points": sparse_points,
        "fuller_supervision_points_same_mesh": n_cases * full_nodes_per_case,
        "sparse_fraction_of_fuller_same_mesh": sparse_points / float(n_cases * full_nodes_per_case),
        "default_checkpoint_uses_fem_supervision": bool(getattr(config, "USE_SUPERVISION_DATA", False)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=str(ONE_LAYER_DIR / "pinn_model.pth"))
    parser.add_argument("--ne-x", type=int, default=16)
    parser.add_argument("--ne-y", type=int, default=16)
    parser.add_argument("--ne-z", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--out-csv", default=str(GRAPHS_DATA_DIR / "one_layer_efficiency_timing.csv"))
    parser.add_argument("--out-summary", default=str(GRAPHS_DATA_DIR / "one_layer_efficiency_timing_summary.json"))
    args = parser.parse_args()

    ensure_output_dirs()
    device = select_device()
    pinn, model_path = load_pinn(device, args.model_path)
    cases = [
        OneLayerCase("one_layer_mid", 5.0, 0.10),
        OneLayerCase("one_layer_random_like", 7.25, 0.073),
    ]

    print("Running one warm-up solve/evaluation outside the reported timing rows...")
    _ = evaluate_case_grid(pinn, device, cases[0], args.ne_x, args.ne_y, args.ne_z)

    rows = []
    for repeat in range(args.repeats):
        for case in cases:
            result = evaluate_case_grid(pinn, device, case, args.ne_x, args.ne_y, args.ne_z)
            rows.append(
                {
                    "repeat": str(repeat),
                    "case_id": case.case_id,
                    "n_eval_points": str(result["n_eval_points"]),
                    "fem_seconds": f"{result['fem_seconds']:.6f}",
                    "pinn_eval_seconds": f"{result['pinn_eval_seconds']:.6f}",
                    "speedup_fem_over_pinn_eval": f"{result['fem_seconds'] / max(result['pinn_eval_seconds'], 1e-12):.6f}",
                    "top_uz_mae_pct": f"{result['top_uz_mae_pct']:.6f}",
                    "volume_mae_pct": f"{result['volume_mae_pct']:.6f}",
                }
            )
            print(f"{case.case_id} repeat {repeat}: FEM={result['fem_seconds']:.3f}s PINN eval={result['pinn_eval_seconds']:.4f}s")

    rows_to_csv(
        Path(args.out_csv),
        ["repeat", "case_id", "n_eval_points", "fem_seconds", "pinn_eval_seconds", "speedup_fem_over_pinn_eval", "top_uz_mae_pct", "volume_mae_pct"],
        rows,
    )
    fem = np.array([float(r["fem_seconds"]) for r in rows], dtype=float)
    pinn_times = np.array([float(r["pinn_eval_seconds"]) for r in rows], dtype=float)
    training_cost = _training_time_from_artifacts(Path(model_path))
    train_seconds = float(training_cost.get("total_training_seconds", 0.0)) if isinstance(training_cost, dict) else 0.0
    summary = {
        "model_path": str(model_path),
        "mesh": {"ne_x": args.ne_x, "ne_y": args.ne_y, "ne_z": args.ne_z},
        "fem_seconds_mean": float(fem.mean()),
        "fem_seconds_worst": float(fem.max()),
        "fem_repeats_measured": int(len(fem)),
        "estimated_fem_seconds_for_1e6_configs": float(fem.mean() * 1_000_000.0),
        "pinn_eval_seconds_mean": float(pinn_times.mean()),
        "pinn_eval_seconds_worst": float(pinn_times.max()),
        "estimated_pinn_inference_seconds_for_1e6_configs": float(pinn_times.mean() * 1_000_000.0),
        "mean_per_case_speedup": float((fem / np.maximum(pinn_times, 1e-12)).mean()),
        "one_time_training_cost": training_cost,
        "estimated_pinn_total_seconds_for_1e6_configs_including_training_if_known": float(train_seconds + pinn_times.mean() * 1_000_000.0),
        "per_case_evaluation_cost_note": "PINN evaluation time excludes one-time training; FEM time is per solved case.",
        "sparse_vs_fuller_supervision": _supervision_accounting(args.ne_x, args.ne_y, args.ne_z),
    }
    write_json(Path(args.out_summary), summary)
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_summary}")


if __name__ == "__main__":
    main()
