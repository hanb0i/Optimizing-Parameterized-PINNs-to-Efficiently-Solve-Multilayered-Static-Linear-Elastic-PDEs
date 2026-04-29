"""Generate timing and efficiency reports for FEM and PINN workflows."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from three_layer_experiment_utils import (
    GRAPHS_DATA_DIR,
    ThreeLayerCase,
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

    out = {}
    logs_dir = GRAPHS_DATA_DIR / "ablation_logs"
    if logs_dir.exists():
        for log_path in logs_dir.glob("*_train.log"):
            text = log_path.read_text(errors="ignore")
            match = TRAIN_TIME_RE.search(text)
            if match and (run_dir.name in log_path.stem or model_path.name == "pinn_model.pth"):
                out["total_training_seconds"] = float(match.group("sec"))
                out["source_log"] = str(log_path)
                break
    return out


def _supervision_accounting(ne_x: int, ne_y: int, ne_z: int) -> dict:
    # Defaults mirror pinn_config.py after env parsing. Keep this script as an accounting report,
    # not a hidden data generator.
    data_e = list(getattr(config, "DATA_E_VALUES", [1.0, 10.0]))
    data_t1 = list(getattr(config, "DATA_T1_VALUES", [0.02, 0.10]))
    data_t2 = list(getattr(config, "DATA_T2_VALUES", [0.02, 0.10]))
    data_t3 = list(getattr(config, "DATA_T3_VALUES", [0.02, 0.10]))
    n_cases = len(data_e) ** 3 * len(data_t1) * len(data_t2) * len(data_t3)
    full_nodes_per_case = (ne_x + 1) * (ne_y + 1) * (ne_z + 1)
    sparse_points = int(getattr(config, "N_DATA_POINTS", 36000))
    return {
        "supervision_cases": n_cases,
        "mesh_nodes_per_case": full_nodes_per_case,
        "sparse_supervision_points": sparse_points,
        "fuller_supervision_points_same_mesh": n_cases * full_nodes_per_case,
        "sparse_fraction_of_fuller_same_mesh": sparse_points / float(n_cases * full_nodes_per_case),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--ne-x", type=int, default=16)
    parser.add_argument("--ne-y", type=int, default=16)
    parser.add_argument("--ne-z", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--out-csv", default=str(GRAPHS_DATA_DIR / "efficiency_timing.csv"))
    parser.add_argument("--out-summary", default=str(GRAPHS_DATA_DIR / "efficiency_timing_summary.json"))
    args = parser.parse_args()

    ensure_output_dirs()
    device = select_device()
    pinn, model_path = load_pinn(device, args.model_path)

    cases = [
        ThreeLayerCase("supervised_extreme", 1.0, 10.0, 10.0, 0.10, 0.02, 0.02),
        ThreeLayerCase("mixed_midrange", 4.0, 6.5, 8.0, 0.045, 0.065, 0.085),
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
            print(
                f"{case.case_id} repeat {repeat}: FEM={result['fem_seconds']:.3f}s "
                f"PINN eval={result['pinn_eval_seconds']:.4f}s"
            )

    rows_to_csv(
        Path(args.out_csv),
        [
            "repeat",
            "case_id",
            "n_eval_points",
            "fem_seconds",
            "pinn_eval_seconds",
            "speedup_fem_over_pinn_eval",
            "top_uz_mae_pct",
            "volume_mae_pct",
        ],
        rows,
    )

    fem_times = np.array([float(r["fem_seconds"]) for r in rows], dtype=float)
    pinn_times = np.array([float(r["pinn_eval_seconds"]) for r in rows], dtype=float)
    train_time = _training_time_from_artifacts(Path(model_path))
    train_seconds = float(train_time.get("total_training_seconds", 0.0)) if isinstance(train_time, dict) else 0.0
    accounting = _supervision_accounting(args.ne_x, args.ne_y, args.ne_z)
    summary = {
        "model_path": str(model_path),
        "mesh": {"ne_x": args.ne_x, "ne_y": args.ne_y, "ne_z": args.ne_z},
        "fem_seconds_mean": float(fem_times.mean()),
        "fem_seconds_worst": float(fem_times.max()),
        "fem_repeats_measured": int(len(fem_times)),
        "estimated_fem_seconds_for_1e6_configs": float(fem_times.mean() * 1_000_000.0),
        "pinn_eval_seconds_mean": float(pinn_times.mean()),
        "pinn_eval_seconds_worst": float(pinn_times.max()),
        "estimated_pinn_inference_seconds_for_1e6_configs": float(pinn_times.mean() * 1_000_000.0),
        "mean_per_case_speedup": float((fem_times / np.maximum(pinn_times, 1e-12)).mean()),
        "one_time_training_cost": train_time,
        "estimated_pinn_total_seconds_for_1e6_configs_including_training_if_known": float(train_seconds + pinn_times.mean() * 1_000_000.0),
        "per_case_evaluation_cost_note": "PINN evaluation time excludes one-time training; FEM time is per solved case.",
        "sparse_vs_fuller_supervision": accounting,
    }
    write_json(Path(args.out_summary), summary)
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_summary}")


if __name__ == "__main__":
    main()
