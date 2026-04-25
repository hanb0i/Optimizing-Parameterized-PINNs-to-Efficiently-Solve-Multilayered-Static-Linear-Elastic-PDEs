"""Optimize three-layer designs with the PINN surrogate and confirm top designs with FEM."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from three_layer_experiment_utils import (
    GRAPHS_DATA_DIR,
    case_grid_top_surface_metrics,
    config,
    evaluate_case_grid,
    evaluate_case_top_surface,
    ensure_output_dirs,
    load_pinn,
    random_interior_cases,
    rows_to_csv,
    select_device,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION_PATH = REPO_ROOT / "graphs" / "data" / "three_layer_compliance_calibration.json"
BENCHMARK_NE_X = 8
BENCHMARK_NE_Y = 8
BENCHMARK_NE_Z = 4
OBJECTIVE_CHOICES = ("peak_downward_abs", "mean_patch_abs")


def _objective_value(result: dict, objective: str) -> float:
    return float(result[objective])


def _candidate_row(result: dict, rank: int | None = None) -> dict:
    case = result["case"]
    row = {
        "case_id": case.case_id,
        "e1": f"{case.e1:.8g}",
        "e2": f"{case.e2:.8g}",
        "e3": f"{case.e3:.8g}",
        "t1": f"{case.t1:.8g}",
        "t2": f"{case.t2:.8g}",
        "t3": f"{case.t3:.8g}",
        "total_thickness": f"{case.thickness:.8g}",
        "peak_downward_uz": f"{result['peak_downward_uz']:.10g}",
        "peak_downward_abs": f"{result['peak_downward_abs']:.10g}",
        "mean_patch_uz": f"{result['mean_patch_uz']:.10g}",
        "mean_patch_abs": f"{result['mean_patch_abs']:.10g}",
        "pinn_eval_seconds": f"{result['pinn_eval_seconds']:.6f}",
        "n_eval_points": str(result["n_eval_points"]),
    }
    if rank is not None:
        row["rank"] = str(rank)
    return row


def _confirmation_row(rank: int, pinn_result: dict, fem_result: dict) -> dict:
    case = pinn_result["case"]
    return {
        "rank": str(rank),
        "case_id": case.case_id,
        "e1": f"{case.e1:.8g}",
        "e2": f"{case.e2:.8g}",
        "e3": f"{case.e3:.8g}",
        "t1": f"{case.t1:.8g}",
        "t2": f"{case.t2:.8g}",
        "t3": f"{case.t3:.8g}",
        "total_thickness": f"{case.thickness:.8g}",
        "pinn_peak_downward_uz": f"{pinn_result['peak_downward_uz']:.10g}",
        "pinn_peak_downward_abs": f"{pinn_result['peak_downward_abs']:.10g}",
        "fem_peak_downward_uz": f"{fem_result['peak_downward_uz']:.10g}",
        "fem_peak_downward_abs": f"{fem_result['peak_downward_abs']:.10g}",
        "abs_gap_peak_downward_abs": f"{abs(pinn_result['peak_downward_abs'] - fem_result['peak_downward_abs']):.10g}",
        "rel_gap_peak_downward_pct": (
            f"{100.0 * abs(pinn_result['peak_downward_abs'] - fem_result['peak_downward_abs']) / max(fem_result['peak_downward_abs'], 1e-12):.6f}"
        ),
        "pinn_mean_patch_uz": f"{pinn_result['mean_patch_uz']:.10g}",
        "pinn_mean_patch_abs": f"{pinn_result['mean_patch_abs']:.10g}",
        "fem_mean_patch_uz": f"{fem_result['mean_patch_uz']:.10g}",
        "fem_mean_patch_abs": f"{fem_result['mean_patch_abs']:.10g}",
        "top_uz_mae_pct": f"{fem_result['top_uz_mae_pct']:.6f}",
        "top_uz_max_pct": f"{fem_result['top_uz_max_pct']:.6f}",
        "volume_mae_pct": f"{fem_result['volume_mae_pct']:.6f}",
        "volume_max_pct": f"{fem_result['volume_max_pct']:.6f}",
        "pinn_eval_seconds": f"{pinn_result['pinn_eval_seconds']:.6f}",
        "fem_seconds": f"{fem_result['fem_seconds']:.6f}",
        "n_eval_points": str(fem_result["n_eval_points"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--calibration-json", default=None)
    parser.add_argument("--objective", choices=OBJECTIVE_CHOICES, default="mean_patch_abs")
    parser.add_argument("--n-candidates", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260423)
    parser.add_argument("--surrogate-ne-x", type=int, default=20)
    parser.add_argument("--surrogate-ne-y", type=int, default=20)
    parser.add_argument("--fem-ne-x", type=int, default=BENCHMARK_NE_X)
    parser.add_argument("--fem-ne-y", type=int, default=BENCHMARK_NE_Y)
    parser.add_argument("--fem-ne-z", type=int, default=BENCHMARK_NE_Z)
    parser.add_argument("--out-candidates-csv", default=str(GRAPHS_DATA_DIR / "surrogate_optimization_mean_patch_candidates.csv"))
    parser.add_argument("--out-topk-csv", default=str(GRAPHS_DATA_DIR / "surrogate_optimization_mean_patch_topk.csv"))
    parser.add_argument("--out-confirmation-csv", default=str(GRAPHS_DATA_DIR / "surrogate_optimization_mean_patch_confirmation.csv"))
    parser.add_argument("--out-summary", default=str(GRAPHS_DATA_DIR / "surrogate_optimization_mean_patch_summary.json"))
    args = parser.parse_args()

    ensure_output_dirs()
    calibration_path = args.calibration_json or os.getenv("PINN_CALIBRATION_JSON")
    if calibration_path:
        os.environ["PINN_CALIBRATION_JSON"] = calibration_path
    elif DEFAULT_CALIBRATION_PATH.exists():
        os.environ["PINN_CALIBRATION_JSON"] = str(DEFAULT_CALIBRATION_PATH)
        calibration_path = str(DEFAULT_CALIBRATION_PATH)

    device = select_device()
    pinn, model_path = load_pinn(device, args.model_path)

    candidate_results = []
    print(f"Screening {args.n_candidates} random interior designs with the PINN surrogate...")
    for idx, case in enumerate(random_interior_cases(args.n_candidates, args.seed), start=1):
        result = evaluate_case_top_surface(pinn, device, case, args.surrogate_ne_x, args.surrogate_ne_y)
        candidate_results.append(result)
        if idx == 1 or idx % 50 == 0 or idx == args.n_candidates:
            print(
                f"  {idx:4d}/{args.n_candidates}: best {args.objective} so far = "
                f"{min(_objective_value(r, args.objective) for r in candidate_results):.6g}"
            )

    candidate_results.sort(key=lambda r: (_objective_value(r, args.objective), r["peak_downward_abs"], r["mean_patch_abs"]))
    top_k = candidate_results[: max(1, min(args.top_k, len(candidate_results)))]

    candidate_rows = [_candidate_row(result) for result in candidate_results]
    rows_to_csv(
        Path(args.out_candidates_csv),
        [
            "case_id",
            "e1",
            "e2",
            "e3",
            "t1",
            "t2",
            "t3",
            "total_thickness",
            "peak_downward_uz",
            "peak_downward_abs",
            "mean_patch_uz",
            "mean_patch_abs",
            "pinn_eval_seconds",
            "n_eval_points",
        ],
        candidate_rows,
    )

    topk_rows = [_candidate_row(result, rank=rank) for rank, result in enumerate(top_k, start=1)]
    rows_to_csv(
        Path(args.out_topk_csv),
        [
            "rank",
            "case_id",
            "e1",
            "e2",
            "e3",
            "t1",
            "t2",
            "t3",
            "total_thickness",
            "peak_downward_uz",
            "peak_downward_abs",
            "mean_patch_uz",
            "mean_patch_abs",
            "pinn_eval_seconds",
            "n_eval_points",
        ],
        topk_rows,
    )

    print(f"Confirming the top {len(top_k)} surrogate-ranked designs with FEM...")
    confirmation_rows = []
    fem_confirmations = []
    for rank, pinn_result in enumerate(top_k, start=1):
        grid_result = evaluate_case_grid(
            pinn,
            device,
            pinn_result["case"],
            args.fem_ne_x,
            args.fem_ne_y,
            args.fem_ne_z,
        )
        top_metrics = case_grid_top_surface_metrics(grid_result)
        fem_result = {
            "peak_downward_uz": top_metrics["fem_top_metrics"]["peak_downward_uz"],
            "peak_downward_abs": top_metrics["fem_top_metrics"]["peak_downward_abs"],
            "mean_patch_uz": top_metrics["fem_top_metrics"]["mean_patch_uz"],
            "mean_patch_abs": top_metrics["fem_top_metrics"]["mean_patch_abs"],
            "top_uz_mae_pct": grid_result["top_uz_mae_pct"],
            "top_uz_max_pct": grid_result["top_uz_max_pct"],
            "volume_mae_pct": grid_result["volume_mae_pct"],
            "volume_max_pct": grid_result["volume_max_pct"],
            "fem_seconds": grid_result["fem_seconds"],
            "n_eval_points": int(top_metrics["x_grid"].size),
        }
        pinn_confirm_result = {
            "case": pinn_result["case"],
            "peak_downward_uz": top_metrics["pinn_top_metrics"]["peak_downward_uz"],
            "peak_downward_abs": top_metrics["pinn_top_metrics"]["peak_downward_abs"],
            "mean_patch_uz": top_metrics["pinn_top_metrics"]["mean_patch_uz"],
            "mean_patch_abs": top_metrics["pinn_top_metrics"]["mean_patch_abs"],
            "pinn_eval_seconds": grid_result["pinn_eval_seconds"],
        }
        fem_confirmations.append(fem_result)
        confirmation_rows.append(_confirmation_row(rank, pinn_confirm_result, fem_result))
        print(
            f"  rank {rank}: PINN {args.objective}={_objective_value(pinn_confirm_result, args.objective):.6g}, "
            f"FEM {args.objective}={_objective_value(fem_result, args.objective):.6g}"
        )

    rows_to_csv(
        Path(args.out_confirmation_csv),
        [
            "rank",
            "case_id",
            "e1",
            "e2",
            "e3",
            "t1",
            "t2",
            "t3",
            "total_thickness",
            "pinn_peak_downward_uz",
            "pinn_peak_downward_abs",
            "fem_peak_downward_uz",
            "fem_peak_downward_abs",
            "abs_gap_peak_downward_abs",
            "rel_gap_peak_downward_pct",
            "pinn_mean_patch_uz",
            "pinn_mean_patch_abs",
            "fem_mean_patch_uz",
            "fem_mean_patch_abs",
            "top_uz_mae_pct",
            "top_uz_max_pct",
            "volume_mae_pct",
            "volume_max_pct",
            "pinn_eval_seconds",
            "fem_seconds",
            "n_eval_points",
        ],
        confirmation_rows,
    )

    surrogate_objectives = np.array([_objective_value(r, args.objective) for r in candidate_results], dtype=float)
    pinn_times = np.array([r["pinn_eval_seconds"] for r in candidate_results], dtype=float)
    fem_objectives = np.array([_objective_value(r, args.objective) for r in fem_confirmations], dtype=float)
    top_mae = np.array([r["top_uz_mae_pct"] for r in fem_confirmations], dtype=float)
    vol_mae = np.array([r["volume_mae_pct"] for r in fem_confirmations], dtype=float)
    fem_gaps = np.array(
        [
            abs(_objective_value({"peak_downward_abs": float(pinn_result["pinn_peak_downward_abs"]), "mean_patch_abs": float(pinn_result["pinn_mean_patch_abs"])}, args.objective) - _objective_value(fem_result, args.objective))
            for pinn_result, fem_result in zip(confirmation_rows, fem_confirmations)
        ],
        dtype=float,
    )
    fem_best_idx = int(np.argmin(fem_objectives))
    summary = {
        "model_path": str(model_path),
        "seed": int(args.seed),
        "n_candidates": int(args.n_candidates),
        "top_k": int(len(top_k)),
        "calibration_json": calibration_path,
        "benchmark_protocol": "random_interior_generalization_style_confirmation",
        "objective": {
            "name": args.objective,
            "description": (
                "Minimize the absolute value of the most negative top-surface u_z predicted by the surrogate."
                if args.objective == "peak_downward_abs"
                else "Minimize the mean absolute top-surface deflection over the load patch predicted by the surrogate."
            ),
        },
        "surrogate_grid": {"ne_x": int(args.surrogate_ne_x), "ne_y": int(args.surrogate_ne_y)},
        "fem_confirmation_mesh": {"ne_x": int(args.fem_ne_x), "ne_y": int(args.fem_ne_y), "ne_z": int(args.fem_ne_z)},
        "surrogate_screening": {
            "best_objective_value": float(surrogate_objectives.min()),
            "median_objective_value": float(np.median(surrogate_objectives)),
            "worst_objective_value": float(surrogate_objectives.max()),
            "mean_pinn_eval_seconds": float(pinn_times.mean()),
            "total_pinn_eval_seconds": float(pinn_times.sum()),
        },
        "best_surrogate_design": {
            "rank": 1,
            "case_id": top_k[0]["case"].case_id,
            "e": [float(top_k[0]["case"].e1), float(top_k[0]["case"].e2), float(top_k[0]["case"].e3)],
            "t": [float(top_k[0]["case"].t1), float(top_k[0]["case"].t2), float(top_k[0]["case"].t3)],
            "peak_downward_uz": float(top_k[0]["peak_downward_uz"]),
            "peak_downward_abs": float(top_k[0]["peak_downward_abs"]),
            "mean_patch_uz": float(top_k[0]["mean_patch_uz"]),
            "mean_patch_abs": float(top_k[0]["mean_patch_abs"]),
        },
        "best_fem_confirmed_design_among_top_k": {
            "rank_within_top_k": fem_best_idx + 1,
            "case_id": top_k[fem_best_idx]["case"].case_id,
            "e": [
                float(top_k[fem_best_idx]["case"].e1),
                float(top_k[fem_best_idx]["case"].e2),
                float(top_k[fem_best_idx]["case"].e3),
            ],
            "t": [
                float(top_k[fem_best_idx]["case"].t1),
                float(top_k[fem_best_idx]["case"].t2),
                float(top_k[fem_best_idx]["case"].t3),
            ],
            "fem_peak_downward_uz": float(fem_confirmations[fem_best_idx]["peak_downward_uz"]),
            "fem_peak_downward_abs": float(fem_confirmations[fem_best_idx]["peak_downward_abs"]),
        },
        "fem_confirmation": {
            "mean_abs_gap_objective": float(fem_gaps.mean()),
            "worst_abs_gap_objective": float(fem_gaps.max()),
            "mean_rel_gap_objective_pct": float(
                np.mean(
                    [
                        100.0
                        * abs(
                            _objective_value(
                                {
                                    "peak_downward_abs": float(pinn_result["pinn_peak_downward_abs"]),
                                    "mean_patch_abs": float(pinn_result["pinn_mean_patch_abs"]),
                                },
                                args.objective,
                            )
                            - _objective_value(fem_result, args.objective)
                        )
                        / max(_objective_value(fem_result, args.objective), 1e-12)
                        for pinn_result, fem_result in zip(confirmation_rows, fem_confirmations)
                    ]
                )
            ),
            "top_uz_mae_pct_mean": float(top_mae.mean()),
            "top_uz_mae_pct_worst": float(top_mae.max()),
            "volume_mae_pct_mean": float(vol_mae.mean()),
            "volume_mae_pct_worst": float(vol_mae.max()),
        },
    }
    write_json(Path(args.out_summary), summary)

    print(f"Wrote {args.out_candidates_csv}")
    print(f"Wrote {args.out_topk_csv}")
    print(f"Wrote {args.out_confirmation_csv}")
    print(f"Wrote {args.out_summary}")


if __name__ == "__main__":
    main()
