"""Optimize three-layer designs with the PINN surrogate and confirm top designs with FEM."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from three_layer_experiment_utils import (
    GRAPHS_DATA_DIR,
    ThreeLayerCase,
    case_grid_top_surface_metrics,
    config,
    evaluate_case_grid,
    evaluate_case_top_surface,
    ensure_output_dirs,
    load_pinn,
    rows_to_csv,
    select_device,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION_PATH = REPO_ROOT / "graphs" / "data" / "three_layer_compliance_calibration.json"
BENCHMARK_NE_X = 16
BENCHMARK_NE_Y = 16
BENCHMARK_NE_Z = 8
OBJECTIVE_CHOICES = ("peak_downward_abs", "mean_patch_abs")


def _torch_ranges(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    lows = torch.tensor(
        [config.E_RANGE[0], config.E_RANGE[0], config.E_RANGE[0], config.T1_RANGE[0], config.T2_RANGE[0], config.T3_RANGE[0]],
        dtype=torch.float32,
        device=device,
    )
    highs = torch.tensor(
        [config.E_RANGE[1], config.E_RANGE[1], config.E_RANGE[1], config.T1_RANGE[1], config.T2_RANGE[1], config.T3_RANGE[1]],
        dtype=torch.float32,
        device=device,
    )
    return lows, highs


def _params_from_raw(raw: torch.Tensor, lows: torch.Tensor, highs: torch.Tensor) -> torch.Tensor:
    return lows + (highs - lows) * torch.sigmoid(raw)


def _differentiable_top_points(params: torch.Tensor, ne_x: int, ne_y: int, device: torch.device) -> torch.Tensor:
    x_nodes = torch.linspace(0.0, float(config.Lx), int(ne_x) + 1, device=device)
    y_nodes = torch.linspace(0.0, float(config.Ly), int(ne_y) + 1, device=device)
    xg, yg = torch.meshgrid(x_nodes, y_nodes, indexing="ij")
    e1, e2, e3, t1, t2, t3 = [params[i] for i in range(6)]
    thickness = t1 + t2 + t3
    n = xg.numel()
    return torch.stack(
        [
            xg.reshape(-1),
            yg.reshape(-1),
            thickness.expand(n),
            e1.expand(n),
            t1.expand(n),
            e2.expand(n),
            t2.expand(n),
            e3.expand(n),
            t3.expand(n),
            torch.full((n,), float(getattr(config, "RESTITUTION_REF", 0.5)), device=device),
            torch.full((n,), float(getattr(config, "FRICTION_REF", 0.3)), device=device),
            torch.full((n,), float(getattr(config, "IMPACT_VELOCITY_REF", 1.0)), device=device),
        ],
        dim=1,
    )


def _torch_u_from_v(v: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    e_scale = (pts[:, 3:4] + pts[:, 5:6] + pts[:, 7:8]) / 3.0
    t_scale = pts[:, 4:5] + pts[:, 6:7] + pts[:, 8:9]
    e_pow = float(getattr(config, "E_COMPLIANCE_POWER", 1.0))
    alpha = float(getattr(config, "THICKNESS_COMPLIANCE_ALPHA", 0.0))
    scale = float(getattr(config, "DISPLACEMENT_COMPLIANCE_SCALE", 1.0))
    h_ref = float(getattr(config, "H", 1.0))
    return scale * v / (e_scale ** e_pow) * (h_ref / torch.clamp(t_scale, min=1e-8)) ** alpha


def _differentiable_design_objective(pinn, params: torch.Tensor, ne_x: int, ne_y: int, device: torch.device) -> torch.Tensor:
    """Weighted objective for nontrivial layered design.

    The first term controls transmitted/average load-patch deflection. The
    second term penalizes material usage, preventing the optimum from being
    "make all layers as stiff and thick as possible." The third term softly
    discourages concentrating deformation in only one layer by penalizing large
    stiffness contrast.
    """
    pts = _differentiable_top_points(params, ne_x, ne_y, device)
    v = pinn(pts)
    u = _torch_u_from_v(v, pts)
    x = pts[:, 0]
    y = pts[:, 1]
    patch = (
        (x >= float(config.LOAD_PATCH_X[0]))
        & (x <= float(config.LOAD_PATCH_X[1]))
        & (y >= float(config.LOAD_PATCH_Y[0]))
        & (y <= float(config.LOAD_PATCH_Y[1]))
    )
    uz_patch = u[patch, 2] if torch.any(patch) else u[:, 2]
    mean_patch_abs = torch.mean(torch.abs(uz_patch))
    e = params[:3]
    t = params[3:]
    material_cost = torch.sum(e * t) / (3.0 * float(config.E_RANGE[1]) * float(config.T3_RANGE[1]))
    contrast_penalty = torch.var(torch.log(torch.clamp(e, min=1e-8)))
    constraint_penalty = torch.relu(torch.min(e) - 4.999) ** 2
    return mean_patch_abs + 0.03 * material_cost + 0.005 * contrast_penalty + 10.0 * constraint_penalty


def _gradient_optimize_designs(pinn, device: torch.device, n_starts: int, steps: int, lr: float, ne_x: int, ne_y: int, seed: int) -> list[dict]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    lows, highs = _torch_ranges(device)
    results = []
    for start_idx in range(n_starts):
        initial = torch.tensor(rng.uniform(lows.cpu().numpy(), highs.cpu().numpy()), dtype=torch.float32, device=device)
        raw = torch.logit(torch.clamp((initial - lows) / (highs - lows), 1e-4, 1.0 - 1e-4)).detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([raw], lr=lr)
        best = None
        for step in range(steps):
            opt.zero_grad()
            params = _params_from_raw(raw, lows, highs)
            objective = _differentiable_design_objective(pinn, params, ne_x, ne_y, device)
            objective.backward()
            opt.step()
            with torch.no_grad():
                params_now = _params_from_raw(raw, lows, highs).detach().cpu().numpy()
                objective_now = float(objective.detach().cpu())
                feasible = bool(np.min(params_now[:3]) < 5.0)
                if best is None or objective_now < best["objective"]:
                    best = {"objective": objective_now, "params": params_now.copy(), "step": step, "feasible": feasible}
        assert best is not None
        e1, e2, e3, t1, t2, t3 = [float(v) for v in best["params"]]
        case = ThreeLayerCase(
            f"gradient_start_{start_idx:03d}",
            e1,
            e2,
            e3,
            t1,
            t2,
            t3,
        )
        eval_result = evaluate_case_top_surface(pinn, device, case, ne_x, ne_y)
        eval_result["optimizer_objective"] = best["objective"]
        eval_result["optimizer_step"] = best["step"]
        eval_result["constraint_min_E_lt_5"] = best["feasible"]
        results.append(eval_result)
    return results


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
        "optimizer_objective": f"{float(result.get('optimizer_objective', np.nan)):.10g}",
        "optimizer_step": str(result.get("optimizer_step", "")),
        "constraint_min_E_lt_5": str(result.get("constraint_min_E_lt_5", "")),
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
    parser.add_argument("--n-candidates", type=int, default=24, help="Number of gradient-descent starts")
    parser.add_argument("--optimization-steps", type=int, default=250)
    parser.add_argument("--optimization-lr", type=float, default=0.05)
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

    print(f"Running {args.n_candidates} multi-start gradient optimizations with the PINN surrogate...")
    candidate_results = _gradient_optimize_designs(
        pinn,
        device,
        n_starts=args.n_candidates,
        steps=args.optimization_steps,
        lr=args.optimization_lr,
        ne_x=args.surrogate_ne_x,
        ne_y=args.surrogate_ne_y,
        seed=args.seed,
    )
    for idx, result in enumerate(candidate_results, start=1):
        print(
            f"  start {idx:3d}: objective={result['optimizer_objective']:.6g}, "
            f"{args.objective}={_objective_value(result, args.objective):.6g}, "
            f"min(E)={min(result['case'].e):.3g}"
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
            "optimizer_objective",
            "optimizer_step",
            "constraint_min_E_lt_5",
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
            "optimizer_objective",
            "optimizer_step",
            "constraint_min_E_lt_5",
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
        "optimization_protocol": {
            "method": "multi_start_adam_gradient_descent_on_pinn_surrogate",
            "n_starts": int(args.n_candidates),
            "steps_per_start": int(args.optimization_steps),
            "learning_rate": float(args.optimization_lr),
            "constraint": "at least one layer Young's modulus must be less than 5",
        },
        "objective": {
            "name": "weighted_mean_patch_deflection_material_cost_and_contrast",
            "description": (
                "Minimize mean absolute load-patch deflection plus small material-cost and stiffness-contrast penalties. "
                "This is more meaningful than minimizing max displacement alone because it penalizes transmitted response "
                "over the loaded area while discouraging the trivial all-stiff/all-thick solution."
            ),
            "reported_surrogate_metric": args.objective,
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
