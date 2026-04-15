"""Tune existing compliance-scaling parameters against FEM references.

This does not train a surrogate and does not replace the PINN. It optimizes the
global post-network compliance map already used by the PINN workflow:

    u = scale * v / E_mean**p * (H / t_total)**alpha
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.optimize as opt
import torch

from three_layer_experiment_utils import (
    GRAPHS_DATA_DIR,
    ThreeLayerCase,
    config,
    ensure_output_dirs,
    load_pinn,
    make_points,
    max_pct,
    random_interior_cases,
    select_device,
    solve_fem_case,
    write_json,
)


def _u_from_params(v: np.ndarray, pts: np.ndarray, scale: float, e_pow: float, alpha: float) -> np.ndarray:
    e_scale = (pts[:, 3:4] + pts[:, 5:6] + pts[:, 7:8]) / 3.0
    t_scale = pts[:, 4:5] + pts[:, 6:7] + pts[:, 8:9]
    h_ref = float(getattr(config, "H", 1.0))
    return scale * v / (e_scale**e_pow) * (h_ref / np.clip(t_scale, 1e-8, None)) ** alpha


def _prepare_case(pinn, device, case: ThreeLayerCase, ne_x: int, ne_y: int, ne_z: int) -> dict:
    x_nodes, y_nodes, z_nodes, u_fem, _ = solve_fem_case(case, ne_x, ne_y, ne_z)
    xg, yg, zg = np.meshgrid(x_nodes, y_nodes, z_nodes, indexing="ij")
    pts = make_points(xg.ravel(), yg.ravel(), zg.ravel(), case)
    with torch.no_grad():
        v = pinn(torch.tensor(pts, dtype=torch.float32, device=device)).detach().cpu().numpy()
    return {
        "case": case,
        "pts": pts,
        "v": v,
        "u_fem": u_fem.reshape(-1, 3),
        "top_mask": np.isclose(pts[:, 2], case.thickness),
    }


def _mae_pct(pred: np.ndarray, ref: np.ndarray) -> float:
    denom = float(np.max(np.abs(ref)))
    return 100.0 * float(np.mean(np.abs(pred - ref))) / denom if denom > 0 else 0.0


def _metrics(prepared: list[dict], params: np.ndarray) -> dict:
    scale, e_pow, alpha = [float(v) for v in params]
    top_maes = []
    volume_maes = []
    top_maxes = []
    for blob in prepared:
        u_pred = _u_from_params(blob["v"], blob["pts"], scale, e_pow, alpha)
        u_ref = blob["u_fem"]
        top = blob["top_mask"]
        top_maes.append(_mae_pct(u_pred[top, 2], u_ref[top, 2]))
        volume_maes.append(_mae_pct(u_pred, u_ref))
        top_maxes.append(max_pct(u_pred[top, 2], u_ref[top, 2]))
    return {
        "top_mae_pct_mean": float(np.mean(top_maes)),
        "top_mae_pct_worst": float(np.max(top_maes)),
        "volume_mae_pct_mean": float(np.mean(volume_maes)),
        "volume_mae_pct_worst": float(np.max(volume_maes)),
        "top_max_pct_worst": float(np.max(top_maxes)),
        "all_top_cases_below_5_pct": bool(np.all(np.array(top_maes) < 5.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--seed", type=int, default=20260415)
    parser.add_argument("--n-calibration", type=int, default=4)
    parser.add_argument("--n-holdout", type=int, default=4)
    parser.add_argument("--ne-x", type=int, default=8)
    parser.add_argument("--ne-y", type=int, default=8)
    parser.add_argument("--ne-z", type=int, default=4)
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument("--out", default=str(GRAPHS_DATA_DIR / "compliance_calibration.json"))
    args = parser.parse_args()

    ensure_output_dirs()
    device = select_device()
    pinn, model_path = load_pinn(device, args.model_path)

    random_cases = random_interior_cases(args.n_calibration + args.n_holdout, args.seed)
    calibration_cases = [
        ThreeLayerCase("supervised_extreme_soft_bottom", 1.0, 10.0, 10.0, 0.10, 0.02, 0.02),
        ThreeLayerCase("supervised_extreme_soft_middle", 10.0, 1.0, 10.0, 0.02, 0.10, 0.02),
        *random_cases[: args.n_calibration],
    ]
    holdout_cases = random_cases[args.n_calibration :]

    calibration = [_prepare_case(pinn, device, case, args.ne_x, args.ne_y, args.ne_z) for case in calibration_cases]
    holdout = [_prepare_case(pinn, device, case, args.ne_x, args.ne_y, args.ne_z) for case in holdout_cases]

    def objective(params: np.ndarray) -> float:
        m = _metrics(calibration, params)
        return m["top_mae_pct_worst"] + 0.25 * m["top_mae_pct_mean"] + 0.1 * m["volume_mae_pct_mean"]

    initial = np.array(
        [
            float(getattr(config, "DISPLACEMENT_COMPLIANCE_SCALE", 1.0)),
            float(getattr(config, "E_COMPLIANCE_POWER", 0.95)),
            float(getattr(config, "THICKNESS_COMPLIANCE_ALPHA", 3.0)),
        ]
    )
    bounds = [(0.1, 3.0), (0.0, 1.5), (0.0, 4.5)]
    de = opt.differential_evolution(objective, bounds=bounds, seed=args.seed, maxiter=args.maxiter, polish=False)
    local = opt.minimize(objective, de.x, method="Nelder-Mead", options={"maxiter": 200, "xatol": 1e-4, "fatol": 1e-4})
    best = np.clip(local.x if local.success else de.x, [b[0] for b in bounds], [b[1] for b in bounds])

    payload = {
        "model_path": str(model_path),
        "mesh": {"ne_x": args.ne_x, "ne_y": args.ne_y, "ne_z": args.ne_z},
        "initial_params": {
            "PINN_DISPLACEMENT_COMPLIANCE_SCALE": float(initial[0]),
            "PINN_E_COMPLIANCE_POWER": float(initial[1]),
            "PINN_THICKNESS_COMPLIANCE_ALPHA": float(initial[2]),
        },
        "tuned_params": {
            "PINN_DISPLACEMENT_COMPLIANCE_SCALE": float(best[0]),
            "PINN_E_COMPLIANCE_POWER": float(best[1]),
            "PINN_THICKNESS_COMPLIANCE_ALPHA": float(best[2]),
        },
        "calibration_metrics_initial": _metrics(calibration, initial),
        "calibration_metrics_tuned": _metrics(calibration, best),
        "holdout_metrics_initial": _metrics(holdout, initial) if holdout else None,
        "holdout_metrics_tuned": _metrics(holdout, best) if holdout else None,
        "note": "Use the tuned PINN_* env vars for evaluation only after reporting this calibration protocol.",
    }
    write_json(Path(args.out), payload)
    print(f"Wrote {args.out}")
    print("Tuned params:")
    for key, value in payload["tuned_params"].items():
        print(f"  {key}={value:.8g}")
    print(f"Holdout tuned top worst: {payload['holdout_metrics_tuned']['top_mae_pct_worst']:.2f}%")


if __name__ == "__main__":
    main()
