import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

if __package__ is None:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from surrogate_workflow import config
from surrogate_workflow import baseline
from surrogate_workflow import data as data_utils
from surrogate_workflow import surrogate
from surrogate_workflow import validate


def _load_trained_model(device: torch.device):
    try:
        payload = torch.load(config.MODEL_PATH, map_location=device, weights_only=True)
    except Exception:
        payload = torch.load(config.MODEL_PATH, map_location=device, weights_only=False)
    model = surrogate.MLPRegressor(
        input_dim=len(payload["param_names"]),
        output_dim=1,
        hidden_layers=int(payload["config"]["hidden_layers"]),
        hidden_units=int(payload["config"]["hidden_units"]),
        activation=str(payload["config"]["activation"]),
        fourier_dim=int(payload["config"].get("fourier_dim", 0)),
        fourier_scale=float(payload["config"].get("fourier_scale", 1.0)),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def _relative_error(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.abs(y_true - y_pred) / (np.abs(y_true) + eps)


def _plot_rel_error_hist(rel_err: np.ndarray, out_path: str):
    plt.figure(figsize=(7, 4))
    plt.hist(rel_err * 100.0, bins=40, alpha=0.9, color="tab:blue")
    plt.xlabel("Relative error (%)")
    plt.ylabel("Count")
    plt.title("Surrogate Relative Error (Test)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_param_grid(
    model,
    dataset: dict,
    device: torch.device,
    out_path: str,
    x_param: str,
    y_param: str,
    n: int,
):
    if x_param not in config.DESIGN_RANGES or y_param not in config.DESIGN_RANGES:
        raise ValueError(f"Params must be in {list(config.DESIGN_RANGES.keys())}")
    if x_param == y_param:
        raise ValueError("x-param and y-param must differ.")

    x_low, x_high = config.DESIGN_RANGES[x_param]
    y_low, y_high = config.DESIGN_RANGES[y_param]
    x_vals = np.linspace(x_low, x_high, n)
    y_vals = np.linspace(y_low, y_high, n)

    mu_mid = config.mid_design()
    param_names = list(config.DESIGN_RANGES.keys())
    idx_x = param_names.index(x_param)
    idx_y = param_names.index(y_param)

    y_true = np.zeros((n, n), dtype=float)
    y_pred = np.zeros((n, n), dtype=float)
    for i, xv in enumerate(x_vals):
        for j, yv in enumerate(y_vals):
            mu = mu_mid.copy()
            mu[idx_x] = xv
            mu[idx_y] = yv
            y_true[i, j] = baseline.compute_response(mu)
            x_norm, _, _ = data_utils.normalize_inputs(mu.reshape(1, -1), config.DESIGN_RANGES)
            y_norm = surrogate.predict(model, x_norm, device)[0]
            y_pred[i, j] = validate.denormalize_y(
                y_norm,
                dataset["y_min"],
                dataset["y_max"],
                dataset.get("y_transform", "identity"),
                dataset.get("y_eps", 1e-12),
            )

    rel_err = _relative_error(y_true, y_pred)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    im0 = axes[0].imshow(y_true, origin="lower", cmap="viridis")
    axes[0].set_title("Baseline (PINN) y")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(y_pred, origin="lower", cmap="viridis")
    axes[1].set_title("Surrogate y")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(rel_err * 100.0, origin="lower", cmap="magma")
    axes[2].set_title("Rel error (%)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks([0, n - 1])
        ax.set_yticks([0, n - 1])
        ax.set_xticklabels([f"{x_low:g}", f"{x_high:g}"])
        ax.set_yticklabels([f"{y_low:g}", f"{y_high:g}"])
        ax.set_xlabel(x_param)
        ax.set_ylabel(y_param)

    fig.suptitle(f"Mid-design grid: {y_param} vs {x_param}", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

def _predict_single(model, dataset: dict, device: torch.device, mu: np.ndarray) -> float:
    x_norm, _, _ = data_utils.normalize_inputs(mu.reshape(1, -1), config.DESIGN_RANGES)
    y_norm = surrogate.predict(model, x_norm, device)[0]
    return float(
        validate.denormalize_y(
            y_norm,
            dataset["y_min"],
            dataset["y_max"],
            dataset.get("y_transform", "identity"),
            dataset.get("y_eps", 1e-12),
        )
    )


def main():
    parser = argparse.ArgumentParser(description="Verify Phase 1 surrogate outputs")
    parser.add_argument("--device", default=None, help="Override device (cpu/cuda/mps)")
    parser.add_argument("--grid", type=int, default=11, help="Grid resolution for verification plot")
    parser.add_argument("--x-param", default="E1", help="X-axis design param for grid plot")
    parser.add_argument("--y-param", default=None, help="Y-axis design param for grid plot")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    dataset = data_utils.load_dataset(config.DATASET_PATH)
    model, _payload = _load_trained_model(device)

    if args.y_param is None:
        # Default to the last layer's modulus so it works for 2-layer and 3-layer modes.
        e_params = [k for k in config.DESIGN_RANGES.keys() if k.startswith("E")]
        args.y_param = e_params[-1] if e_params else list(config.DESIGN_RANGES.keys())[-1]

    n_samples = dataset["x_norm"].shape[0]
    train_idx, val_idx, test_idx = data_utils.split_indices(
        n_samples,
        float(config.TRAIN_FRACTION),
        float(config.VAL_FRACTION),
        int(config.SEED),
        n_anchors=int(dataset.get("n_anchors", 0)),
    )

    y_norm = surrogate.predict(model, dataset["x_norm"], device)
    y_pred = validate.denormalize_y(
        y_norm,
        dataset["y_min"],
        dataset["y_max"],
        dataset.get("y_transform", _payload.get("y_transform", "identity")),
        dataset.get("y_eps", _payload.get("y_eps", 1e-12)),
    )
    y_true = dataset["y_raw"]

    rel_err = _relative_error(y_true[test_idx], y_pred[test_idx])
    mae = float(np.mean(np.abs(y_true[test_idx] - y_pred[test_idx])))
    p50 = float(np.percentile(rel_err, 50) * 100.0)
    p95 = float(np.percentile(rel_err, 95) * 100.0)
    worst = float(np.max(rel_err) * 100.0)
    print(f"Test MAE: {mae:.6e}")
    print(f"Test rel error p50/p95/worst: {p50:.2f}% / {p95:.2f}% / {worst:.2f}%")

    # Corner check (targets the thickness/elasticity extremes used in PINN sweeps).
    param_names = list(config.DESIGN_RANGES.keys())
    lows = [config.DESIGN_RANGES[p][0] for p in param_names]
    highs = [config.DESIGN_RANGES[p][1] for p in param_names]
    corners = []
    for mask in range(1 << len(param_names)):
        corner = []
        for i in range(len(param_names)):
            corner.append(highs[i] if (mask & (1 << i)) else lows[i])
        corners.append(np.asarray(corner, dtype=float))
    y_corner_true = np.asarray([baseline.compute_response(mu) for mu in corners], dtype=float)
    x_corner_norm, _, _ = data_utils.normalize_inputs(np.stack(corners, axis=0), config.DESIGN_RANGES)
    y_corner_norm = surrogate.predict(model, x_corner_norm, device)
    y_corner_pred = validate.denormalize_y(
        y_corner_norm,
        dataset["y_min"],
        dataset["y_max"],
        dataset.get("y_transform", _payload.get("y_transform", "identity")),
        dataset.get("y_eps", _payload.get("y_eps", 1e-12)),
    )
    corner_rel = _relative_error(y_corner_true, y_corner_pred)
    worst_idx = int(np.argmax(corner_rel)) if corner_rel.size else -1
    worst_pct = float(np.max(corner_rel) * 100.0) if corner_rel.size else 0.0
    print(f"Corner rel error worst: {worst_pct:.2f}%  (n={len(corners)})")
    if worst_idx >= 0:
        mu_worst = corners[worst_idx]
        print(
            f"  Worst corner mu={mu_worst.tolist()} y_true={float(y_corner_true[worst_idx]):.6g} "
            f"y_pred={float(y_corner_pred[worst_idx]):.6g}"
        )

    # Two-layer "sweep case" check: matches the extreme settings used in compare_two_layer_pinn_fem.py.
    if len(param_names) == 4:
        e_low, e_high = config.DESIGN_RANGES["E1"]
        t1_low, t1_high = config.DESIGN_RANGES["t1"]
        t2_low, t2_high = config.DESIGN_RANGES["t2"]
        sweep_cases = [
            ("two_layer_soft_bottom", np.array([e_low, t1_low, e_high, t2_high], dtype=float)),
            ("two_layer_soft_top", np.array([e_high, t1_high, e_low, t2_low], dtype=float)),
        ]
        for name, mu in sweep_cases:
            y_true = float(baseline.compute_response(mu))
            y_pred = _predict_single(model, dataset, device, mu)
            rel = float(_relative_error(np.array([y_true]), np.array([y_pred]))[0] * 100.0)
            print(f"{name} rel error: {rel:.2f}% (y_true={y_true:.6g}, y_pred={y_pred:.6g})")

        # E-grid at (t1_min, t2_min): 2x2 E sweep points.
        grid = []
        for e1 in (e_low, e_high):
            for e2 in (e_low, e_high):
                grid.append(np.array([e1, t1_low, e2, t2_low], dtype=float))
        y_true = np.asarray([baseline.compute_response(mu) for mu in grid], dtype=float)
        y_pred = np.asarray([_predict_single(model, dataset, device, mu) for mu in grid], dtype=float)
        rel = _relative_error(y_true, y_pred)
        print(f"Two-layer E-grid (t1_min,t2_min) worst rel error: {float(np.max(rel) * 100.0):.2f}%")

    _plot_rel_error_hist(rel_err, os.path.join(config.PLOTS_DIR, "test_rel_error_hist.png"))
    _plot_param_grid(
        model,
        dataset,
        device,
        os.path.join(config.PLOTS_DIR, f"grid_{args.y_param}_vs_{args.x_param}.png"),
        x_param=str(args.x_param),
        y_param=str(args.y_param),
        n=int(args.grid),
    )

    print(f"Wrote plots to {config.PLOTS_DIR}")


if __name__ == "__main__":
    main()
