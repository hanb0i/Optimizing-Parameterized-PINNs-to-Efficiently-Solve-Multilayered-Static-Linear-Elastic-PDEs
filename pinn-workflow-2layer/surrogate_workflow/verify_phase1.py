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


def _predict_single(model, dataset: dict, device: torch.device, mu: np.ndarray) -> float:
    x_norm, _, _ = data_utils.normalize_inputs(mu.reshape(1, -1), config.DESIGN_RANGES)
    y_norm = surrogate.predict(model, x_norm, device)[0]
    y_pred = validate.denormalize_y(
        y_norm,
        dataset["y_min"],
        dataset["y_max"],
        dataset.get("y_transform", "identity"),
        dataset.get("y_eps", 1e-12),
    )
    return float(np.asarray(y_pred).reshape(-1)[0])


def _two_layer_case_checks(model, dataset: dict, device: torch.device) -> float:
    params = list(config.DESIGN_PARAMS)
    if params != ["E1", "t1", "E2", "t2"]:
        raise ValueError(f"Unexpected DESIGN_PARAMS for two-layer case checks: {params}")

    e_low, e_high = config.DESIGN_RANGES["E1"]
    t1_low, t1_high = config.DESIGN_RANGES["t1"]
    t2_low, t2_high = config.DESIGN_RANGES["t2"]

    cases = [
        ("two_layer_soft_bottom", np.array([e_low, t1_low, e_high, t2_high], dtype=float)),
        ("two_layer_soft_top", np.array([e_high, t1_high, e_low, t2_low], dtype=float)),
    ]

    worst = 0.0
    for name, mu in cases:
        y_true = float(baseline.compute_response(mu))
        y_pred = float(_predict_single(model, dataset, device, mu))
        rel = float(_relative_error(np.array([y_true]), np.array([y_pred]))[0] * 100.0)
        worst = max(worst, rel)
        print(f"{name} rel error: {rel:.2f}% (y_true={y_true:.6g}, y_pred={y_pred:.6g})")

    # Thin-stack E grid at (t1_min, t2_min): 2x2 E sweep points.
    grid = []
    for e1 in (e_low, e_high):
        for e2 in (e_low, e_high):
            grid.append(np.array([e1, t1_low, e2, t2_low], dtype=float))
    y_true = np.asarray([baseline.compute_response(mu) for mu in grid], dtype=float)
    y_pred = np.asarray([_predict_single(model, dataset, device, mu) for mu in grid], dtype=float)
    rel = _relative_error(y_true, y_pred)
    worst_grid = float(np.max(rel) * 100.0) if rel.size else 0.0
    print(f"Two-layer thin-stack E-grid worst rel error: {worst_grid:.2f}% (n={len(grid)})")
    worst = max(worst, worst_grid)

    target = float(getattr(config, "TARGET_REL_ERR_PCT", 5.0))
    print(f"Two-layer case-check worst rel error: {worst:.2f}% (target {target:.2f}%)")
    return float(worst)


def main():
    parser = argparse.ArgumentParser(description="Verify Phase 1 surrogate outputs (2-layer)")
    parser.add_argument("--device", default=None, help="Override device (cpu/cuda/mps)")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    dataset = data_utils.load_dataset(config.DATASET_PATH)
    expected = list(getattr(config, "DESIGN_PARAMS", []))
    found = list(dataset.get("param_names", []))
    if expected and found != expected:
        raise ValueError(
            "Surrogate dataset does not match current DESIGN_PARAMS.\n"
            f"  expected: {expected}\n"
            f"  found:    {found}\n"
            "Re-run: python3 pinn-workflow-2layer/surrogate_workflow/run_phase1.py --regenerate"
        )
    model, payload = _load_trained_model(device)
    payload_params = list(payload.get("param_names", []))
    if expected and payload_params != expected:
        raise ValueError(
            "Surrogate model does not match current DESIGN_PARAMS.\n"
            f"  expected: {expected}\n"
            f"  found:    {payload_params}\n"
            "Re-run: python3 pinn-workflow-2layer/surrogate_workflow/run_phase1.py --regenerate"
        )

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
        dataset.get("y_transform", payload.get("y_transform", "identity")),
        dataset.get("y_eps", payload.get("y_eps", 1e-12)),
    )
    y_true = dataset["y_raw"]

    rel_err = _relative_error(y_true[test_idx], y_pred[test_idx])
    mae = float(np.mean(np.abs(y_true[test_idx] - y_pred[test_idx])))
    p50 = float(np.percentile(rel_err, 50) * 100.0)
    p95 = float(np.percentile(rel_err, 95) * 100.0)
    worst = float(np.max(rel_err) * 100.0)
    print(f"Test MAE: {mae:.6e}")
    print(f"Test rel error p50/p95/worst: {p50:.2f}% / {p95:.2f}% / {worst:.2f}%")

    _two_layer_case_checks(model, dataset, device)

    _plot_rel_error_hist(rel_err, os.path.join(config.PLOTS_DIR, "test_rel_error_hist.png"))
    print(f"Wrote plots to {config.PLOTS_DIR}")


if __name__ == "__main__":
    main()

