from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))
os.environ.setdefault("PYTHONPYCACHEPREFIX", str(REPO_ROOT / ".pycache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load_model_and_data():
    sys.path.insert(0, str(REPO_ROOT / "pinn-workflow"))
    from surrogate_workflow import config  # noqa: E402
    from surrogate_workflow import baseline  # noqa: E402
    from surrogate_workflow import data as data_utils  # noqa: E402
    from surrogate_workflow import surrogate  # noqa: E402
    from surrogate_workflow import validate  # noqa: E402

    import torch  # noqa: E402

    device = torch.device("cpu")
    dataset = data_utils.load_dataset(config.DATASET_PATH)

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

    return config, baseline, data_utils, surrogate, validate, device, dataset, payload, model


def _predict_mu(config, data_utils, surrogate, validate, device, dataset, payload, model, mu: np.ndarray) -> float:
    x_norm, _, _ = data_utils.normalize_inputs(mu.reshape(1, -1), config.DESIGN_RANGES)
    y_norm = surrogate.predict(model, x_norm, device)[0]
    return float(
        validate.denormalize_y(
            y_norm,
            dataset["y_min"],
            dataset["y_max"],
            dataset.get("y_transform", payload.get("y_transform", "identity")),
            dataset.get("y_eps", payload.get("y_eps", 1e-12)),
        )
    )


def _relative_error(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.abs(y_true - y_pred) / (np.abs(y_true) + eps)


def _plot_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    case_points: list[tuple[str, float, float]] | None,
    grid_points: list[tuple[float, float]] | None,
    png_path: Path,
    pdf_path: Path,
) -> None:
    abs_err = np.abs(y_true - y_pred)
    rel_err = _relative_error(y_true, y_pred)

    mae = float(np.mean(abs_err))
    p50 = float(np.percentile(rel_err, 50) * 100.0)
    p95 = float(np.percentile(rel_err, 95) * 100.0)
    worst = float(np.max(rel_err) * 100.0)

    case_worst = None
    if case_points or grid_points:
        y_case_true = []
        y_case_pred = []
        if case_points:
            for _name, yt, yp in case_points:
                y_case_true.append(float(yt))
                y_case_pred.append(float(yp))
        if grid_points:
            for yt, yp in grid_points:
                y_case_true.append(float(yt))
                y_case_pred.append(float(yp))
        if y_case_true:
            case_worst = float(np.max(_relative_error(np.asarray(y_case_true), np.asarray(y_case_pred))) * 100.0)

    plt.figure(figsize=(7.0, 7.0))
    plt.scatter(y_true, y_pred, s=16, alpha=0.50, edgecolors="none", label="Test samples")
    min_v = float(min(np.min(y_true), np.min(y_pred)))
    max_v = float(max(np.max(y_true), np.max(y_pred)))

    if grid_points:
        gx = np.asarray([p[0] for p in grid_points], dtype=float)
        gy = np.asarray([p[1] for p in grid_points], dtype=float)
        plt.scatter(gx, gy, s=80, marker="x", linewidths=2.0, color="tab:red", label="Thin-stack E-grid")
        min_v = float(min(min_v, np.min(gx), np.min(gy)))
        max_v = float(max(max_v, np.max(gx), np.max(gy)))

    if case_points:
        colors = ["tab:orange", "tab:green", "tab:purple"]
        for i, (name, yt, yp) in enumerate(case_points):
            c = colors[i % len(colors)]
            plt.scatter([yt], [yp], s=110, marker="o", facecolors="none", edgecolors=c, linewidths=2.5, label=name)
            plt.annotate(
                name.replace("three_layer_", "").replace("_", " "),
                xy=(yt, yp),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=9,
                color=c,
            )
            min_v = float(min(min_v, yt, yp))
            max_v = float(max(max_v, yt, yp))

    plt.plot([min_v, max_v], [min_v, max_v], "k--", linewidth=1)
    plt.xlabel("Baseline y")
    plt.ylabel("Surrogate y")
    plt.title("Surrogate vs Baseline (Test)")
    ax = plt.gca()
    extra = f"\nCase-check worst rel. err: {case_worst:.2f}%" if case_worst is not None else ""
    ax.text(
        0.02,
        0.98,
        f"MAE: {mae:.3e}\nRel. err (p50/p95/worst): {p50:.2f}% / {p95:.2f}% / {worst:.2f}%\nN={int(y_true.size)}{extra}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    plt.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=200)
    plt.savefig(pdf_path)
    plt.close()


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    return True


def main() -> None:
    figs = REPO_ROOT / "graphs" / "figures"
    config, baseline, data_utils, surrogate, validate, device, dataset, payload, model = _load_model_and_data()

    n_samples = dataset["x_norm"].shape[0]
    _train_idx, _val_idx, test_idx = data_utils.split_indices(
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

    case_points: list[tuple[str, float, float]] = []
    grid_points: list[tuple[float, float]] = []
    if list(config.DESIGN_RANGES.keys()) == ["E1", "t1", "E2", "t2", "E3", "t3"]:
        e_low, e_high = config.DESIGN_RANGES["E1"]
        t1_low, t1_high = config.DESIGN_RANGES["t1"]
        t2_low, t2_high = config.DESIGN_RANGES["t2"]
        t3_low, t3_high = config.DESIGN_RANGES["t3"]

        cases = [
            ("three_layer_soft_bottom", np.array([e_low, t1_high, e_high, t2_low, e_high, t3_low], dtype=float)),
            ("three_layer_soft_middle", np.array([e_high, t1_low, e_low, t2_high, e_high, t3_low], dtype=float)),
            ("three_layer_soft_top", np.array([e_high, t1_low, e_high, t2_low, e_low, t3_high], dtype=float)),
        ]
        for name, mu in cases:
            yt = float(baseline.compute_response(mu))
            yp = _predict_mu(config, data_utils, surrogate, validate, device, dataset, payload, model, mu)
            case_points.append((name, yt, yp))

        for e1 in (e_low, e_high):
            for e2 in (e_low, e_high):
                for e3 in (e_low, e_high):
                    mu = np.array([e1, t1_low, e2, t2_low, e3, t3_low], dtype=float)
                    yt = float(baseline.compute_response(mu))
                    yp = _predict_mu(config, data_utils, surrogate, validate, device, dataset, payload, model, mu)
                    grid_points.append((yt, yp))

    _plot_scatter(
        y_true[test_idx],
        y_pred[test_idx],
        case_points=case_points,
        grid_points=grid_points,
        png_path=figs / "fig_surrogate_scatter.png",
        pdf_path=figs / "fig_surrogate_scatter.pdf",
    )

    plots = REPO_ROOT / "pinn-workflow" / "surrogate_workflow" / "outputs" / "plots"
    _copy_if_exists(plots / "test_rel_error_hist.png", figs / "fig_surrogate_test_rel_error_hist.png")
    _copy_if_exists(plots / "grid_E3_vs_E1.png", figs / "fig_surrogate_grid_E3_vs_E1.png")
    _copy_if_exists(plots / "trend_fidelity.png", figs / "fig_surrogate_trend_fidelity.png")

    print(f"Wrote figures to {figs}")


if __name__ == "__main__":
    main()
