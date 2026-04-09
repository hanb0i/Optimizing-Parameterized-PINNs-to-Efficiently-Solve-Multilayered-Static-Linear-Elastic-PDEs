from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from _common import REPO_ROOT, apply_ieee_style, save_figure, watermark_placeholder

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class SurrogateArtifacts:
    dataset_path: Path
    model_path: Path


def _relative_error(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.abs(y_true - y_pred) / (np.abs(y_true) + eps)


def _load_artifacts(root: Path, workflow_dir: str) -> SurrogateArtifacts:
    out_dir = root / workflow_dir / "surrogate_workflow" / "outputs"
    return SurrogateArtifacts(
        dataset_path=out_dir / "phase1_dataset.npz",
        model_path=out_dir / "surrogate_model.pt",
    )


def _import_stack(workflow_root: Path):
    if str(workflow_root) not in sys.path:
        sys.path.insert(0, str(workflow_root))
    from surrogate_workflow import config as scfg  # noqa: WPS433
    from surrogate_workflow import baseline  # noqa: WPS433
    from surrogate_workflow import data as data_utils  # noqa: WPS433
    from surrogate_workflow import surrogate as surrogate_lib  # noqa: WPS433
    from surrogate_workflow import validate  # noqa: WPS433
    import torch  # noqa: WPS433

    return scfg, baseline, data_utils, surrogate_lib, validate, torch


def _load_model(surrogate_lib, torch_mod, model_path: Path, device):
    try:
        payload = torch_mod.load(str(model_path), map_location=device, weights_only=True)
    except Exception:
        payload = torch_mod.load(str(model_path), map_location=device, weights_only=False)

    model = surrogate_lib.MLPRegressor(
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


def _predict_single(scfg, data_utils, surrogate_lib, validate, model, dataset: dict, device, mu: np.ndarray) -> float:
    x_norm, _, _ = data_utils.normalize_inputs(mu.reshape(1, -1), scfg.DESIGN_RANGES)
    y_norm = surrogate_lib.predict(model, x_norm, device)[0]
    y_pred = validate.denormalize_y(
        y_norm,
        dataset["y_min"],
        dataset["y_max"],
        dataset.get("y_transform", "identity"),
        dataset.get("y_eps", 1e-12),
    )
    return float(np.asarray(y_pred).reshape(-1)[0])


def _casecheck_worst(scfg, baseline, data_utils, surrogate_lib, validate, model, dataset: dict, device) -> float:
    params = list(scfg.DESIGN_PARAMS)
    if params == ["E1", "t1", "E2", "t2"]:
        e_low, e_high = scfg.DESIGN_RANGES["E1"]
        t1_low, t1_high = scfg.DESIGN_RANGES["t1"]
        t2_low, t2_high = scfg.DESIGN_RANGES["t2"]
        cases = [
            np.array([e_low, t1_low, e_high, t2_high], dtype=float),
            np.array([e_high, t1_high, e_low, t2_low], dtype=float),
        ]
        grid = [np.array([e1, t1_low, e2, t2_low], dtype=float) for e1 in (e_low, e_high) for e2 in (e_low, e_high)]
        mus = cases + grid
    elif params == ["E1", "t1", "E2", "t2", "E3", "t3"]:
        e_low, e_high = scfg.DESIGN_RANGES["E1"]
        t1_low, t1_high = scfg.DESIGN_RANGES["t1"]
        t2_low, t2_high = scfg.DESIGN_RANGES["t2"]
        t3_low, t3_high = scfg.DESIGN_RANGES["t3"]
        cases = [
            np.array([e_low, t1_high, e_high, t2_low, e_high, t3_low], dtype=float),
            np.array([e_high, t1_low, e_low, t2_high, e_high, t3_low], dtype=float),
            np.array([e_high, t1_low, e_high, t2_low, e_low, t3_high], dtype=float),
        ]
        grid = [
            np.array([e1, t1_low, e2, t2_low, e3, t3_low], dtype=float)
            for e1 in (e_low, e_high)
            for e2 in (e_low, e_high)
            for e3 in (e_low, e_high)
        ]
        mus = cases + grid
    else:
        return float("nan")

    y_true = np.asarray([baseline.compute_response(mu) for mu in mus], dtype=float)
    y_pred = np.asarray(
        [_predict_single(scfg, data_utils, surrogate_lib, validate, model, dataset, device, mu) for mu in mus],
        dtype=float,
    )
    rel = _relative_error(y_true, y_pred)
    return float(np.max(rel) * 100.0) if rel.size else float("nan")


def main() -> None:
    apply_ieee_style()

    art_3 = _load_artifacts(REPO_ROOT, "pinn-workflow")
    art_2 = _load_artifacts(REPO_ROOT, "pinn-workflow-2layer")

    fig, ax = plt.subplots(figsize=(6.2, 1.75))
    ax.axis("off")

    rows = []
    missing = []

    # Two-layer.
    if art_2.dataset_path.exists() and art_2.model_path.exists():
        scfg, baseline, data_utils, surrogate_lib, validate, torch_mod = _import_stack(REPO_ROOT / "pinn-workflow-2layer")
        device = torch_mod.device("cpu")
        dataset = data_utils.load_dataset(str(art_2.dataset_path))
        model, _payload = _load_model(surrogate_lib, torch_mod, art_2.model_path, device)
        worst = _casecheck_worst(scfg, baseline, data_utils, surrogate_lib, validate, model, dataset, device)
        rows.append(["Two-layer surrogate", f"{worst:.2f}"])
    else:
        rows.append(["Two-layer surrogate", "MISSING"])
        missing.append(f"- {art_2.dataset_path}")
        missing.append(f"- {art_2.model_path}")

    # Three-layer.
    if art_3.dataset_path.exists() and art_3.model_path.exists():
        scfg, baseline, data_utils, surrogate_lib, validate, torch_mod = _import_stack(REPO_ROOT / "pinn-workflow")
        device = torch_mod.device("cpu")
        dataset = data_utils.load_dataset(str(art_3.dataset_path))
        model, _payload = _load_model(surrogate_lib, torch_mod, art_3.model_path, device)
        worst = _casecheck_worst(scfg, baseline, data_utils, surrogate_lib, validate, model, dataset, device)
        rows.append(["Three-layer surrogate", f"{worst:.2f}"])
    else:
        rows.append(["Three-layer surrogate", "MISSING"])
        missing.append(f"- {art_3.dataset_path}")
        missing.append(f"- {art_3.model_path}")

    table = ax.table(
        cellText=rows,
        colLabels=["Configuration", "Case-Check Worst Rel. Error (%)"],
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.55, 0.45],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)
    table.scale(1.0, 1.2)

    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("0.95")

    ax.set_title("SURROGATE-VS-PINN VERIFICATION RESULTS", pad=10)

    if missing:
        watermark_placeholder(ax, "MISSING SURROGATE OUTPUTS")
        ax.text(
            0.02,
            -0.20,
            "Missing:\n" + "\n".join(missing[:6]) + ("\n..." if len(missing) > 6 else ""),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color="0.35",
        )

    out_paths = save_figure(fig, "fig_surrogate_verification_results")
    plt.close(fig)

    print("Wrote:")
    for p in out_paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
