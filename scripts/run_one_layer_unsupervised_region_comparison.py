"""Compare one-layer PINN and FEM in regions not directly supervised by FEM data."""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parents[1] / ".cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from one_layer_experiment_utils import (
    GRAPHS_DATA_DIR,
    GRAPHS_FIG_DIR,
    OneLayerCase,
    ensure_output_dirs,
    evaluate_case_grid,
    is_supervised_parameter_case,
    load_pinn,
    mae_pct,
    max_pct,
    random_interior_cases,
    rows_to_csv,
    select_device,
    write_json,
)


def _spatial_labels(z_nodes: np.ndarray, case: OneLayerCase) -> list[str]:
    labels = []
    tol = 0.5 * float(np.min(np.diff(z_nodes))) if len(z_nodes) > 1 else 1e-9
    for z in z_nodes:
        if abs(float(z) - case.thickness) <= tol:
            labels.append("top_surface")
        elif abs(float(z)) <= tol:
            labels.append("bottom_surface")
        else:
            labels.append("interior_volume")
    return labels


def _plot_cross_section(result: dict, out_path: Path) -> None:
    case: OneLayerCase = result["case"]
    x_nodes = result["x_nodes"]
    y_nodes = result["y_nodes"]
    z_nodes = result["z_nodes"]
    mid_y = len(y_nodes) // 2
    xg, zg = np.meshgrid(x_nodes, z_nodes, indexing="ij")
    err = np.abs(result["u_pinn"][:, mid_y, :, 2] - result["u_fem"][:, mid_y, :, 2])

    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    im = ax.contourf(xg, zg, err, levels=40, cmap="magma")
    fig.colorbar(im, ax=ax, label="abs u_z error")
    ax.axhline(case.thickness, color="cyan", linestyle=":", linewidth=1.0, label="top surface")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(f"One-layer unsupervised-region error, {case.case_id}")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--seed", type=int, default=20260415)
    parser.add_argument("--ne-x", type=int, default=10)
    parser.add_argument("--ne-y", type=int, default=10)
    parser.add_argument("--ne-z", type=int, default=6)
    parser.add_argument("--out-csv", default=str(GRAPHS_DATA_DIR / "one_layer_unsupervised_region_comparison.csv"))
    parser.add_argument("--out-summary", default=str(GRAPHS_DATA_DIR / "one_layer_unsupervised_region_summary.json"))
    args = parser.parse_args()

    ensure_output_dirs()
    device = select_device()
    pinn, model_path = load_pinn(device, args.model_path)

    cases = [
        OneLayerCase("one_layer_supervised_param_mid", 5.0, 0.10),
        random_interior_cases(1, args.seed)[0],
    ]
    rows = []
    for case in cases:
        result = evaluate_case_grid(pinn, device, case, args.ne_x, args.ne_y, args.ne_z)
        parameter_region = "supervised_parameter_grid" if is_supervised_parameter_case(case) else "unsupervised_random_parameter"
        labels = _spatial_labels(result["z_nodes"], case)
        groups = defaultdict(list)
        refs = defaultdict(list)
        for k, label in enumerate(labels):
            groups[label].append(result["u_pinn"][:, :, k, :].reshape(-1, 3))
            refs[label].append(result["u_fem"][:, :, k, :].reshape(-1, 3))

        for label, chunks in groups.items():
            pred = np.concatenate(chunks, axis=0)
            ref = np.concatenate(refs[label], axis=0)
            rows.append(
                {
                    "case_id": case.case_id,
                    "parameter_region": parameter_region,
                    "spatial_region": label,
                    "E": f"{case.E:.8g}",
                    "thickness": f"{case.thickness:.8g}",
                    "mae_pct": f"{mae_pct(pred, ref):.6f}",
                    "max_pct": f"{max_pct(pred, ref):.6f}",
                    "n_points": str(len(pred)),
                }
            )
        _plot_cross_section(result, GRAPHS_FIG_DIR / f"one_layer_unsupervised_region_{case.case_id}.png")
        print(f"{case.case_id}: top={result['top_uz_mae_pct']:.2f}% volume={result['volume_mae_pct']:.2f}%")

    rows_to_csv(
        Path(args.out_csv),
        ["case_id", "parameter_region", "spatial_region", "E", "thickness", "mae_pct", "max_pct", "n_points"],
        rows,
    )
    write_json(
        Path(args.out_summary),
        {
            "model_path": str(model_path),
            "mesh": {"ne_x": args.ne_x, "ne_y": args.ne_y, "ne_z": args.ne_z},
            "notes": [
                "The default one-layer checkpoint is pure-physics, so no locations were directly supervised by FEM during its training.",
                "Parameter-region labels indicate whether a case lies on the optional sparse-supervision grid.",
                "There is no material interface in the one-layer model; interface-continuity ablation is not applicable.",
            ],
            "rows": rows,
        },
    )
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_summary}")


if __name__ == "__main__":
    main()
