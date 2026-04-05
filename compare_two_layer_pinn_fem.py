"""Validation script comparing 2-layer PINN predictions against FEA solutions."""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PINN_WORKFLOW_DIR = os.path.join(REPO_ROOT, "pinn-workflow")
FEA_SOLVER_DIR = os.path.join(REPO_ROOT, "fea-workflow", "solver")

if PINN_WORKFLOW_DIR not in sys.path:
    sys.path.insert(0, PINN_WORKFLOW_DIR)
if FEA_SOLVER_DIR not in sys.path:
    sys.path.insert(0, FEA_SOLVER_DIR)

import pinn_config as config
import model
import fem_solver


def _load_pinn(device):
    pinn = model.MultiLayerPINN().to(device)
    model_path = os.path.join(PINN_WORKFLOW_DIR, "pinn_model.pth")
    sd = torch.load(model_path, map_location=device, weights_only=True)
    sd = model.adapt_legacy_state_dict(sd, pinn.state_dict())
    pinn.load_state_dict(sd, strict=False)
    pinn.eval()
    return pinn


def _run_fea(e1, e2, t1, t2, ne_x=10, ne_y=10, ne_z=4):
    thickness = float(t1) + float(t2)
    cfg = {
        "geometry": {
            "Lx": config.Lx,
            "Ly": config.Ly,
            "H": thickness,
            "ne_x": int(ne_x),
            "ne_y": int(ne_y),
            "ne_z": int(ne_z),
        },
        "material": {
            "E_layers": [float(e1), float(e2)],
            "t_layers": [float(t1), float(t2)],
            "nu": config.nu_vals[0],
        },
        "load_patch": {
            "pressure": config.p0,
            "x_start": config.LOAD_PATCH_X[0] / config.Lx,
            "x_end": config.LOAD_PATCH_X[1] / config.Lx,
            "y_start": config.LOAD_PATCH_Y[0] / config.Ly,
            "y_end": config.LOAD_PATCH_Y[1] / config.Ly,
        },
    }
    return fem_solver.solve_three_layer_fem(cfg)


def _ref_params():
    return (
        float(getattr(config, "RESTITUTION_REF", 0.5)),
        float(getattr(config, "FRICTION_REF", 0.3)),
        float(getattr(config, "IMPACT_VELOCITY_REF", 1.0)),
    )


def _u_from_v(v, pts):
    e_scale = (pts[:, 3:4] + pts[:, 5:6]) / 2.0
    t_scale = pts[:, 4:5] + pts[:, 6:7]
    e_pow = float(getattr(config, "E_COMPLIANCE_POWER", 1.0))
    alpha = float(getattr(config, "THICKNESS_COMPLIANCE_ALPHA", 0.0))
    scale = float(getattr(config, "DISPLACEMENT_COMPLIANCE_SCALE", 1.0))
    h_ref = float(getattr(config, "H", 1.0))
    return scale * v / (e_scale ** e_pow) * (h_ref / np.clip(t_scale, 1e-8, None)) ** alpha


def _predict_pinn(pinn, device, x_flat, y_flat, z_flat, e1, e2, t1, t2):
    r_ref, mu_ref, v0_ref = _ref_params()
    pts = np.stack(
        [
            x_flat,
            y_flat,
            z_flat,
            np.full_like(x_flat, float(e1)),
            np.full_like(x_flat, float(t1)),
            np.full_like(x_flat, float(e2)),
            np.full_like(x_flat, float(t2)),
            np.full_like(x_flat, r_ref),
            np.full_like(x_flat, mu_ref),
            np.full_like(x_flat, v0_ref),
        ],
        axis=1,
    )
    with torch.no_grad():
        v = pinn(torch.tensor(pts, dtype=torch.float32, device=device)).cpu().numpy()
    return _u_from_v(v, pts)


def _mae_pct(u_pred, u_ref):
    mae = float(np.mean(np.abs(u_pred - u_ref)))
    denom = float(np.max(np.abs(u_ref)))
    return 100.0 * mae / denom if denom > 0 else 0.0


def main():
    if os.environ.get("PINN_FORCE_CPU", "0") == "1":
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    output_dir = os.path.join(PINN_WORKFLOW_DIR, "visualization")
    os.makedirs(output_dir, exist_ok=True)

    pinn = _load_pinn(device)

    e_range = getattr(config, "E_RANGE", [1.0, 10.0])
    e_values = [float(e_range[0]), float(e_range[-1])]
    t_values = [0.05, 0.08]

    thickness = float(t_values[0]) + float(t_values[0])
    x_nodes, y_nodes, z_nodes, u_fea = _run_fea(1.0, 10.0, t_values[0], t_values[0])
    x_nodes = np.array(x_nodes)
    y_nodes = np.array(y_nodes)
    z_nodes = np.array(z_nodes)

    x_grid, y_grid = np.meshgrid(x_nodes, y_nodes, indexing="ij")
    top_z = np.full(x_grid.size, thickness)
    u_pinn_top = _predict_pinn(
        pinn, device, x_grid.ravel(), y_grid.ravel(), top_z, 1.0, 10.0, t_values[0], t_values[0]
    ).reshape(len(x_nodes), len(y_nodes), 3)

    u_z_fea_top = u_fea[:, :, -1, 2]
    u_z_pinn_top = u_pinn_top[:, :, 2]
    abs_err_top = np.abs(u_z_pinn_top - u_z_fea_top)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    vmin = float(min(np.min(u_z_fea_top), np.min(u_z_pinn_top)))
    vmax = float(max(np.max(u_z_fea_top), np.max(u_z_pinn_top)))

    c0 = axes[0].contourf(x_grid, y_grid, u_z_fea_top, levels=50, cmap="jet", vmin=vmin, vmax=vmax)
    plt.colorbar(c0, ax=axes[0])
    axes[0].set_title("Two-Layer FEA\n(E1=1, E2=10)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    c1 = axes[1].contourf(x_grid, y_grid, u_z_pinn_top, levels=50, cmap="jet", vmin=vmin, vmax=vmax)
    plt.colorbar(c1, ax=axes[1])
    axes[1].set_title("Two-Layer PINN")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    c2 = axes[2].contourf(x_grid, y_grid, abs_err_top, levels=50, cmap="magma")
    plt.colorbar(c2, ax=axes[2])
    axes[2].set_title(f"Abs Error\nMAE={float(np.mean(abs_err_top)):.5f}")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "two_layer_comparison_top.png"), dpi=160)
    plt.close(fig)

    cases = []
    fea_cache = {}
    for t1 in t_values:
        for t2 in t_values:
            for e1 in e_values:
                for e2 in e_values:
                    key = (e1, e2, t1, t2)
                    if key not in fea_cache:
                        fea_cache[key] = _run_fea(e1, e2, t1, t2, ne_x=8, ne_y=8, ne_z=4)
                    x_nodes, y_nodes, _, u_fea = fea_cache[key]
                    x_nodes = np.array(x_nodes)
                    y_nodes = np.array(y_nodes)
                    thick = float(t1) + float(t2)
                    x_grid, y_grid = np.meshgrid(x_nodes, y_nodes, indexing="ij")
                    u_pinn_top = _predict_pinn(
                        pinn, device, x_grid.ravel(), y_grid.ravel(), np.full(x_grid.size, thick),
                        e1, e2, t1, t2
                    ).reshape(len(x_nodes), len(y_nodes), 3)
                    u_z_fea_top = np.array(u_fea)[:, :, -1, 2]
                    u_z_pinn_top = u_pinn_top[:, :, 2]
                    cases.append(((e1, e2, t1, t2), _mae_pct(u_z_pinn_top, u_z_fea_top)))

    mae_pcts = np.array([v for _, v in cases], dtype=float)
    worst_idx = int(np.argmax(mae_pcts))
    worst_case, worst_mae = cases[worst_idx]
    mean_mae = float(np.mean(mae_pcts)) if len(mae_pcts) else 0.0

    for (e1, e2, t1, t2), mae in cases:
        print(f"2-layer case: E1={e1:g}, E2={e2:g}, t1={t1:g}, t2={t2:g} => Top-surface MAE={mae:.2f}%")
    print(f"Two-layer sweep mean MAE={mean_mae:.2f}%")
    print(f"Two-layer sweep worst MAE={worst_mae:.2f}%")
    print(f"Worst case: E1={worst_case[0]:g}, E2={worst_case[1]:g}, t1={worst_case[2]:g}, t2={worst_case[3]:g}")
    print(f"Saved output to {output_dir}")


if __name__ == "__main__":
    main()
