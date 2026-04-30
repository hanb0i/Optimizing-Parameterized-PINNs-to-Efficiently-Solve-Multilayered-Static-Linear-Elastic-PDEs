"""Fit simple post-hoc calibration (global multiplier per case)."""

import os
import sys
import json

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PINN_WORKFLOW_DIR = os.path.join(REPO_ROOT, "three-layer-workflow")
FEA_SOLVER_DIR = os.path.join(REPO_ROOT, "fea-workflow", "solver")

sys.path.insert(0, PINN_WORKFLOW_DIR)
sys.path.insert(0, FEA_SOLVER_DIR)

import pinn_config as config
import model
import fem_solver


def _ref_params():
    return (
        float(getattr(config, "RESTITUTION_REF", 0.5)),
        float(getattr(config, "FRICTION_REF", 0.3)),
        float(getattr(config, "IMPACT_VELOCITY_REF", 1.0)),
    )


def _load_pinn(device):
    pinn = model.MultiLayerPINN().to(device)
    model_path = os.path.join(PINN_WORKFLOW_DIR, "pinn_model.pth")
    sd = torch.load(model_path, map_location=device, weights_only=True)
    sd = model.adapt_legacy_state_dict(sd, pinn.state_dict())
    pinn.load_state_dict(sd, strict=False)
    pinn.eval()
    return pinn


def _predict_pinn(pinn, device, x, y, z, e1, e2, e3, t1, t2, t3):
    r, mu, v0 = _ref_params()
    pts = np.stack([x, y, z, np.full_like(x, e1), np.full_like(x, t1),
                    np.full_like(x, e2), np.full_like(x, t2),
                    np.full_like(x, e3), np.full_like(x, t3),
                    np.full_like(x, r), np.full_like(x, mu), np.full_like(x, v0)], axis=1)
    pts_t = torch.tensor(pts, dtype=torch.float32, device=device)
    with torch.no_grad():
        v = pinn(pts_t).cpu().numpy()
    e_scale = (pts[:, 3:4] + pts[:, 5:6] + pts[:, 7:8]) / 3.0
    t_scale = pts[:, 4:5] + pts[:, 6:7] + pts[:, 8:9]
    e_pow = float(getattr(config, "E_COMPLIANCE_POWER", 1.0))
    alpha = float(getattr(config, "THICKNESS_COMPLIANCE_ALPHA", 0.0))
    scale = float(getattr(config, "DISPLACEMENT_COMPLIANCE_SCALE", 1.0))
    h_ref = float(getattr(config, "H", 1.0))
    u = scale * v / (e_scale ** e_pow) * (h_ref / np.clip(t_scale, 1e-8, None)) ** alpha
    return u


def _run_fea(e1, e2, e3, t1, t2, t3):
    thickness = float(t1) + float(t2) + float(t3)
    cfg = {
        "geometry": {"Lx": config.Lx, "Ly": config.Ly, "H": thickness, "ne_x": 16, "ne_y": 16, "ne_z": 8},
        "material": {"E_layers": [float(e1), float(e2), float(e3)], "nu": 0.3},
        "t_layers": [float(t1), float(t2), float(t3)],
        "load_patch": {
            "pressure": float(config.p0),
            "x_start": config.LOAD_PATCH_X[0] / config.Lx,
            "x_end": config.LOAD_PATCH_X[1] / config.Lx,
            "y_start": config.LOAD_PATCH_Y[0] / config.Ly,
            "y_end": config.LOAD_PATCH_Y[1] / config.Ly,
        },
        "boundary_conditions": {"bottom": "fixed"},
    }
    x_nodes, y_nodes, z_nodes, u_fea = fem_solver.solve_three_layer_fem(cfg)
    return x_nodes, y_nodes, z_nodes, u_fea


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pinn = _load_pinn(device)

    # All benchmark cases
    e_values = [1.0, 10.0]
    t_values = [0.02, 0.10]
    
    cases = []
    for t1 in t_values:
        for t2 in t_values:
            for t3 in t_values:
                for e1 in e_values:
                    for e2 in e_values:
                        for e3 in e_values:
                            cases.append((e1, e2, e3, t1, t2, t3))

    print(f"Evaluating {len(cases)} cases...")
    
    all_mae_before = []
    all_mae_after = []
    
    for e1, e2, e3, t1, t2, t3 in cases:
        x_nodes, y_nodes, z_nodes, u_fea = _run_fea(e1, e2, e3, t1, t2, t3)
        thickness = t1 + t2 + t3
        
        x_grid, y_grid = np.meshgrid(x_nodes, y_nodes, indexing="ij")
        x_flat = x_grid.ravel()
        y_flat = y_grid.ravel()
        z_top = np.full_like(x_flat, thickness)
        
        u_pinn = _predict_pinn(pinn, device, x_flat, y_flat, z_top, e1, e2, e3, t1, t2, t3)
        u_fea_top = u_fea[:, :, -1, :].reshape(-1, 3)
        
        # Find best global multiplier for this case
        ratios = u_fea_top[:, 2] / np.clip(u_pinn[:, 2], 1e-8, None)
        median_ratio = np.median(ratios)
        
        u_calibrated = u_pinn[:, 2] * median_ratio
        
        mae_before = np.mean(np.abs(u_pinn[:, 2] - u_fea_top[:, 2]) / (np.abs(u_fea_top[:, 2]) + 1e-8)) * 100
        mae_after = np.mean(np.abs(u_calibrated - u_fea_top[:, 2]) / (np.abs(u_fea_top[:, 2]) + 1e-8)) * 100
        
        all_mae_before.append(mae_before)
        all_mae_after.append(mae_after)
        
        print(f"E=[{e1:g},{e2:g},{e3:g}] t=[{t1:g},{t2:g},{t3:g}]: MAE {mae_before:.2f}% -> {mae_after:.2f}% (ratio={median_ratio:.4f})")
    
    print(f"\nMean MAE before: {np.mean(all_mae_before):.2f}%")
    print(f"Mean MAE after: {np.mean(all_mae_after):.2f}%")
    print(f"Worst MAE before: {np.max(all_mae_before):.2f}%")
    print(f"Worst MAE after: {np.max(all_mae_after):.2f}%")


if __name__ == "__main__":
    main()
