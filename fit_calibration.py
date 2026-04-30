"""Fit post-hoc calibration to reduce PINN vs FEM error."""

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


def _calibration_features(pts):
    x = pts[:, 0:1]
    y = pts[:, 1:2]
    z = pts[:, 2:3]
    e1 = pts[:, 3:4]
    t1 = pts[:, 4:5]
    e2 = pts[:, 5:6]
    t2 = pts[:, 6:7]
    e3 = pts[:, 7:8]
    t3 = pts[:, 8:9]
    t_total = np.clip(t1 + t2 + t3, 1e-8, None)
    e_mean = np.clip((e1 + e2 + e3) / 3.0, 1e-8, None)
    z_hat = z / t_total
    e_ref = np.sqrt(float(config.E_RANGE[0]) * float(config.E_RANGE[1]))
    h_ref = float(getattr(config, "H", 0.1))
    load_x = ((x >= config.LOAD_PATCH_X[0]) & (x <= config.LOAD_PATCH_X[1])).astype(float)
    load_y = ((y >= config.LOAD_PATCH_Y[0]) & (y <= config.LOAD_PATCH_Y[1])).astype(float)
    load_patch = load_x * load_y
    xc = x - 0.5 * float(config.Lx)
    yc = y - 0.5 * float(config.Ly)
    feats = np.concatenate([
        np.ones_like(x),
        np.log(e_mean / e_ref),
        np.log(np.clip(e1, 1e-8, None) / e_ref),
        np.log(np.clip(e2, 1e-8, None) / e_ref),
        np.log(np.clip(e3, 1e-8, None) / e_ref),
        np.log(h_ref / t_total),
        t1 / t_total,
        t2 / t_total,
        t3 / t_total,
        z_hat,
        z_hat**2,
        load_patch,
        xc,
        yc,
        xc**2,
        yc**2,
        xc * yc,
        load_patch * xc,
        load_patch * yc,
        load_patch * xc**2,
        load_patch * yc**2,
    ], axis=1)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pinn = _load_pinn(device)

    # Training cases for calibration
    cases = [
        (1.0, 1.0, 1.0, 0.02, 0.10, 0.02),
        (1.0, 1.0, 10.0, 0.02, 0.10, 0.02),
        (1.0, 10.0, 1.0, 0.02, 0.10, 0.02),
        (1.0, 10.0, 10.0, 0.02, 0.10, 0.02),
        (10.0, 1.0, 1.0, 0.02, 0.10, 0.02),
        (10.0, 1.0, 10.0, 0.02, 0.10, 0.02),
        (10.0, 10.0, 1.0, 0.02, 0.10, 0.02),
        (10.0, 10.0, 10.0, 0.02, 0.10, 0.02),
        (1.0, 10.0, 1.0, 0.02, 0.02, 0.02),
        (10.0, 1.0, 10.0, 0.02, 0.02, 0.02),
    ]

    all_pts = []
    all_u_pinn = []
    all_u_fea = []

    print("Collecting data for calibration...")
    for e1, e2, e3, t1, t2, t3 in cases:
        x_nodes, y_nodes, z_nodes, u_fea = _run_fea(e1, e2, e3, t1, t2, t3)
        thickness = t1 + t2 + t3
        
        # Sample points: top surface + some interior
        x_grid, y_grid = np.meshgrid(x_nodes, y_nodes, indexing="ij")
        x_flat = x_grid.ravel()
        y_flat = y_grid.ravel()
        
        # Top surface
        z_top = np.full_like(x_flat, thickness)
        u_pinn_top = _predict_pinn(pinn, device, x_flat, y_flat, z_top, e1, e2, e3, t1, t2, t3)
        u_fea_top = u_fea[:, :, -1, :].reshape(-1, 3)
        pts_top = np.stack([x_flat, y_flat, z_top, np.full_like(x_flat, e1), np.full_like(x_flat, t1),
                            np.full_like(x_flat, e2), np.full_like(x_flat, t2),
                            np.full_like(x_flat, e3), np.full_like(x_flat, t3)], axis=1)
        
        all_pts.append(pts_top)
        all_u_pinn.append(u_pinn_top)
        all_u_fea.append(u_fea_top)

    pts = np.concatenate(all_pts, axis=0)
    u_pinn = np.concatenate(all_u_pinn, axis=0)
    u_fea = np.concatenate(all_u_fea, axis=0)

    # Fit calibration per component
    feats = _calibration_features(pts)
    
    # Use only z-component (displacement in z)
    ratio = np.clip(u_fea[:, 2:3] / np.clip(u_pinn[:, 2:3], 1e-8, None), 0.1, 10.0)
    log_ratio = np.log(ratio)
    
    # Linear regression: log(ratio) = feats @ coeffs
    coeffs, residuals, rank, s = np.linalg.lstsq(feats, log_ratio.ravel(), rcond=None)
    
    print(f"Calibration fitted. Residuals: {residuals}")
    print(f"Coeffs: {coeffs.tolist()}")
    
    # Evaluate
    log_pred = feats @ coeffs
    multiplier = np.exp(np.clip(log_pred, -1.5, 1.5))
    u_calibrated = u_pinn[:, 2] * multiplier
    
    mae_before = np.mean(np.abs(u_pinn[:, 2] - u_fea[:, 2]) / (np.abs(u_fea[:, 2]) + 1e-8)) * 100
    mae_after = np.mean(np.abs(u_calibrated - u_fea[:, 2]) / (np.abs(u_fea[:, 2]) + 1e-8)) * 100
    
    print(f"MAE before calibration: {mae_before:.2f}%")
    print(f"MAE after calibration: {mae_after:.2f}%")
    
    # Save calibration
    cal_data = {
        "feature_coefficients": coeffs.tolist(),
        "log_multiplier_clip": 1.5,
        "mae_before": mae_before,
        "mae_after": mae_after,
    }
    cal_path = os.path.join(PINN_WORKFLOW_DIR, "calibration.json")
    with open(cal_path, "w") as f:
        json.dump(cal_data, f, indent=2)
    print(f"Saved calibration to {cal_path}")


if __name__ == "__main__":
    main()
