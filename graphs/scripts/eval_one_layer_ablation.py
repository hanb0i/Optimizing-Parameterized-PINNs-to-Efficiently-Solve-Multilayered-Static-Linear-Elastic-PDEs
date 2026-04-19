#!/usr/bin/env python3
"""One-layer PINN ablation via grid-sweep MAE% with different compliance settings."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

REPO_ROOT = Path(__file__).resolve().parents[2]
ONE_LAYER_DIR = REPO_ROOT / "one-layer-workflow"
FEA_SOLVER_DIR = REPO_ROOT / "fea-workflow" / "solver"
DATA_DIR = REPO_ROOT / "graphs" / "data"

for p in (ONE_LAYER_DIR, FEA_SOLVER_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pinn_config as config
import model
import fem_solver


def _load_pinn(ckpt_path: Path, device: torch.device):
    pinn = model.MultiLayerPINN().to(device)
    if ckpt_path.exists():
        sd = torch.load(str(ckpt_path), map_location=device, weights_only=True)
        target_sd = pinn.state_dict()
        w_key = "layer.net.0.weight"
        if w_key in sd and w_key in target_sd:
            src_w = sd[w_key]
            tgt_w = target_sd[w_key]
            if src_w.shape != tgt_w.shape and src_w.shape[0] == tgt_w.shape[0]:
                if src_w.shape[1] == 8 and tgt_w.shape[1] == 11:
                    adapted = torch.zeros_like(tgt_w)
                    adapted[:, 0:5] = src_w[:, 0:5]
                    adapted[:, 8:11] = src_w[:, 5:8]
                    sd[w_key] = adapted
                elif src_w.shape[1] == 10 and tgt_w.shape[1] == 11:
                    adapted = torch.zeros_like(tgt_w)
                    adapted[:, 0:7] = src_w[:, 0:7]
                    adapted[:, 8:11] = src_w[:, 7:10]
                    sd[w_key] = adapted
        pinn.load_state_dict(sd, strict=False)
    pinn.eval()
    return pinn


def _u_from_v(v, E_val, thickness, e_pow, alpha, scale):
    t_scale = 1.0 if alpha == 0.0 else (float(config.H) / max(1e-8, float(thickness))) ** alpha
    return (scale * v / (float(E_val) ** e_pow)) * t_scale


def _run_fea(E_val, thickness):
    cfg = {
        "geometry": {"Lx": config.Lx, "Ly": config.Ly, "H": thickness},
        "material": {"E": E_val, "nu": config.nu_vals[0]},
        "load_patch": {
            "pressure": config.p0,
            "x_start": config.LOAD_PATCH_X[0] / config.Lx,
            "x_end": config.LOAD_PATCH_X[1] / config.Lx,
            "y_start": config.LOAD_PATCH_Y[0] / config.Ly,
            "y_end": config.LOAD_PATCH_Y[1] / config.Ly,
        },
    }
    return fem_solver.solve_fem(cfg)


def _grid_sweep(pinn, device, e_pow, alpha, scale):
    t_values = [0.05, 0.1, 0.15]
    e_values = [1.0, 5.0, 10.0]

    nx = 51
    ny = 51
    x_range = np.linspace(0, config.Lx, nx)
    y_range = np.linspace(0, config.Ly, ny)
    X, Y = np.meshgrid(x_range, y_range)
    X_flat = X.flatten()
    Y_flat = Y.flatten()

    r_ref = float(getattr(config, "RESTITUTION_REF", 0.5))
    mu_ref = float(getattr(config, "FRICTION_REF", 0.3))
    v0_ref = float(getattr(config, "IMPACT_VELOCITY_REF", 1.0))

    cases = []
    for t_val in t_values:
        x_n, y_n, z_n, u_fea_ref = _run_fea(1.0, t_val)
        x_n = np.array(x_n)
        y_n = np.array(y_n)

        # Top surface FEA reference for E=1
        u_z_fea_top_e1 = np.array(u_fea_ref, dtype=float)[:, :, -1, 2]

        for E_val in e_values:
            # Scale FEA displacement for this E
            u_z_fea_top_raw = u_z_fea_top_e1 / float(E_val)  # (nx_fea, ny_fea)

            # Interpolate FEA onto evaluation grid
            interp = RegularGridInterpolator((x_n, y_n), u_z_fea_top_raw, method="linear", bounds_error=False, fill_value=None)
            u_z_fea_top = interp(np.stack([X_flat, Y_flat], axis=1)).reshape(ny, nx)

            # PINN prediction
            Z_flat = np.ones_like(X_flat) * t_val
            T_flat = np.ones_like(X_flat) * t_val
            E_flat = np.ones_like(X_flat) * E_val
            R_flat = np.ones_like(X_flat) * r_ref
            MU_flat = np.ones_like(X_flat) * mu_ref
            V0_flat = np.ones_like(X_flat) * v0_ref
            pts = np.stack([X_flat, Y_flat, Z_flat, E_flat, T_flat, R_flat, MU_flat, V0_flat], axis=1)

            with torch.no_grad():
                v = pinn(torch.tensor(pts, dtype=torch.float32).to(device)).cpu().numpy()
            u_pinn = _u_from_v(v, E_val, t_val, e_pow, alpha, scale)
            u_z_pinn_top = u_pinn[:, 2].reshape(ny, nx)

            # MAE%
            mae = float(np.mean(np.abs(u_z_pinn_top - u_z_fea_top)))
            denom = float(np.max(np.abs(u_z_fea_top)))
            mae_pct = 100.0 * mae / denom if denom > 0 else 0.0
            cases.append(mae_pct)
            print(f"  E={E_val:.1f}, t={t_val:.2f}: MAE={mae_pct:.2f}%")

    cases_arr = np.array(cases, dtype=float)
    return float(np.mean(cases_arr)), float(np.max(cases_arr))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    ckpt_path = ONE_LAYER_DIR / "pinn_model.pth"

    pinn = _load_pinn(ckpt_path, device)
    print(f"Loaded model: {ckpt_path}")
    print(f"Device: {device}")

    # Ablation variants
    variants = [
        ("Pure-physics baseline", 0.0, 0.0, 1.0),
        ("+ Compliance-aware scaling", 0.973, 1.234, 1.0),
        ("+ Tuned compliance-aware scaling", 0.9743937316337181, 0.5784595573516915, 0.7360429019668411),
    ]

    rows = []
    for name, e_pow, alpha, scale in variants:
        print(f"\n=== {name} (e_pow={e_pow:.4f}, alpha={alpha:.4f}, scale={scale:.4f}) ===")
        mean_mae, worst_mae = _grid_sweep(pinn, device, e_pow, alpha, scale)
        print(f"  Mean MAE: {mean_mae:.2f}%")
        print(f"  Worst MAE: {worst_mae:.2f}%")
        rows.append({"variant": name, "mean_mae": f"{mean_mae:.4f}", "worst_mae": f"{worst_mae:.4f}"})

    out_csv = DATA_DIR / "one_layer_ablation_grid_sweep.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "mean_mae", "worst_mae"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote results to {out_csv}")


if __name__ == "__main__":
    main()
