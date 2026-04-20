#!/usr/bin/env python3
"""Generate cross-section error heatmaps for the best cases from 50-run benchmark."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PREVIEW_DIR = REPO_ROOT / "graphs" / "figures" / "ablation_preview"
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


def _three_layer_best():
    three_layer_dir = REPO_ROOT / "three-layer-workflow"
    fea_dir = REPO_ROOT / "fea-workflow" / "solver"
    sys.path.insert(0, str(three_layer_dir))
    sys.path.insert(0, str(fea_dir))

    import pinn_config as config
    import model
    import fem_solver

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pinn = model.MultiLayerPINN().to(device)
    ckpt = three_layer_dir / "pinn_model.pth"
    sd = torch.load(str(ckpt), map_location=device, weights_only=True)
    sd = model.adapt_legacy_state_dict(sd, pinn.state_dict())
    pinn.load_state_dict(sd, strict=False)
    pinn.eval()

    # Best case from 50-run: random_interior_007
    e1, e2, e3 = 5.3672985, 9.9661356, 5.089864
    t1, t2, t3 = 0.020662307, 0.052478186, 0.097581473
    thickness = t1 + t2 + t3

    cfg = {
        "geometry": {"Lx": config.Lx, "Ly": config.Ly, "H": thickness, "ne_x": 10, "ne_y": 10, "ne_z": 4},
        "material": {"E_layers": [e1, e2, e3], "t_layers": [t1, t2, t3], "nu": config.nu_vals[0]},
        "load_patch": {
            "pressure": config.p0,
            "x_start": config.LOAD_PATCH_X[0] / config.Lx,
            "x_end": config.LOAD_PATCH_X[1] / config.Lx,
            "y_start": config.LOAD_PATCH_Y[0] / config.Ly,
            "y_end": config.LOAD_PATCH_Y[1] / config.Ly,
        },
    }
    x_nodes, y_nodes, z_nodes, u_fea = fem_solver.solve_three_layer_fem(cfg)
    x_nodes = np.array(x_nodes)
    y_nodes = np.array(y_nodes)
    z_nodes = np.array(z_nodes)

    mid_y_idx = len(y_nodes) // 2
    x_cross, z_cross = np.meshgrid(x_nodes, z_nodes, indexing="ij")
    y_cross = np.full(x_cross.size, y_nodes[mid_y_idx])

    r_ref = float(getattr(config, "RESTITUTION_REF", 0.5))
    mu_ref = float(getattr(config, "FRICTION_REF", 0.3))
    v0_ref = float(getattr(config, "IMPACT_VELOCITY_REF", 1.0))
    pts = np.stack([
        x_cross.ravel(), y_cross, z_cross.ravel(),
        np.full(x_cross.size, e1), np.full(x_cross.size, t1),
        np.full(x_cross.size, e2), np.full(x_cross.size, t2),
        np.full(x_cross.size, e3), np.full(x_cross.size, t3),
        np.full(x_cross.size, r_ref), np.full(x_cross.size, mu_ref), np.full(x_cross.size, v0_ref),
    ], axis=1)

    e_pow = float(getattr(config, "E_COMPLIANCE_POWER", 1.0))
    alpha = float(getattr(config, "THICKNESS_COMPLIANCE_ALPHA", 0.0))
    scale = float(getattr(config, "DISPLACEMENT_COMPLIANCE_SCALE", 1.0))
    h_ref = float(getattr(config, "H", 1.0))

    with torch.no_grad():
        v = pinn(torch.tensor(pts, dtype=torch.float32, device=device)).cpu().numpy()

    e_scale = (pts[:, 3:4] + pts[:, 5:6] + pts[:, 7:8]) / 3.0
    t_scale = pts[:, 4:5] + pts[:, 6:7] + pts[:, 8:9]
    u_pinn = scale * v / (e_scale ** e_pow) * (h_ref / np.clip(t_scale, 1e-8, None)) ** alpha
    u_pinn_cross = u_pinn.reshape(len(x_nodes), len(z_nodes), 3)

    u_fea_cross = np.array(u_fea)[:, mid_y_idx, :, 2]
    abs_err_cross = np.abs(u_pinn_cross[:, :, 2] - u_fea_cross)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.contourf(x_cross, z_cross, abs_err_cross, levels=50, cmap="magma")
    plt.colorbar(im, ax=ax, label="|u_z PINN − u_z FEA|")
    ax.axhline(float(t1), color="white", linestyle="--", linewidth=1.5)
    ax.axhline(float(t1) + float(t2), color="white", linestyle="--", linewidth=1.5)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(
        f"Three-Layer Best Case Cross-Section Error (y=0.5)\n"
        f"E=[{e1:.2f},{e2:.2f},{e3:.2f}], t=[{t1:.3f},{t2:.3f},{t3:.3f}]\n"
        f"Top MAE = 3.25%"
    )
    fig.tight_layout()
    fig.savefig(PREVIEW_DIR / "three_layer_best_case_cross_section.png", dpi=200)
    plt.close(fig)
    print(f"Saved: {PREVIEW_DIR / 'three_layer_best_case_cross_section.png'}")


def _one_layer_best():
    one_layer_dir = REPO_ROOT / "one-layer-workflow"
    fea_dir = REPO_ROOT / "fea-workflow" / "solver"
    for p in list(sys.path):
        if "three-layer-workflow" in p:
            sys.path.remove(p)
    sys.path.insert(0, str(one_layer_dir))
    sys.path.insert(0, str(fea_dir))

    import importlib
    for mod in ["model", "pinn_config", "fem_solver"]:
        if mod in sys.modules:
            del sys.modules[mod]

    import pinn_config as config
    import model
    import fem_solver

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pinn = model.MultiLayerPINN().to(device)
    ckpt = one_layer_dir / "pinn_model.pth"
    sd = torch.load(str(ckpt), map_location=device, weights_only=True)
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

    # Best case from 50-run: one_layer_random_038
    E_val = 1.6984737
    t_val = 0.11693676

    cfg = {
        "geometry": {"Lx": config.Lx, "Ly": config.Ly, "H": t_val},
        "material": {"E": E_val, "nu": config.nu_vals[0]},
        "load_patch": {
            "pressure": config.p0,
            "x_start": config.LOAD_PATCH_X[0] / config.Lx,
            "x_end": config.LOAD_PATCH_X[1] / config.Lx,
            "y_start": config.LOAD_PATCH_Y[0] / config.Ly,
            "y_end": config.LOAD_PATCH_Y[1] / config.Ly,
        },
    }
    x_nodes, y_nodes, z_nodes, u_fea = fem_solver.solve_fem(cfg)
    x_nodes = np.array(x_nodes)
    y_nodes = np.array(y_nodes)
    z_nodes = np.array(z_nodes)

    mid_y_idx = len(y_nodes) // 2
    x_cross, z_cross = np.meshgrid(x_nodes, z_nodes, indexing="ij")

    r_ref = float(getattr(config, "RESTITUTION_REF", 0.5))
    mu_ref = float(getattr(config, "FRICTION_REF", 0.3))
    v0_ref = float(getattr(config, "IMPACT_VELOCITY_REF", 1.0))

    pts = np.stack([
        x_cross.ravel(),
        np.full(x_cross.size, y_nodes[mid_y_idx]),
        z_cross.ravel(),
        np.full(x_cross.size, E_val),
        np.full(x_cross.size, t_val),
        np.full(x_cross.size, r_ref),
        np.full(x_cross.size, mu_ref),
        np.full(x_cross.size, v0_ref),
    ], axis=1)

    with torch.no_grad():
        v = pinn(torch.tensor(pts, dtype=torch.float32, device=device)).cpu().numpy()

    alpha = float(getattr(config, "THICKNESS_COMPLIANCE_ALPHA", 0.0))
    e_pow = float(getattr(config, "E_COMPLIANCE_POWER", 1.0))
    t_scale = 1.0 if alpha == 0.0 else (float(config.H) / max(1e-8, float(t_val))) ** alpha
    u_pinn = (v / (float(E_val) ** e_pow)) * t_scale
    u_pinn_cross = u_pinn.reshape(len(x_nodes), len(z_nodes), 3)

    u_fea_cross = np.array(u_fea)[:, mid_y_idx, :, 2]
    abs_err_cross = np.abs(u_pinn_cross[:, :, 2] - u_fea_cross)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.contourf(x_cross, z_cross, abs_err_cross, levels=50, cmap="magma")
    plt.colorbar(im, ax=ax, label="|u_z PINN − u_z FEA|")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(
        f"One-Layer Best Case Cross-Section Error (y=0.5)\n"
        f"E={E_val:.2f}, t={t_val:.3f}\n"
        f"Top MAE = 3.07%"
    )
    fig.tight_layout()
    fig.savefig(PREVIEW_DIR / "one_layer_best_case_cross_section.png", dpi=200)
    plt.close(fig)
    print(f"Saved: {PREVIEW_DIR / 'one_layer_best_case_cross_section.png'}")


if __name__ == "__main__":
    _three_layer_best()
    _one_layer_best()
