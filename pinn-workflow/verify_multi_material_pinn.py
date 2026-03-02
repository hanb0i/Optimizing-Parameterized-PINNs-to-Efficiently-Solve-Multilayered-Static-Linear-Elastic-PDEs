import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# Path setup
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FEA_SOLVER_DIR = os.path.join(os.path.dirname(ROOT_DIR), "fea-workflow", "solver")

sys.path.append(ROOT_DIR)
sys.path.append(FEA_SOLVER_DIR)

import pinn_config as config
import model
import fem_solver

def load_pinn(model_path, device):
    pinn = model.MultiLayerPINN().to(device)
    if os.path.exists(model_path):
        pinn.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Loaded PINN from {model_path}")
    else:
        raise FileNotFoundError(f"Model {model_path} not found")
    pinn.eval()
    return pinn

def run_pinn_prediction(pinn, x_coords, y_coords, z_coords, e_vals, t_vals, r, mu, v0, device):
    nx, ny, nz = len(x_coords), len(y_coords), len(z_coords)
    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    
    pts = np.zeros((X.size, 12))
    pts[:, 0] = X.ravel()
    pts[:, 1] = Y.ravel()
    pts[:, 2] = Z.ravel()
    pts[:, 3] = e_vals[0]
    pts[:, 4] = e_vals[1]
    pts[:, 5] = e_vals[2]
    pts[:, 6] = t_vals[0] # t1 (Top?) - wait, model uses [e1,e2,e3, t1,t2,t3]
    pts[:, 7] = t_vals[1]
    pts[:, 8] = t_vals[2]
    pts[:, 9] = r
    pts[:, 10] = mu
    pts[:, 11] = v0
    
    with torch.no_grad():
        u_pinn = pinn(torch.tensor(pts, dtype=torch.float32).to(device)).cpu().numpy()
        
    return u_pinn.reshape(nx, ny, nz, 3)

def run_comparison(case_name, e_vals, t_vals, pinn, device, out_dir):
    print(f"\n--- Verifying Case: {case_name} ---")
    E1, E2, E3 = e_vals
    T1, T2, T3 = t_vals
    H_total = sum(t_vals)
    
    # 1. Run FEA
    fea_cfg = {
        'geometry': {
            'Lx': config.Lx, 'Ly': config.Ly, 'H': H_total,
            'layer_thicknesses': [T3, T2, T1] # FEA stacks Bottom to Top
        },
        'material': [
            {'E': E3, 'nu': 0.3},
            {'E': E2, 'nu': 0.3},
            {'E': E1, 'nu': 0.3}
        ],
        'load_patch': {
            'pressure': config.p0,
            'x_start': config.LOAD_PATCH_X[0]/config.Lx,
            'x_end': config.LOAD_PATCH_X[1]/config.Lx,
            'y_start': config.LOAD_PATCH_Y[0]/config.Ly,
            'y_end': config.LOAD_PATCH_Y[1]/config.Ly,
        },
        'use_soft_mask': True,
        'mesh': {'ne_x': 20, 'ne_y': 20, 'ne_z': 30}
    }
    
    x_fea, y_fea, z_fea, u_fea = fem_solver.solve_fem(fea_cfg)
    
    # 2. Run PINN
    x_pinn = np.linspace(0, config.Lx, 41)
    y_pinn = np.linspace(0, config.Ly, 41)
    z_pinn = np.linspace(0, H_total, 41)
    
    u_pinn = run_pinn_prediction(pinn, x_pinn, y_pinn, z_pinn, e_vals, t_vals, 
                                 config.RESTITUTION_REF, config.FRICTION_REF, config.IMPACT_VELOCITY_REF, device)
    
    # 3. Visualization: Top Surface Uz
    plt.figure(figsize=(15, 6))
    
    # FEA Top (z = H_total)
    uz_fea_top = u_fea[:, :, -1, 2].T # Transpose for plotting
    X_fea, Y_fea = np.meshgrid(x_fea, y_fea)
    
    plt.subplot(1, 3, 1)
    plt.contourf(X_fea, Y_fea, uz_fea_top, 20, cmap='jet')
    plt.colorbar(); plt.title(f"FEA Top Uz\nPeak: {np.min(uz_fea_top):.4f}")
    
    # PINN Top (z = H_total)
    uz_pinn_top = u_pinn[:, :, -1, 2].T
    X_pinn, Y_pinn = np.meshgrid(x_pinn, y_pinn)
    
    plt.subplot(1, 3, 2)
    plt.contourf(X_pinn, Y_pinn, uz_pinn_top, 20, cmap='jet')
    plt.colorbar(); plt.title(f"PINN Top Uz\nPeak: {np.min(uz_pinn_top):.4f}")
    
    # Difference (Interpolated)
    interp = RegularGridInterpolator((x_fea, y_fea), u_fea[:, :, -1, 2], bounds_error=False, fill_value=0.0)
    uz_fea_interp = interp(np.stack([X_pinn.flatten(), Y_pinn.flatten()], axis=1)).reshape(X_pinn.shape)
    diff = np.abs(uz_pinn_top - uz_fea_interp.T)
    
    plt.subplot(1, 3, 3)
    plt.contourf(X_pinn, Y_pinn, diff, 20, cmap='magma')
    plt.colorbar(); plt.title(f"Abs Diff\nMAE: {np.mean(diff):.4f}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{case_name}_top.png"))
    plt.close()
    
    # 4. Visualization: X-Z Cross section at y=0.5
    plt.figure(figsize=(15, 6))
    
    # FEA XZ
    y_idx_fea = len(y_fea) // 2
    uz_fea_xz = u_fea[:, y_idx_fea, :, 2].T
    X_fea_xz, Z_fea_xz = np.meshgrid(x_fea, z_fea)
    
    plt.subplot(1, 3, 1)
    plt.contourf(X_fea_xz, Z_fea_xz, uz_fea_xz, 20, cmap='jet')
    plt.colorbar(); plt.title("FEA X-Z Section")
    
    # PINN XZ
    y_idx_pinn = len(y_pinn) // 2
    uz_pinn_xz = u_pinn[:, y_idx_pinn, :, 2].T
    X_pinn_xz, Z_pinn_xz = np.meshgrid(x_pinn, z_pinn)
    
    plt.subplot(1, 3, 2)
    plt.contourf(X_pinn_xz, Z_pinn_xz, uz_pinn_xz, 20, cmap='jet')
    plt.colorbar(); plt.title("PINN X-Z Section")
    
    # Difference
    interp_xz = RegularGridInterpolator((x_fea, z_fea), u_fea[:, y_idx_fea, :, 2], bounds_error=False, fill_value=0.0)
    uz_fea_xz_interp = interp_xz(np.stack([X_pinn_xz.flatten(), Z_pinn_xz.flatten()], axis=1)).reshape(X_pinn_xz.shape)
    diff_xz = np.abs(uz_pinn_xz - uz_fea_xz_interp.T)
    
    plt.subplot(1, 3, 3)
    plt.contourf(X_pinn_xz, Z_pinn_xz, diff_xz, 20, cmap='magma')
    plt.colorbar(); plt.title("Abs Diff XZ")
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{case_name}_xz.png"))
    plt.close()
    
    peak_fea = np.min(uz_fea_top)
    peak_pinn = np.min(uz_pinn_top)
    peak_err = abs(peak_pinn - peak_fea) / abs(peak_fea) if abs(peak_fea) > 1e-9 else 0.0
    print(f"  Peaks: FEA={peak_fea:.6f}, PINN={peak_pinn:.6f}")
    print(f"  Field MAE: {np.mean(diff):.6f}")
    print(f"  Peak Error: {peak_err*100:.2f}%")
    return peak_err

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_path = os.path.join(os.path.dirname(ROOT_DIR), "pinn_model.pth")
    viz_dir = os.path.join(ROOT_DIR, "visualization", "multi_material")
    os.makedirs(viz_dir, exist_ok=True)
    
    pinn = load_pinn(model_path, device)
    
    test_cases = [
        {
            "name": "Sandwich_Stiff_Soft",
            "e_vals": [10.0, 1.0, 10.0],
            "t_vals": [0.02, 0.06, 0.02]
        },
        {
            "name": "Graded_Stiff_to_Soft",
            "e_vals": [10.0, 5.0, 1.0],
            "t_vals": [0.03, 0.04, 0.03]
        },
        {
            "name": "Thick_Face_Soft_Core",
            "e_vals": [8.0, 1.5, 8.0],
            "t_vals": [0.04, 0.02, 0.04]
        }
    ]
    
    total_peak_err = 0
    for case in test_cases:
        err = run_comparison(case["name"], case["e_vals"], case["t_vals"], pinn, device, viz_dir)
        total_peak_err += err
        
    avg_err = (total_peak_err / len(test_cases)) * 100
    print(f"\nAverage Peak Error across all cases: {avg_err:.2f}%")
