import os
import sys
import numpy as np
import torch
import time

# Add paths for dependencies
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FEA_SOLVER_DIR = os.path.join(os.path.dirname(ROOT_DIR), "fea-workflow", "solver")
sys.path.append(ROOT_DIR)
sys.path.append(FEA_SOLVER_DIR)

import pinn_config as config
import model
import fem_solver_complex
import matplotlib.pyplot as plt

from scipy.interpolate import RBFInterpolator

def run_12d_performance_benchmark():
    print("=== Time-Equivalency Benchmark: PINN 12D vs. FEM Surrogate (RBF) ===\n")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pinn = model.MultiLayerPINN().to(device)
    ckpt = "pinn_model.pth"
    if os.path.exists(ckpt):
        pinn.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        print(f"Loaded PINN from {ckpt}")
    else:
        print("Error: pinn_model.pth not found.")
        return
    pinn.eval()

    # 1. Train RBF Surrogate (FEM Baseline)
    print("\n[1/3] Training RBF Surrogate (Time-Equivalent Budget: ~60 FEA samples)...")
    num_train = 60
    e_min, e_max = config.E_RANGE
    t_min, t_max = config.THICKNESS_RANGE
    
    train_params = []
    train_labels = []
    
    for i in range(num_train):
        e = [np.random.uniform(e_min, e_max) for _ in range(3)]
        t = [np.random.uniform(t_min, t_max) for _ in range(3)]
        r, mu, v0 = np.random.uniform(0.1, 0.9), np.random.uniform(0, 0.6), np.random.uniform(0.2, 2.0)
        p = e + t + [r, mu, v0]
        
        # Run FEA
        fea_cfg = {
            'geometry': {'Lx': config.Lx, 'Ly': config.Ly, 'H': sum(t)},
            'mesh': {'ne_x': 20, 'ne_y': 20, 'ne_z': 10},
            'material': [{'E': ei, 'nu': 0.3} for ei in e],
            'load_patch': {'x_start': 0.333, 'x_end': 0.667, 'y_start': 0.333, 'y_end': 0.667, 'pressure': 1.0}
        }
        material_grid = np.zeros((20, 20, 10), dtype=int)
        dz = sum(t)/10
        for k in range(10):
            z_c = (k+0.5)*dz
            if z_c < t[0]: material_grid[:,:,k] = 0
            elif z_c < t[0]+t[1]: material_grid[:,:,k] = 1
            else: material_grid[:,:,k] = 2
            
        _, _, _, u_grid = fem_solver_complex.solve_fem_complex(fea_cfg, material_grid)
        peak = np.abs(np.min(u_grid[:, :, -1, 2])) # Peak displacement
        
        train_params.append(p)
        train_labels.append(peak)
        print(f"  Sample {i+1}: Peak={peak:.4f}")

    rbf = RBFInterpolator(np.array(train_params), np.array(train_labels), kernel='thin_plate_spline')

    # 2. Run Test Trials
    print("\n[2/3] Running 10 Test Trials (PINN vs RBF)...")
    num_test = 10
    results = []
    
    for i in range(num_test):
        e = [np.random.uniform(e_min, e_max) for _ in range(3)]
        t = [np.random.uniform(t_min, t_max) for _ in range(3)]
        r, mu, v0 = np.random.uniform(0.1, 0.9), np.random.uniform(0, 0.6), np.random.uniform(0.2, 2.0)
        p = e + t + [r, mu, v0]
        
        # Ground Truth FEA
        fea_cfg = {
            'geometry': {'Lx': config.Lx, 'Ly': config.Ly, 'H': sum(t)},
            'mesh': {'ne_x': 30, 'ne_y': 30, 'ne_z': 15},
            'material': [{'E': ei, 'nu': 0.3} for ei in e],
            'load_patch': {'x_start': 0.333, 'x_end': 0.667, 'y_start': 0.333, 'y_end': 0.667, 'pressure': 1.0}
        }
        material_grid = np.zeros((30, 30, 15), dtype=int)
        dz = sum(t)/15
        for k in range(15):
            z_c = (k+0.5)*dz
            if z_c < t[0]: material_grid[:,:,k] = 0
            elif z_c < t[0]+t[1]: material_grid[:,:,k] = 1
            else: material_grid[:,:,k] = 2
        _, _, _, u_grid = fem_solver_complex.solve_fem_complex(fea_cfg, material_grid)
        gt_peak = np.abs(np.min(u_grid[:, :, -1, 2]))
        
        # PINN Predict
        # We query the center top node
        test_pt = [0.5, 0.5, sum(t)] + p
        test_tensor = torch.tensor([test_pt], dtype=torch.float32).to(device)
        u_pinn = pinn(test_tensor).cpu().detach().numpy()[0, 2]
        pinn_peak = np.abs(u_pinn)
        
        # RBF Predict
        rbf_peak = rbf(np.array([p]))[0]
        
        err_pinn = abs(pinn_peak - gt_peak) / gt_peak * 100
        err_rbf = abs(rbf_peak - gt_peak) / gt_peak * 100
        
        print(f"  Trial {i+1}: GT={gt_peak:.4f} | PINN={pinn_peak:.4f} ({err_pinn:.1f}%) | RBF={rbf_peak:.4f} ({err_rbf:.1f}%)")
        results.append({'pinn': err_pinn, 'rbf': err_rbf, 'id': f"Trial {i+1}"})

    # 3. Visualization
    print("\n[3/3] Generating Tornado Comparison Graph...")
    ids = [r['id'] for r in results]
    pinn_errs = [r['pinn'] for r in results]
    rbf_errs = [r['rbf'] for r in results]
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 14
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y = np.arange(len(ids))
    width = 0.4
    
    bar1 = ax.barh(y - width/2, pinn_errs, width, label='PINN Surrogate Error %', color='#2ecc71', alpha=0.9)
    bar2 = ax.barh(y + width/2, rbf_errs, width, label='FEM Surrogate (RBF) Error %', color='#3498db', alpha=0.9)
    
    ax.set_yticks(y)
    ax.set_yticklabels(ids)
    ax.invert_yaxis()
    ax.set_xlabel('Relative Error (%)', fontweight='bold')
    ax.set_title(f'Time-Equivalence Benchmark (Budget: ~2-3 mins)\nPINN vs RBF Surrogate (Trained on {num_train} FEA Samples)', fontweight='bold', pad=20)
    ax.legend(loc='lower right')
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Annotate values
    for i, bar in enumerate(bar1):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f'{pinn_errs[i]:.1f}%', va='center', fontweight='bold', color='#27ae60')
    for i, bar in enumerate(bar2):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f'{rbf_errs[i]:.1f}%', va='center', fontweight='bold', color='#2980b9')

    plt.tight_layout()
    outfile = os.path.join(ROOT_DIR, "visualization", "pinn_vs_rbf_12d.png")
    plt.savefig(outfile, dpi=150)
    print(f"Graph saved: {outfile}")

if __name__ == "__main__":
    run_12d_performance_benchmark()
