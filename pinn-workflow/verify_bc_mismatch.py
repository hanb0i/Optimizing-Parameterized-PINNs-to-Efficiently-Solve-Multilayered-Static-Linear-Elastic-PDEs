
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Add paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(ROOT_DIR)
FEA_DIR = os.path.join(REPO_ROOT, "fea-workflow")
FEA_SOLVER_DIR = os.path.join(FEA_DIR, "solver")

sys.path.append(ROOT_DIR)
sys.path.append(FEA_DIR)
sys.path.append(FEA_SOLVER_DIR)

from surrogate_api import ParametricSurrogate

# --- 1. Custom FEA Solver with SWITCHABLE BCs ---
def solve_fem_custom_bc(cfg, bc_mode='plate_bending'):
    """
    bc_mode: 'plate_bending' (Sides fixed, Bottom free) OR 'block_compression' (Bottom fixed)
    """
    Lx, Ly, H = cfg['geometry']['Lx'], cfg['geometry']['Ly'], cfg['geometry']['H']
    ne_x = cfg.get('mesh', {}).get('ne_x', 20)
    ne_y = cfg.get('mesh', {}).get('ne_y', 20)
    ne_z = cfg.get('mesh', {}).get('ne_z', 10)
    
    dx, dy, dz = Lx/ne_x, Ly/ne_y, H/ne_z
    nx, ny, nz = ne_x+1, ne_y+1, ne_z+1
    n_dof = nx * ny * nz * 3
    
    # Material (Single Layer)
    E_val = cfg['material']['E']
    nu_val = cfg['material']['nu']
    lam = (E_val * nu_val) / ((1 + nu_val) * (1 - 2 * nu_val))
    mu = E_val / (2 * (1 + nu_val))
    
    # Stiffness Assembly (Simplified for single material)
    # ... (Reuse standard logic or import? Better to reimplement quickly for transparency)
    # Actually, let's reuse fem_solver logic but hijack the BC part requires modifying source.
    # Instead, let's just use fem_solver and assume it does Plate Bending, 
    # then write a quick override or copy-paste essential assembly.
    
    # Importing fem_solver internals is messy. Let's just implement the crucial BC difference here.
    # ... Wait, solving full 3D FEM in a script is too long. 
    # Let's import fem_solver and MONKEY PATCH the BC section? No, unreadable.
    # Let's write a simplified solver here.

    # Pre-calc stiffness matrix (Ke)
    gp = [-1/np.sqrt(3), 1/np.sqrt(3)]
    invJ = np.diag([2/dx, 2/dy, 2/dz])
    detJ = dx * dy * dz / 8.0
    
    C_diag = [lam+2*mu, lam+2*mu, lam+2*mu, mu, mu, mu]
    C = np.zeros((6, 6))
    C[0:3, 0:3] = lam
    np.fill_diagonal(C, C_diag)
    
    Ke = np.zeros((24, 24))
    for r in gp:
        for s in gp:
            for t in gp:
                B = np.zeros((6, 24))
                node_signs = [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                              [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]
                for i in range(8):
                    xi, eta, zeta = node_signs[i]
                    dN_dxi = 0.125 * xi * (1 + eta * s) * (1 + zeta * t)
                    dN_deta = 0.125 * eta * (1 + xi * r) * (1 + zeta * t)
                    dN_dzeta = 0.125 * zeta * (1 + xi * r) * (1 + eta * s)
                    d_global = invJ @ np.array([dN_dxi, dN_deta, dN_dzeta])
                    nx_val, ny_val, nz_val = d_global
                    col = 3 * i
                    B[0, col] = nx_val
                    B[1, col+1] = ny_val
                    B[2, col+2] = nz_val
                    B[3, col+1] = nz_val; B[3, col+2] = ny_val
                    B[4, col] = nz_val; B[4, col+2] = nx_val
                    B[5, col] = ny_val; B[5, col+1] = nx_val
                Ke += B.T @ C @ B * detJ

    # Assembly
    el_indices = np.arange(ne_x * ne_y * ne_z)
    ek, ej, ei = np.unravel_index(el_indices, (ne_z, ne_y, ne_x))
    
    n0 = (ei) + (ej)*nx + (ek)*nx*ny
    n1 = (ei+1) + (ej)*nx + (ek)*nx*ny
    n2 = (ei+1) + (ej+1)*nx + (ek)*nx*ny
    n3 = (ei) + (ej+1)*nx + (ek)*nx*ny
    n4 = (ei) + (ej)*nx + (ek+1)*nx*ny
    n5 = (ei+1) + (ej)*nx + (ek+1)*nx*ny
    n6 = (ei+1) + (ej+1)*nx + (ek+1)*nx*ny
    n7 = (ei) + (ej+1)*nx + (ek+1)*nx*ny
    
    conn = np.stack([n0, n1, n2, n3, n4, n5, n6, n7], axis=1)
    
    dof_indices = np.zeros((conn.shape[0], 24), dtype=int)
    for i in range(8):
        dof_indices[:, i*3:i*3+3] = conn[:, i:i+1] * 3 + np.array([0,1,2])
        
    vals = np.tile(Ke.ravel(), conn.shape[0])
    dof_rows = np.repeat(dof_indices, 24, axis=1).ravel()
    dof_cols = np.tile(dof_indices, (1, 24)).ravel()
    
    K = sp.coo_matrix((vals, (dof_rows, dof_cols)), shape=(n_dof, n_dof)).tocsr()
    
    # Apply Load
    F = np.zeros(n_dof)
    p0 = 1.0
    patch_x_min, patch_x_max = 0.333*Lx, 0.667*Lx
    patch_y_min, patch_y_max = 0.333*Ly, 0.667*Ly
    
    # Surface nodes (Top)
    k = nz-1
    x_grid = np.linspace(0, Lx, nx)
    y_grid = np.linspace(0, Ly, ny)
    for j in range(ny):
        for i in range(nx):
            if (x_grid[i] >= patch_x_min and x_grid[i] <= patch_x_max and 
                y_grid[j] >= patch_y_min and y_grid[j] <= patch_y_max):
                n_idx = i + j*nx + k*nx*ny
                F[3*n_idx + 2] -= p0 * dx * dy # Approx point load area
    
    # --- BOUNDARY CONDITIONS ---
    fixed_dofs = []
    
    if bc_mode == 'plate_bending':
        # Fixed Sides (x=0, x=L, y=0, y=L)
        for j in range(ny):
            for k in range(nz):
                n_start = 0 + j*nx + k*nx*ny; n_end = (nx-1) + j*nx + k*nx*ny
                fixed_dofs.extend([3*n_start, 3*n_start+1, 3*n_start+2])
                fixed_dofs.extend([3*n_end, 3*n_end+1, 3*n_end+2])
        for i in range(nx):
            for k in range(nz):
                n_start = i + 0*nx + k*nx*ny; n_end = i + (ny-1)*nx + k*nx*ny
                fixed_dofs.extend([3*n_start, 3*n_start+1, 3*n_start+2])
                fixed_dofs.extend([3*n_end, 3*n_end+1, 3*n_end+2])
                
    elif bc_mode == 'block_compression':
        # Fixed Bottom (z=0)
        k = 0
        for j in range(ny):
            for i in range(nx):
                n_idx = i + j*nx + k*nx*ny
                fixed_dofs.extend([3*n_idx, 3*n_idx+1, 3*n_idx+2])
                
    fixed_dofs = np.unique(fixed_dofs)
    penalty = 1e12
    K = K + sp.coo_matrix((np.ones(len(fixed_dofs))*penalty, (fixed_dofs, fixed_dofs)), shape=(n_dof, n_dof)).tocsr()
    
    # Solve
    u = spla.spsolve(K, F)
    
    # Extract Center Displacement
    center_idx = i + j*nx + (nz-1)*nx*ny # Approximate top center
    # Better: find i,j of center
    ic, jc = nx//2, ny//2
    idx_top = ic + jc*nx + (nz-1)*nx*ny
    idx_bot = ic + jc*nx + 0*nx*ny
    
    u_top = u[3*idx_top+2]
    u_bot = u[3*idx_bot+2]
    
    return u_top, u_bot

# --- 2. Main Verification Logic ---
def run_bc_check():
    print("=== Model Physics Entailment Check: BC Mismatch Analysis ===\n")
    
    # Test Parameters
    E_test = 5.0
    h_test = 0.1
    p_params = {'geometry': {'Lx': 1.0, 'Ly': 1.0, 'H': h_test}, 'material': {'E': E_test, 'nu': 0.3}}
    
    # 1. Run FEA (Plate Bending - Current 3-Layer Logic)
    u_top_bend, u_bot_bend = solve_fem_custom_bc(p_params, bc_mode='plate_bending')
    print(f"FEA (Mode: Plate Bending):")
    print(f"  Top Disp: {u_top_bend:.5f}")
    print(f"  Bot Disp: {u_bot_bend:.5f}")
    print(f"  Net Comp: {abs(u_top_bend - u_bot_bend):.5f}")
    
    # 2. Run FEA (Block Compression - Suspected PINN Training Logic)
    u_top_comp, u_bot_comp = solve_fem_custom_bc(p_params, bc_mode='block_compression')
    print(f"\nFEA (Mode: Block Compression):")
    print(f"  Top Disp: {u_top_comp:.5f}")
    print(f"  Bot Disp: {u_bot_comp:.5f}")
    print(f"  Net Comp: {abs(u_top_comp - u_bot_comp):.5f}")
    
    # 3. Run PINN (The Actual Physics Model)
    import model
    import torch
    import pinn_config as config
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pinn = model.MultiLayerPINN().to(device)
    # Load the newly trained PINN
    ckpt_path = os.path.join(ROOT_DIR, "pinn_model.pth")
    print(f"\nLoading PINN from {ckpt_path}...")
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    pinn.load_state_dict(sd)
    pinn.eval()
    
    # Prepare input tensor for center point
    # [x, y, z, E, t, r, mu, v0]
    # Center top point
    x_c = 0.5
    y_c = 0.5
    z_c = h_test
    
    # In pinn_config.py: E_RANGE=[1,10], etc.
    # The PINN expects normalized inputs?
    # No, physics.py uses x_int directly. 
    # But usually data.py normalizes inputs?
    # Let's check data.py or check how inputs are fed.
    # physics.py lines 106: E_local = x_int[:, 3:4].
    # It seems to expect raw values if E_local is used directly for compliance.
    
    # Construct input
    # Param order: x,y,z, E, t, r, mu, v0 (11 dims? No, 8?)
    # train.py line 25: adapts to 11 inputs? 
    # Let's check model.py input dim.
    # If 11 dims: x,y,z, E, t, r, mu, v0, + 3 Fourier?
    # train.py 24: "8->11 inputs".
    # pinn_config.py: FOURIER_DIM=0.
    
    # Let's assume standard 8 inputs + 0 fourier = 8?
    # Wait, Reference PINN has 5 params + 3 coords = 8.
    
    # Coordinates are [0, Lx], [0, Ly], [0, H].
    # Params are raw values.
    
    pts = np.array([[x_c, y_c, z_c, E_test, h_test, 0.5, 0.3, 1.0]], dtype=np.float32)
    pts_tensor = torch.tensor(pts).to(device)
    
    with torch.no_grad():
        v_pred = pinn(pts_tensor, 0).cpu().numpy()
    
    # Convert v to u
    # u = v / E^p * (H/t)^alpha
    # Phase 2 Config: alpha=0.0, p=1.0
    alpha = 0.0
    epow = 1.0
    
    u_pinn = (v_pred[0, 2] / (E_test**epow)) * ((0.1/h_test)**alpha)
    surr_pred = abs(u_pinn)
    
    print(f"\nPINN Prediction (Direct):")
    print(f"  Predicted Disp: {surr_pred:.5f}")
    
    # 4. Comparison
    err_bend = abs(surr_pred - abs(u_top_bend)) / abs(u_top_bend) * 100
    err_comp = abs(surr_pred - abs(u_top_comp)) / abs(u_top_comp) * 100
    
    print(f"\n--- CONCLUSION ---")
    print(f"Error vs Bending Mode:    {err_bend:.1f}%")
    print(f"Error vs Compression Mode: {err_comp:.1f}%")
    
    if err_bend < 20.0:
        print("\n[SUCCESS] PINN has learned the Bending Mode!")
    elif err_comp < 20.0:
        print("\n[FAILURE] PINN is still stuck in Compression Mode.")
    else:
        print("\n[UNCLEAR] PINN is somewhere in between.")

if __name__ == "__main__":
    run_bc_check()
