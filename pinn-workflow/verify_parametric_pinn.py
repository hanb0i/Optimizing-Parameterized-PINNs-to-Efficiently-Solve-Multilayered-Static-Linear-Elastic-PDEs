
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

# Add paths
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PINN_WORKFLOW_DIR = os.path.join(REPO_ROOT, "pinn-workflow")
FEA_SOLVER_DIR = os.path.join(REPO_ROOT, "fea-workflow", "solver")

if PINN_WORKFLOW_DIR not in sys.path:
    sys.path.insert(0, PINN_WORKFLOW_DIR)
if FEA_SOLVER_DIR not in sys.path:
    sys.path.insert(0, FEA_SOLVER_DIR)

import pinn_config as config
import model
import fem_solver
import physics

def plot_loss_history():
    history_path = os.path.join(REPO_ROOT, "loss_history.npy")
    if not os.path.exists(history_path):
        history_path = os.path.join(PINN_WORKFLOW_DIR, "loss_history.npy")
        
    if not os.path.exists(history_path):
        print(f"Loss history not found at {history_path}")
        return

    history = np.load(history_path, allow_pickle=True).item()
    adam_hist = history.get('adam', {})
    lbfgs_hist = history.get('lbfgs', {})
    
    # Combined Loss Plot
    plt.figure(figsize=(10, 6))
    
    # Plot Adam
    adam_epochs = range(len(adam_hist.get('total', [])))
    plt.semilogy(adam_epochs, adam_hist.get('total', []), label='Adam Total', alpha=0.7)
    if 'pde' in adam_hist and len(adam_hist['pde']) == len(adam_epochs):
        plt.semilogy(adam_epochs, adam_hist['pde'], label='Adam PDE', linestyle='--', alpha=0.5)
    if 'data' in adam_hist and len(adam_hist['data']) == len(adam_epochs):
        plt.semilogy(adam_epochs, adam_hist['data'], label='Adam Data', linestyle=':', alpha=0.5)
    
    # Plot L-BFGS (append to Adam)
    if lbfgs_hist.get('total'):
        start_epoch = len(adam_epochs)
        lbfgs_steps = range(start_epoch, start_epoch + len(lbfgs_hist['total']))
        plt.semilogy(lbfgs_steps, lbfgs_hist['total'], label='L-BFGS Total', alpha=0.7)
        
    plt.xlabel('Epochs/Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss History')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Save
    # Save
    output_dir = os.path.join(REPO_ROOT, "output visualization")
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "loss_history.png")
    plt.savefig(plot_path)
    print(f"Loss history plot saved to {plot_path}")
    plt.close()

def plot_traction_surface(pinn, E_val, H_val, z_val, title_prefix, device):
    pinn.eval()
    
    # Grid setup
    nx, ny = 100, 100
    x = np.linspace(0, config.Lx, nx)
    y = np.linspace(0, config.Ly, ny)
    X, Y = np.meshgrid(x, y)
    
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    # Normalize z to zeta: zeta = z / H
    # z_val is physical coordinate. 
    # If z_val = H_val (top), zeta = 1.0. If z_val = 0 (bottom), zeta = 0.0.
    zeta_val = z_val / H_val
    zeta_flat = np.ones_like(X_flat) * zeta_val
    
    E_flat = np.ones_like(X_flat) * E_val
    H_flat = np.ones_like(X_flat) * H_val
    
    input_pts = np.stack([X_flat, Y_flat, zeta_flat, E_flat, H_flat], axis=1)
    # Convert to tensor and require gradients
    input_tensor = torch.tensor(input_pts, dtype=torch.float32).to(device)
    input_tensor.requires_grad = True
    
    # Forward pass
    v_pred = pinn(input_tensor)
    u_pred = v_pred / E_val # Compliance scaling
    
    # Compute Gradients and Stress
    # Must manually handle derivatives because physics.gradient expects (N, 3) 
    # but we have (N, 5). Wrapper needed or manual extraction.
    # physics.gradient takes (u, x), where x is the input.
    # If x has 5 cols, autograd.grad will return (N, 5).
    # We only care about d/dx, d/dy, d/dzeta (which is col 2).
    
    grad_u_full = torch.autograd.grad(
        u_pred, input_tensor,
        grad_outputs=torch.ones_like(u_pred),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Extract derivatives w.r.t x, y, zeta
    # Chain rule for z: d/dz = (1/H) * d/dzeta
    dudx = grad_u_full[:, 0].reshape(-1, 1) # This is actually sum of du_i/dx for all i which is wrong?
    # Wait, physics.gradient computes grad per component.
    
    # Let's use physics.gradient but we need to feed it the input tensor.
    # physics.gradient implementation:
    # for i in range(3): u_i = u[:, i]; grad_i = autograd.grad(u_i, x, ...)[0]
    # grad_u[:, i, :] = grad_i[:, :3]
    # If x has 5 columns, grad_i will have 5 columns.
    # We can effectively use physics.gradient and then extract/scale.
    
    grad_u_5d = physics.gradient(u_pred, input_tensor) # (N, 3, 5)
    
    # Construct physical gradient (N, 3, 3) -> [dx, dy, dz]
    grad_u_phys = torch.zeros((input_tensor.shape[0], 3, 3), device=device)
    grad_u_phys[:, :, 0] = grad_u_5d[:, :, 0] # dx
    grad_u_phys[:, :, 1] = grad_u_5d[:, :, 1] # dy
    grad_u_phys[:, :, 2] = grad_u_5d[:, :, 2] / H_val # dz = dzeta / H
    
    # Strain and Stress
    nu = config.nu_vals[0]
    lm = (E_val * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E_val / (2 * (1 + nu))
    
    eps = physics.strain(grad_u_phys)
    sig = physics.stress(eps, lm, mu)
    
    # Traction T = sigma . n
    if "Top" in title_prefix:
        n = torch.tensor([0.0, 0.0, 1.0], device=device).view(1, 3, 1).expand(sig.shape[0], 3, 1)
        T = torch.bmm(sig, n).squeeze(2) # (N, 3)
    else: # Bottom
        n = torch.tensor([0.0, 0.0, -1.0], device=device).view(1, 3, 1).expand(sig.shape[0], 3, 1)
        T = torch.bmm(sig, n).squeeze(2) # (N, 3)
        
    T_np = T.detach().cpu().numpy()
    Tx = T_np[:, 0].reshape(ny, nx)
    Ty = T_np[:, 1].reshape(ny, nx)
    Tz = T_np[:, 2].reshape(ny, nx)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    c1 = axes[0].contourf(X, Y, Tx, levels=50, cmap='RdBu_r')
    axes[0].set_title(f"{title_prefix} Tx")
    plt.colorbar(c1, ax=axes[0])
    
    c2 = axes[1].contourf(X, Y, Ty, levels=50, cmap='RdBu_r')
    axes[1].set_title(f"{title_prefix} Ty")
    plt.colorbar(c2, ax=axes[1])
    
    c3 = axes[2].contourf(X, Y, Tz, levels=50, cmap='RdBu_r')
    axes[2].set_title(f"{title_prefix} Tz")
    plt.colorbar(c3, ax=axes[2])
    
    plt.tight_layout()
    filename = f"traction_{title_prefix.lower().replace(' ', '_')}_E{int(E_val)}_H{H_val}.png"
    plt.savefig(os.path.join(REPO_ROOT, filename))
    print(f"Saved {filename}")
    plt.close()

def plot_comparison_detailed(pinn, E_val, H_val, device, u_fea_grid, x_nodes, y_nodes):
    """
    Generates a 3-column comparison plot: FEA Ground Truth | PINN Prediction | Absolute Error
    """
    pinn.eval()
    nx, ny = len(x_nodes), len(y_nodes)
    X, Y = np.meshgrid(x_nodes, y_nodes)
    
    # Extract FEA Top Surface Vertical Displacement
    # u_fea_grid is (nx, ny, nz, 3)
    u_z_fea = u_fea_grid[:, :, -1, 2].T # Transpose to match grid (ny, nx)
    
    # PINN Prediction
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    zeta_flat = np.ones_like(X_flat) # Top surface
    E_flat = np.ones_like(X_flat) * E_val
    H_flat = np.ones_like(X_flat) * H_val
    
    input_pts = np.stack([X_flat, Y_flat, zeta_flat, E_flat, H_flat], axis=1)
    input_tensor = torch.tensor(input_pts, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        v_pred = pinn(input_tensor).cpu().numpy()
    
    u_pinn_flat = v_pred / E_val
    u_z_pinn = u_pinn_flat[:, 2].reshape(ny, nx)
    
    # Error
    error = np.abs(u_z_fea - u_z_pinn)
    mae = np.mean(error)
    
    # Global range for consistent colorbar
    vmin = min(np.min(u_z_fea), np.min(u_z_pinn))
    vmax = max(np.max(u_z_fea), np.max(u_z_pinn))
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. FEA
    c1 = axes[0].contourf(X, Y, u_z_fea, levels=50, cmap='turbo', vmin=vmin, vmax=vmax)
    axes[0].set_title(f"FEA (Ground Truth)\nPeak: {np.min(u_z_fea):.4f}")
    plt.colorbar(c1, ax=axes[0])
    
    # 2. PINN
    c2 = axes[1].contourf(X, Y, u_z_pinn, levels=50, cmap='turbo', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"PINN Prediction\nPeak: {np.min(u_z_pinn):.4f}")
    plt.colorbar(c2, ax=axes[1])
    
    # 3. Error
    c3 = axes[2].contourf(X, Y, error, levels=50, cmap='inferno')
    axes[2].set_title(f"Absolute Error\nMAE: {mae:.4f}")
    plt.colorbar(c3, ax=axes[2])
    
    plt.suptitle(f"Top Surface Deflection (vertical) | E={E_val}, H={H_val}")
    plt.tight_layout()
    
    filename = f"comparison_E{int(E_val)}_H{H_val}.png"
    output_dir = os.path.join(REPO_ROOT, "output visualization")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    print(f"Saved comparison plot: {save_path}")
    plt.close()

def plot_pde_residuals(pinn, E_val, H_val, device):
    pinn.eval()
    
    # X-Z Plane at y = Ly/2
    nx, nz = 100, 50
    x = np.linspace(0, config.Lx, nx)
    z = np.linspace(0, H_val, nz) # Physical z
    X, Z = np.meshgrid(x, z, indexing='ij') # (nx, nz)
    
    X_flat = X.flatten()
    Z_flat = Z.flatten()
    Y_flat = np.ones_like(X_flat) * (config.Ly / 2.0)
    
    # Normalize z -> zeta
    zeta_flat = Z_flat / H_val
    E_flat = np.ones_like(X_flat) * E_val
    H_flat = np.ones_like(X_flat) * H_val
    
    input_pts = np.stack([X_flat, Y_flat, zeta_flat, E_flat, H_flat], axis=1)
    input_tensor = torch.tensor(input_pts, dtype=torch.float32).to(device)
    input_tensor.requires_grad = True
    
    # Forward Pass
    v_pred = pinn(input_tensor)
    u_pred = v_pred / E_val
    
    # Gradients
    grad_u_5d = physics.gradient(u_pred, input_tensor)
    grad_u_phys = torch.zeros((input_tensor.shape[0], 3, 3), device=device)
    grad_u_phys[:, :, 0] = grad_u_5d[:, :, 0]
    grad_u_phys[:, :, 1] = grad_u_5d[:, :, 1]
    grad_u_phys[:, :, 2] = grad_u_5d[:, :, 2] / H_val
    
    # Stress
    nu = config.nu_vals[0]
    lm = (E_val * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E_val / (2 * (1 + nu))
    
    eps = physics.strain(grad_u_phys)
    sig = physics.stress(eps, lm, mu)
    
    # Divergence of Stress (Equilibrium)
    # div(sigma)_i = d(sigma_ij)/dx_j
    # Need derivatives of sigma w.r.t input coordinates
    
    div_sig = torch.zeros(input_tensor.shape[0], 3, device=device)
    
    for i in range(3):
        div_i = 0
        for j in range(3):
            sig_ij = sig[:, i, j].unsqueeze(1)
            grad_sig_ij_5d = torch.autograd.grad(
                sig_ij, input_tensor,
                grad_outputs=torch.ones_like(sig_ij),
                create_graph=False, # No graph needed for visualization
                retain_graph=True
            )[0]
            
            # Map 5D gradients to physical 3D gradients
            # col 0 -> dx, col 1 -> dy, col 2 -> dzeta
            d_dx = grad_sig_ij_5d[:, 0]
            d_dy = grad_sig_ij_5d[:, 1]
            d_dz = grad_sig_ij_5d[:, 2] / H_val
            
            if j == 0: div_i += d_dx
            elif j == 1: div_i += d_dy
            elif j == 2: div_i += d_dz
            
        div_sig[:, i] = div_i

    # Residual = -div(sigma) (ignoring body forces for now, typically 0)
    residual = -div_sig
    residual_mag = torch.sqrt(torch.sum(residual**2, dim=1))
    
    # Reshape for plotting
    res_mag_grid = residual_mag.detach().cpu().numpy().reshape(nx, nz)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.contourf(X, Z, res_mag_grid, levels=50, cmap='viridis')
    plt.colorbar(label='PDE Residual Magnitude')
    plt.title(f"PDE Residual (XZ Plane, y=0.5) E={E_val}, H={H_val}")
    plt.xlabel("x")
    plt.ylabel("z")
    
    filename = f"pde_residual_E{int(E_val)}_H{H_val}.png"
    plt.savefig(os.path.join(REPO_ROOT, filename))
    print(f"Saved {filename}")
    plt.close()


def run_fea(E_val, H_val):
    print(f"Running FEA for E={E_val}, H={H_val}...")
    # Mock config for FEA solver
    cfg = {
        'geometry': {'Lx': config.Lx, 'Ly': config.Ly, 'H': H_val},
        'material': {'E': E_val, 'nu': config.nu_vals[0]},
        'load_patch': {
            'pressure': config.p0,
            'x_start': config.LOAD_PATCH_X[0]/config.Lx,
            'x_end': config.LOAD_PATCH_X[1]/config.Lx,
            'y_start': config.LOAD_PATCH_Y[0]/config.Ly,
            'y_end': config.LOAD_PATCH_Y[1]/config.Ly
        }
    }
    x, y, z, u = fem_solver.solve_fem(cfg)
    return x, y, z, u

def main():
    E_test_values = [1.0, 10.0] # Selected values for clear comparison
    H_test_values = [0.1, 0.5, 1.0] # Requested H variations
    results = {}

    # Load PINN
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    pinn = model.MultiLayerPINN().to(device)
    model_path = os.path.join(REPO_ROOT, "pinn_model.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(PINN_WORKFLOW_DIR, "pinn_model.pth")
        
    print(f"Loading model from: {model_path}")
    pinn.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    pinn.eval()
    
    # Plot training loss history
    plot_loss_history()

    # Create figure: rows are H variations, columns are E variations
    fig, axes = plt.subplots(len(H_test_values), len(E_test_values), figsize=(12, 12))

    for h_idx, H_val in enumerate(H_test_values):
        for e_idx, E_val in enumerate(E_test_values):
            # 1. Run FEA
            x_nodes, y_nodes, z_nodes, u_fea_grid = run_fea(E_val, H_val)
            
            # Extract Top Surface for Visualization
            # u_fea_grid shape: (nx, ny, nz, 3)
            u_z_fea_top = u_fea_grid[:, :, -1, 2].T # Transpose for pcolormesh (y, x)
            
            # Meshgrid for plotting
            X, Y = np.meshgrid(x_nodes, y_nodes)
            
            # 2. Run PINN
            # Create grid points matching FEA nodes
            nx, ny = len(x_nodes), len(y_nodes)
            X_flat = X.flatten()
            Y_flat = Y.flatten()
            zeta_flat = np.ones_like(X_flat) # Top surface is zeta=1
            E_flat = np.ones_like(X_flat) * E_val
            H_flat = np.ones_like(X_flat) * H_val
            
            # Prepare input (N, 5)
            input_pts = np.stack([X_flat, Y_flat, zeta_flat, E_flat, H_flat], axis=1)
            input_tensor = torch.tensor(input_pts, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                v_pinn_flat = pinn(input_tensor).cpu().numpy()
                
            # Physics compliance scaling: u = v / E
            u_pinn_flat = v_pinn_flat / (E_val)
                
            u_z_pinn_top = u_pinn_flat[:, 2].reshape(ny, nx)
            
            # 3. Compute Error
            abs_diff = np.abs(u_z_fea_top - u_z_pinn_top)
            mae = np.mean(abs_diff)
            peak_fea = np.min(u_z_fea_top)
            peak_pinn = np.min(u_z_pinn_top)
            
            print(f"  Result for E={E_val}, H={H_val}: Peak FEA={peak_fea:.4f}, Peak PINN={peak_pinn:.4f}, MAE={mae:.4f}")
            
            # Additional Visualizations
            plot_comparison_detailed(pinn, E_val, H_val, device, u_fea_grid, x_nodes, y_nodes)
            # plot_traction_surface(pinn, E_val, H_val, H_val, "Top Surface", device)
            # plot_traction_surface(pinn, E_val, H_val, 0.0, "Bottom Surface", device)
            # plot_pde_residuals(pinn, E_val, H_val, device)
            
            # 4. Plot (PINN prediction on top surface)
            ax = axes[h_idx, e_idx]
            c = ax.contourf(X, Y, u_z_pinn_top, levels=50, cmap="jet")
            ax.set_title(f"H={H_val}, E={E_val}\nMAE: {mae:.4f}")
            plt.colorbar(c, ax=ax)

    plt.tight_layout()
    plt.tight_layout()
    output_dir = os.path.join(REPO_ROOT, "output visualization")
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, "parametric_verification.png")
    plt.savefig(result_path)
    print(f"\nVerification plot saved to: {result_path}")
    # plt.show()

if __name__ == "__main__":
    main()
