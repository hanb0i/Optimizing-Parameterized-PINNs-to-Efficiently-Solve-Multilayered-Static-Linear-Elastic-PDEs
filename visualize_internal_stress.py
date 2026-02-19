
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory
sys.path.append("pinn-workflow")
import model
import physics
import pinn_config as config

def visualize_stress():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load Model
    pinn = model.MultiLayerPINN().to(device)
    ckpt_path = "pinn_model.pth"
    print(f"Loading model from {ckpt_path}...")
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    pinn.load_state_dict(sd)
    pinn.eval()
    
    # Create Cross-Section Grid (y=0.5)
    nx, nz = 100, 50
    x = np.linspace(0, config.Lx, nx)
    z = np.linspace(0, getattr(config, 'H', 0.1), nz)
    X, Z = np.meshgrid(x, z)
    
    y_val = config.Ly / 2.0
    
    # Flatten
    pts = np.zeros((nx*nz, 11 + 2*getattr(config, 'FOURIER_DIM', 0))) # Base dim + Fourier
    # Actually model handles Fourier internal expansion if we pass base inputs?
    # Let's check model.py forward. 
    # forward(x) takes raw inputs? No, forward takes `x` which is `current_dim` sized?
    # model.py:
    # `x_coord = x[:, 0:1]` ...
    # `extra_feats` computed inside.
    # `fourier_features` computed inside.
    # So input `x` should be (N, 8). 
    # [x, y, z, E, t, r, mu, v0]
    
    pts = np.zeros((nx*nz, 8), dtype=np.float32)
    pts[:, 0] = X.ravel()
    pts[:, 1] = y_val
    pts[:, 2] = Z.ravel()
    pts[:, 3] = 1.0 # E
    pts[:, 4] = getattr(config, 'H', 0.1) # Thickness
    pts[:, 5] = 0.5 # Restitution
    pts[:, 6] = 0.3 # Friction
    pts[:, 7] = 1.0 # V0
    
    pts_tensor = torch.tensor(pts, dtype=torch.float32).to(device)
    pts_tensor.requires_grad_(True)
    
    # Compute Gradients & Stress
    v = pinn(pts_tensor, 0)
    E_tens = pts_tensor[:, 3:4]
    t_tens = pts_tensor[:, 4:5]
    
    # Compliance Scaling (Check config)
    # Exp C config: THICKNESS_COMPLIANCE_ALPHA = 0.0
    u = v / E_tens 
    
    grad_u = physics.gradient(u, pts_tensor)
    eps = physics.strain(grad_u)
    
    # Material Props
    nu = 0.3
    lm = (E_tens * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E_tens / (2 * (1 + nu))
    lm = lm.unsqueeze(2)
    mu = mu.unsqueeze(2)
    
    sigma = physics.stress(eps, lm, mu)
    
    # Extract components
    s_xx = sigma[:, 0, 0].detach().cpu().numpy().reshape(nz, nx)
    s_zz = sigma[:, 2, 2].detach().cpu().numpy().reshape(nz, nx)
    s_xz = sigma[:, 0, 2].detach().cpu().numpy().reshape(nz, nx)
    u_z = u[:, 2].detach().cpu().numpy().reshape(nz, nx)
    
    # Plot
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    # Uz
    c = axs[0,0].contourf(X, Z, u_z, levels=50, cmap='jet')
    plt.colorbar(c, ax=axs[0,0])
    axs[0,0].set_title("Vertical Displacement Uz")
    
    # Sigma XX (Bending Stress)
    c = axs[0,1].contourf(X, Z, s_xx, levels=50, cmap='RdBu_r')
    plt.colorbar(c, ax=axs[0,1])
    axs[0,1].set_title("Bending Stress Sigma_xx")

    # Sigma ZZ (Vertical Comp)
    c = axs[1,0].contourf(X, Z, s_zz, levels=50, cmap='RdBu_r')
    plt.colorbar(c, ax=axs[1,0])
    axs[1,0].set_title("Vertical Stress Sigma_zz")
    
    # Sigma XZ (Shear)
    c = axs[1,1].contourf(X, Z, s_xz, levels=50, cmap='RdBu_r')
    plt.colorbar(c, ax=axs[1,1])
    axs[1,1].set_title("Shear Stress Sigma_xz")
    
    plt.tight_layout()
    plt.savefig("pinn_internal_stress.png")
    print("Saved 'pinn_internal_stress.png'")

if __name__ == "__main__":
    visualize_stress()
