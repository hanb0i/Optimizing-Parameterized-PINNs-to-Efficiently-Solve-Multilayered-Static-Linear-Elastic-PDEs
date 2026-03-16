
import torch
import numpy as np
import os
import pinn_config as config

# Parameter range helper (keeps baseline configs intact)
def _get_e_range():
    if hasattr(config, "E_RANGE"):
        e_min, e_max = config.E_RANGE
    else:
        e_vals = getattr(config, "E_vals", [1.0])
        e_min, e_max = min(e_vals), max(e_vals)
        if e_min == e_max:
            e_max = e_min + 1.0
    return float(e_min), float(e_max)

def _get_thickness_range():
    if hasattr(config, "THICKNESS_RANGE"):
        t_min, t_max = config.THICKNESS_RANGE
    else:
        t_min, t_max = float(getattr(config, "H", 0.1)), float(getattr(config, "H", 0.1))
        if t_min == t_max:
            t_max = t_min + 0.01
    return float(t_min), float(t_max)

def _get_restitution_range():
    if hasattr(config, "RESTITUTION_RANGE"):
        r_min, r_max = config.RESTITUTION_RANGE
    else:
        r_min, r_max = 0.5, 0.5
    return float(r_min), float(r_max)

def _get_friction_range():
    if hasattr(config, "FRICTION_RANGE"):
        mu_min, mu_max = config.FRICTION_RANGE
    else:
        mu_min, mu_max = 0.3, 0.3
    return float(mu_min), float(mu_max)

def _get_impact_velocity_range():
    if hasattr(config, "IMPACT_VELOCITY_RANGE"):
        v0_min, v0_max = config.IMPACT_VELOCITY_RANGE
    else:
        v0_min, v0_max = 1.0, 1.0
    return float(v0_min), float(v0_max)

# Import FEM solver for generating supervision data
import sys
FEA_SOLVER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fea-workflow", "solver")
if FEA_SOLVER_DIR not in sys.path:
    sys.path.insert(0, FEA_SOLVER_DIR)


def load_fem_supervision_data(n_points_per_e=None, e_values=None, thickness_values=None):
    import fem_solver
    
    if e_values is None:
        if hasattr(config, "DATA_E_VALUES"):
            e_values = config.DATA_E_VALUES
        else:
            e_min, e_max = _get_e_range()
            e_values = [e_min, 0.5 * (e_min + e_max), e_max]
    if thickness_values is None:
        if hasattr(config, "DATA_THICKNESS_VALUES"):
            thickness_values = config.DATA_THICKNESS_VALUES
        else:
            t_min, t_max = _get_thickness_range()
            thickness_values = [t_min, 0.5 * (t_min + t_max), t_max]
    
    if n_points_per_e is None:
        if hasattr(config, "N_DATA_POINTS"):
            n_points_per_e = config.N_DATA_POINTS // max(1, (len(e_values) * len(thickness_values)))
        else:
            n_points_per_e = 0
    
    x_data_list = []
    u_data_list = []
    
    for thickness in thickness_values:
        for E_val in e_values:
            print(f"  Generating FEM supervision for E={E_val}, thickness={thickness}...")
            
            # Run FEM solver
            cfg = {
                'geometry': {'Lx': config.Lx, 'Ly': config.Ly, 'H': thickness},
                'material': {'E': E_val, 'nu': config.nu_vals[0]},
                'load_patch': {
                    'pressure': config.p0,
                    'x_start': config.LOAD_PATCH_X[0] / config.Lx,
                    'x_end': config.LOAD_PATCH_X[1] / config.Lx,
                    'y_start': config.LOAD_PATCH_Y[0] / config.Ly,
                    'y_end': config.LOAD_PATCH_Y[1] / config.Ly
                }
            }
            x_nodes, y_nodes, z_nodes, u_grid = fem_solver.solve_fem(cfg)
            
            # Create mesh grid for all FEM nodes
            nx, ny, nz = len(x_nodes), len(y_nodes), len(z_nodes)
            X, Y, Z = np.meshgrid(x_nodes, y_nodes, z_nodes, indexing='ij')
            
            # Flatten to get all points
            x_flat = X.flatten()
            y_flat = Y.flatten()
            z_flat = Z.flatten()
            u_flat = u_grid.reshape(-1, 3)
            
            # Random sampling (sparse)
            total_points = len(x_flat)
            indices = np.random.choice(total_points, size=min(n_points_per_e, total_points), replace=False)
            
            # Create input points with E and thickness values (2-layer layout)
            r_min, r_max = _get_restitution_range()
            mu_min, mu_max = _get_friction_range()
            v0_min, v0_max = _get_impact_velocity_range()
            restitution = np.ones(len(indices)) * (0.5 * (r_min + r_max))
            friction = np.ones(len(indices)) * (0.5 * (mu_min + mu_max))
            impact_velocity = np.ones(len(indices)) * (0.5 * (v0_min + v0_max))

            t1 = 0.5 * thickness
            t2 = thickness - t1
            x_sampled = np.stack(
                [
                    x_flat[indices],
                    y_flat[indices],
                    z_flat[indices],
                    np.ones(len(indices)) * E_val,
                    np.ones(len(indices)) * t1,
                    np.ones(len(indices)) * E_val,
                    np.ones(len(indices)) * t2,
                    restitution,
                    friction,
                    impact_velocity,
                ],
                axis=1,
            )
            
            u_sampled = u_flat[indices]
            
            x_data_list.append(torch.tensor(x_sampled, dtype=torch.float32))
            u_data_list.append(torch.tensor(u_sampled, dtype=torch.float32))
    
    x_data = torch.cat(x_data_list, dim=0)
    u_data = torch.cat(u_data_list, dim=0)
    
    print(f"  Loaded {len(x_data)} sparse FEM supervision points")
    return x_data, u_data


def sample_domain(n, z_min=0.0, z_max=0.1):
    # n points, 10D: [x, y, z, E1, t1, E2, t2, r, mu, v0]
    e_min, e_max = _get_e_range()
    t_min, t_max = _get_thickness_range()
    r_min, r_max = _get_restitution_range()
    mu_min, mu_max = _get_friction_range()
    v0_min, v0_max = _get_impact_velocity_range()

    x = torch.rand(n, 1) * config.Lx
    y = torch.rand(n, 1) * config.Ly
    
    e1 = torch.rand(n, 1) * (e_max - e_min) + e_min
    e2 = torch.rand(n, 1) * (e_max - e_min) + e_min
    
    t1 = torch.rand(n, 1) * (t_max - t_min) + t_min
    t2 = torch.rand(n, 1) * (t_max - t_min) + t_min
    
    r = torch.rand(n, 1) * (r_max - r_min) + r_min
    mu = torch.rand(n, 1) * (mu_max - mu_min) + mu_min
    v0 = torch.rand(n, 1) * (v0_max - v0_min) + v0_min
    
    t_total = t1 + t2
    z = torch.rand(n, 1) * t_total
    
    return torch.cat([x, y, z, e1, t1, e2, t2, r, mu, v0], dim=1)

def sample_domain_under_patch(n, z_min=0.0, z_max=0.1):
    pts = sample_domain(n)
    pts[:, 0] = torch.rand(n) * (config.LOAD_PATCH_X[1] - config.LOAD_PATCH_X[0]) + config.LOAD_PATCH_X[0]
    pts[:, 1] = torch.rand(n) * (config.LOAD_PATCH_Y[1] - config.LOAD_PATCH_Y[0]) + config.LOAD_PATCH_Y[0]
    return pts

def sample_domain_residual_based(n, z_min, z_max, prev_pts, prev_residuals):
    if prev_residuals.sum() < 1e-12 or torch.isnan(prev_residuals).any():
        return sample_domain(n)
    
    residual_probs = prev_residuals / prev_residuals.sum()
    residual_probs = residual_probs + 1e-10
    residual_probs = residual_probs / residual_probs.sum()
    
    indices = torch.multinomial(residual_probs, n, replacement=True)
    sampled_pts = prev_pts[indices]
    
    noise_scale = getattr(config, "SAMPLING_NOISE_SCALE", 0.05)
    noise = (torch.rand(n, 10) - 0.5) * 2 * noise_scale
    # Scale noise by range for each dimension
    # [Lx, Ly, t_total, E, E, E, t, t, t, r, mu, v0]
    # Simple uniform noise is fine for now as a perturbation
    new_pts = sampled_pts + noise * 0.1 # Dampened noise for stability
    
    # Clamp
    new_pts[:, 0] = torch.clamp(new_pts[:, 0], 0, config.Lx)
    new_pts[:, 1] = torch.clamp(new_pts[:, 1], 0, config.Ly)
    e_min, e_max = _get_e_range()
    new_pts[:, 3] = torch.clamp(new_pts[:, 3], e_min, e_max)
    new_pts[:, 5] = torch.clamp(new_pts[:, 5], e_min, e_max)
    t_min, t_max = _get_thickness_range()
    new_pts[:, 4] = torch.clamp(new_pts[:, 4], t_min, t_max)
    new_pts[:, 6] = torch.clamp(new_pts[:, 6], t_min, t_max)
    r_min, r_max = _get_restitution_range()
    new_pts[:, 7] = torch.clamp(new_pts[:, 7], r_min, r_max)
    mu_min, mu_max = _get_friction_range()
    new_pts[:, 8] = torch.clamp(new_pts[:, 8], mu_min, mu_max)
    v0_min, v0_max = _get_impact_velocity_range()
    new_pts[:, 9] = torch.clamp(new_pts[:, 9], v0_min, v0_max)
    
    t_total = new_pts[:, 4] + new_pts[:, 6]
    new_pts[:, 2] = torch.clamp(new_pts[:, 2], 0.0, t_total)
    
    return new_pts

def sample_boundaries(n):
    # n points, 10D [x,y,z,E1,t1,E2,t2,r,mu,v0]
    pts = sample_domain(n)
    # Pick faces: 0:x=0, 1:x=Lx, 2:y=0, 3:y=Ly
    faces = torch.randint(0, 4, (n,))
    for i in range(n):
        if faces[i] == 0: pts[i, 0] = 0.0
        elif faces[i] == 1: pts[i, 0] = config.Lx
        elif faces[i] == 2: pts[i, 1] = 0.0
        elif faces[i] == 3: pts[i, 1] = config.Ly
    return pts

def sample_top_load(n):
    pts = sample_domain_under_patch(n)
    # z = total thickness
    pts[:, 2] = pts[:, 4] + pts[:, 6]
    return pts

def sample_top_free(n):
    pts = sample_domain(n)
    # Rejection for top free (outside patch)
    x_min, x_max = config.LOAD_PATCH_X
    y_min, y_max = config.LOAD_PATCH_Y
    for i in range(n):
        while True:
            pts[i, 0] = torch.rand(1) * config.Lx
            pts[i, 1] = torch.rand(1) * config.Ly
            if not (x_min < pts[i, 0] < x_max and y_min < pts[i, 1] < y_max):
                break
    pts[:, 2] = pts[:, 4] + pts[:, 6]
    return pts

def sample_bottom(n):
    pts = sample_domain(n)
    pts[:, 2] = 0.0
    return pts

def sample_interface(n):
    pts = sample_domain(n)
    pts[:, 2] = pts[:, 4]  # z = t1
    return pts

def get_data(prev_data=None, residuals=None):
    # n_patch allocation
    n_patch = int(config.N_INTERIOR * config.UNDER_PATCH_FRACTION)
    n_interior = config.N_INTERIOR - n_patch
    
    # 1. Interior
    interior = sample_domain(n_interior)
    if n_patch > 0:
        patch = sample_domain_under_patch(n_patch)
        interior = torch.cat([interior, patch], dim=0)
    
    # 2. Boundaries & Surfaces
    sides = sample_boundaries(config.N_SIDES)
    top_load = sample_top_load(config.N_TOP_LOAD)
    top_free = sample_top_free(config.N_TOP_FREE)
    bottom = sample_bottom(config.N_BOTTOM)
    
    # 3. Interfaces (Physics Anchors)
    interface1 = sample_interface(2000)
    
    return {
        'interior': [interior],
        'sides': [sides],
        'top_load': top_load,
        'top_free': top_free,
        'bottom': bottom,
        'interface1': interface1,
    }
