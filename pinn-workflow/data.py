
import torch
import numpy as np
import os
import pinn_config as config

# Parameter range helper (keeps baseline configs intact)
def _get_param_ranges():
    e_min, e_max = getattr(config, "E_RANGE", [1.0, 10.0])
    h_min, h_max = getattr(config, "H_RANGE", [0.1, 1.0])
    return (float(e_min), float(e_max)), (float(h_min), float(h_max))

def _get_e_range():
    (e_min, e_max), _ = _get_param_ranges()
    return e_min, e_max

def _get_h_range():
    _, (h_min, h_max) = _get_param_ranges()
    return h_min, h_max

# Import FEM solver for generating supervision data
import sys
FEA_SOLVER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fea-workflow", "solver")
if FEA_SOLVER_DIR not in sys.path:
    sys.path.insert(0, FEA_SOLVER_DIR)


def load_fem_supervision_data(n_points_per_e=None, e_values=None):
    """Load sparse FEM supervision data for hybrid training.
    
    Generates FEM solutions for each E value and samples sparse points.
    
    Args:
        n_points_per_e: Number of points to sample per E value (default: config.N_DATA_POINTS // len(E))
        e_values: List of E values to sample (default: config.DATA_E_VALUES)
    
    Returns:
        x_data: (N, 4) tensor of input points [x, y, z, E]
        u_data: (N, 3) tensor of target displacements [ux, uy, uz]
    """
    import fem_solver
    
    if e_values is None:
        if hasattr(config, "DATA_E_VALUES"):
            e_values = config.DATA_E_VALUES
        else:
            e_min, e_max = _get_e_range()
            e_values = [e_min, 0.5 * (e_min + e_max), e_max]
    if n_points_per_e is None:
        if hasattr(config, "N_DATA_POINTS"):
            n_points_per_e = config.N_DATA_POINTS // len(e_values)
        else:
            n_points_per_e = 0
    
    x_data_list = []
    u_data_list = []
    
    for E_val in e_values:
        print(f"  Generating FEM supervision for E={E_val}...")
        
        # Run FEM solver
        cfg = {
            'geometry': {'Lx': config.Lx, 'Ly': config.Ly, 'H': config.H},
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
        
        # Create input points with E value
        x_sampled = np.stack([
            x_flat[indices],
            y_flat[indices],
            z_flat[indices],
            np.ones(len(indices)) * E_val,
            np.ones(len(indices)) * config.H # FEM data is fixed at baseline H
        ], axis=1)
        
        u_sampled = u_flat[indices]
        
        x_data_list.append(torch.tensor(x_sampled, dtype=torch.float32))
        u_data_list.append(torch.tensor(u_sampled, dtype=torch.float32))
    
    x_data = torch.cat(x_data_list, dim=0)
    u_data = torch.cat(u_data_list, dim=0)
    
    print(f"  Loaded {len(x_data)} sparse FEM supervision points")
    return x_data, u_data


def sample_domain(n, z_min, z_max):
    # Uniform sampling
    x = torch.rand(n, 1) * config.Lx
    y = torch.rand(n, 1) * config.Ly
    # Normalized z (zeta)
    zeta = torch.rand(n, 1) # [0, 1]
    
    # Sample Parameters (E, H)
    (e_min, e_max), (h_min, h_max) = _get_param_ranges()
    e = torch.rand(n, 1) * (e_max - e_min) + e_min
    h = torch.rand(n, 1) * (h_max - h_min) + h_min
    
    return torch.cat([x, y, zeta, e, h], dim=1)

def sample_domain_under_patch(n, z_min, z_max):
    """Sample interior points directly under the load patch."""
    x_min, x_max = config.LOAD_PATCH_X
    y_min, y_max = config.LOAD_PATCH_Y
    x = torch.rand(n, 1) * (x_max - x_min) + x_min
    y = torch.rand(n, 1) * (y_max - y_min) + y_min
    zeta = torch.rand(n, 1) # [0, 1]
    
    (e_min, e_max), (h_min, h_max) = _get_param_ranges()
    e = torch.rand(n, 1) * (e_max - e_min) + e_min
    h = torch.rand(n, 1) * (h_max - h_min) + h_min
    
    return torch.cat([x, y, zeta, e, h], dim=1)

def sample_domain_residual_based(n, z_min, z_max, prev_pts, prev_residuals):
    """Sample points weighted by residual magnitude."""
    # Check if residuals are too small - fall back to uniform sampling
    if prev_residuals.sum() < 1e-12 or torch.isnan(prev_residuals).any():
        return sample_domain(n, z_min, z_max)
    
    # Normalize residuals to probabilities
    residual_probs = prev_residuals / prev_residuals.sum()
    residual_probs = residual_probs + 1e-10  # Add small epsilon for numerical stability
    residual_probs = residual_probs / residual_probs.sum()  # Renormalize
    
    # Sample indices based on residual weights
    indices = torch.multinomial(residual_probs, n, replacement=True)
    sampled_pts = prev_pts[indices]
    
    # Add noise to create new points nearby
    noise_scale = getattr(config, "SAMPLING_NOISE_SCALE", 0.05)
    noise_x = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * config.Lx
    noise_y = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * config.Ly
    noise_zeta = (torch.rand(n, 1) - 0.5) * 2 * noise_scale # On normalized scale [0, 1]
    
    (e_min, e_max), (h_min, h_max) = _get_param_ranges()
    noise_e = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * (e_max - e_min)
    noise_h = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * (h_max - h_min)
    
    noise = torch.cat([noise_x, noise_y, noise_zeta, noise_e, noise_h], dim=1)
    
    new_pts = sampled_pts + noise
    
    # Clamp to domain bounds
    new_pts[:, 0] = torch.clamp(new_pts[:, 0], 0, config.Lx)
    new_pts[:, 1] = torch.clamp(new_pts[:, 1], 0, config.Ly)
    new_pts[:, 2] = torch.clamp(new_pts[:, 2], 0.0, 1.0) # Zeta
    new_pts[:, 3] = torch.clamp(new_pts[:, 3], e_min, e_max)
    new_pts[:, 4] = torch.clamp(new_pts[:, 4], h_min, h_max)
    
    return new_pts

def sample_boundaries(n, z_min, z_max):
    # 4 Side faces: x=0, x=Lx, y=0, y=Ly
    n_face = n // 4
    
    (e_min, e_max), (h_min, h_max) = _get_param_ranges()
    # x=0
    y1 = torch.rand(n_face, 1) * config.Ly
    zeta1 = torch.rand(n_face, 1) # Normalized
    x1 = torch.zeros(n_face, 1)
    e1 = torch.rand(n_face, 1) * (e_max - e_min) + e_min
    h1 = torch.rand(n_face, 1) * (h_max - h_min) + h_min
    p1 = torch.cat([x1, y1, zeta1, e1, h1], dim=1)
    
    # x=Lx
    y2 = torch.rand(n_face, 1) * config.Ly
    zeta2 = torch.rand(n_face, 1)
    x2 = torch.ones(n_face, 1) * config.Lx
    e2 = torch.rand(n_face, 1) * (e_max - e_min) + e_min
    h2 = torch.rand(n_face, 1) * (h_max - h_min) + h_min
    p2 = torch.cat([x2, y2, zeta2, e2, h2], dim=1)
    
    # y=0
    x3 = torch.rand(n_face, 1) * config.Lx
    zeta3 = torch.rand(n_face, 1)
    y3 = torch.zeros(n_face, 1)
    e3 = torch.rand(n_face, 1) * (e_max - e_min) + e_min
    h3 = torch.rand(n_face, 1) * (h_max - h_min) + h_min
    p3 = torch.cat([x3, y3, zeta3, e3, h3], dim=1)
    
    # y=Ly
    x4 = torch.rand(n_face, 1) * config.Lx
    zeta4 = torch.rand(n_face, 1)
    y4 = torch.ones(n_face, 1) * config.Ly
    e4 = torch.rand(n_face, 1) * (e_max - e_min) + e_min
    h4 = torch.rand(n_face, 1) * (h_max - h_min) + h_min
    p4 = torch.cat([x4, y4, zeta4, e4, h4], dim=1)
    
    return torch.cat([p1, p2, p3, p4], dim=0)

def sample_boundaries_residual_based(n, z_min, z_max, prev_pts, prev_residuals):
    """Sample boundary points weighted by BC residual."""
    # Check if residuals are too small - fall back to uniform sampling
    if prev_residuals.sum() < 1e-12 or torch.isnan(prev_residuals).any():
        return sample_boundaries(n, z_min, z_max)
    
    residual_probs = prev_residuals / prev_residuals.sum()
    residual_probs = residual_probs + 1e-10
    residual_probs = residual_probs / residual_probs.sum()
    indices = torch.multinomial(residual_probs, n, replacement=True)
    sampled_pts = prev_pts[indices]
    
    noise_scale = getattr(config, "SAMPLING_NOISE_SCALE", 0.05)
    # Keep boundary constraints while perturbing
    new_pts = sampled_pts.clone()
    
    # Add noise to parameters for all points
    (e_min, e_max), (h_min, h_max) = _get_param_ranges()
    noise_e = (torch.rand(n) - 0.5) * 2 * noise_scale * (e_max - e_min)
    noise_h = (torch.rand(n) - 0.5) * 2 * noise_scale * (h_max - h_min)
    new_pts[:, 3] += noise_e
    new_pts[:, 4] += noise_h
    
    # For each face, perturb only the non-fixed coordinates
    for i in range(n):
        pt = new_pts[i]
        if torch.abs(pt[0]) < 1e-6:  # x=0 face
            new_pts[i, 1] += (torch.rand((), device=new_pts.device) - 0.5) * 2 * noise_scale * config.Ly
            new_pts[i, 2] += (torch.rand((), device=new_pts.device) - 0.5) * 2 * noise_scale # Zeta
            new_pts[i, 0] = 0.0
        elif torch.abs(pt[0] - config.Lx) < 1e-6:  # x=Lx face
            new_pts[i, 1] += (torch.rand((), device=new_pts.device) - 0.5) * 2 * noise_scale * config.Ly
            new_pts[i, 2] += (torch.rand((), device=new_pts.device) - 0.5) * 2 * noise_scale # Zeta
            new_pts[i, 0] = config.Lx
        elif torch.abs(pt[1]) < 1e-6:  # y=0 face
            new_pts[i, 0] += (torch.rand((), device=new_pts.device) - 0.5) * 2 * noise_scale * config.Lx
            new_pts[i, 2] += (torch.rand((), device=new_pts.device) - 0.5) * 2 * noise_scale # Zeta
            new_pts[i, 1] = 0.0
        elif torch.abs(pt[1] - config.Ly) < 1e-6:  # y=Ly face
            new_pts[i, 0] += (torch.rand((), device=new_pts.device) - 0.5) * 2 * noise_scale * config.Lx
            new_pts[i, 2] += (torch.rand((), device=new_pts.device) - 0.5) * 2 * noise_scale # Zeta
            new_pts[i, 1] = config.Ly
    
    # Clamp
    new_pts[:, 0] = torch.clamp(new_pts[:, 0], 0, config.Lx)
    new_pts[:, 1] = torch.clamp(new_pts[:, 1], 0, config.Ly)
    new_pts[:, 2] = torch.clamp(new_pts[:, 2], z_min, z_max)
    new_pts[:, 3] = torch.clamp(new_pts[:, 3], e_min, e_max)
    
    return new_pts

def sample_top_load(n):
    """Sample points on load patch only."""
    # Loaded Patch: Lx/3 < x < 2Lx/3 AND Ly/3 < y < 2Ly/3
    xl = torch.rand(n, 1) * (config.Lx/3) + config.Lx/3
    yl = torch.rand(n, 1) * (config.Ly/3) + config.Ly/3
    zeta_l = torch.ones(n, 1) # Normalized top surface
    (e_min, e_max), (h_min, h_max) = _get_param_ranges()
    el = torch.rand(n, 1) * (e_max - e_min) + e_min
    hl = torch.rand(n, 1) * (h_max - h_min) + h_min
    return torch.cat([xl, yl, zeta_l, el, hl], dim=1)

def sample_top_free(n):
    """Sample points on free top surface (outside load patch)."""
    # Rejection sampling for points outside patch
    pts_free_list = []
    count = 0
    while count < n:
        batch = 1000
        x = torch.rand(batch, 1) * config.Lx
        y = torch.rand(batch, 1) * config.Ly
        
        in_patch = (x > config.Lx/3) & (x < 2*config.Lx/3) & \
                   (y > config.Ly/3) & (y < 2*config.Ly/3)
        
        mask_free = ~in_patch.squeeze()
        xf, yf = x[mask_free], y[mask_free]
        if len(xf) > 0:
            zeta_f = torch.ones(len(xf), 1)
            (e_min, e_max), (h_min, h_max) = _get_param_ranges()
            ef = torch.rand(len(xf), 1) * (e_max - e_min) + e_min
            hf = torch.rand(len(xf), 1) * (h_max - h_min) + h_min
            batch_pts = torch.cat([xf, yf, zeta_f, ef, hf], dim=1)
            pts_free_list.append(batch_pts)
            count += len(xf)
    
    pts_free = torch.cat(pts_free_list, dim=0)[:n]
    return pts_free

def sample_surface_residual_based(n, z_val, prev_pts, prev_residuals, constrain_load_patch=False, is_load_patch=False):
    """Sample surface points weighted by traction residual."""
    # Check if residuals are too small - fall back to uniform sampling
    if prev_residuals.sum() < 1e-12 or torch.isnan(prev_residuals).any():
        if constrain_load_patch and is_load_patch:
            return sample_top_load(n)
        elif constrain_load_patch and not is_load_patch:
            return sample_top_free(n)
        else:
            # General surface sampling
            x = torch.rand(n, 1) * config.Lx
            y = torch.rand(n, 1) * config.Ly
            z = torch.ones(n, 1) * z_val
            e_min, e_max = _get_e_range()
            e = torch.rand(n, 1) * (e_max - e_min) + e_min
            h_min, h_max = _get_h_range()
            h = torch.rand(n, 1) * (h_max - h_min) + h_min
            return torch.cat([x, y, z, e, h], dim=1)
    
    residual_probs = prev_residuals / prev_residuals.sum()
    residual_probs = residual_probs + 1e-10
    residual_probs = residual_probs / residual_probs.sum()
    indices = torch.multinomial(residual_probs, n, replacement=True)
    sampled_pts = prev_pts[indices]
    
    noise_scale = getattr(config, "SAMPLING_NOISE_SCALE", 0.05)
    noise_x = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * config.Lx
    noise_y = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * config.Ly
    (e_min, e_max), (h_min, h_max) = _get_param_ranges()
    noise_e = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * (e_max - e_min)
    noise_h = (torch.rand(n, 1) - 0.5) * 2 * noise_scale * (h_max - h_min)
    noise = torch.cat([noise_x, noise_y, torch.zeros(n, 1), noise_e, noise_h], dim=1)
    
    new_pts = sampled_pts + noise
    new_pts[:, 2] = z_val  # Fix zeta coordinate (0 or 1)
    
    # Clamp to domain
    new_pts[:, 0] = torch.clamp(new_pts[:, 0], 0, config.Lx)
    new_pts[:, 1] = torch.clamp(new_pts[:, 1], 0, config.Ly)
    new_pts[:, 3] = torch.clamp(new_pts[:, 3], e_min, e_max)
    
    # If constrained to load patch or free region
    if constrain_load_patch:
        if is_load_patch:
            # Clamp to load patch
            new_pts[:, 0] = torch.clamp(new_pts[:, 0], config.Lx/3, 2*config.Lx/3)
            new_pts[:, 1] = torch.clamp(new_pts[:, 1], config.Ly/3, 2*config.Ly/3)
        else:
            # Keep outside load patch - if inside, push to nearest edge
            for i in range(n):
                x, y = new_pts[i, 0].item(), new_pts[i, 1].item()
                if config.Lx/3 < x < 2*config.Lx/3 and config.Ly/3 < y < 2*config.Ly/3:
                    # Inside patch, push out to nearest boundary
                    dx_low = x - config.Lx/3
                    dx_high = 2*config.Lx/3 - x
                    dy_low = y - config.Ly/3
                    dy_high = 2*config.Ly/3 - y
                    min_dist = min(dx_low, dx_high, dy_low, dy_high)
                    if min_dist == dx_low:
                        new_pts[i, 0] = config.Lx/3 - 0.01
                    elif min_dist == dx_high:
                        new_pts[i, 0] = 2*config.Lx/3 + 0.01
                    elif min_dist == dy_low:
                        new_pts[i, 1] = config.Ly/3 - 0.01
                    else:
                        new_pts[i, 1] = 2*config.Ly/3 + 0.01
                    # Re-clamp
                    new_pts[i, 0] = torch.clamp(new_pts[i, 0], torch.tensor(0.0), torch.tensor(config.Lx))
                    new_pts[i, 1] = torch.clamp(new_pts[i, 1], torch.tensor(0.0), torch.tensor(config.Ly))
    
    return new_pts

def sample_top(n):
    # DEPRECATED: Use sample_top_load and sample_top_free separately
    n_load = n // 2
    n_free = n - n_load
    
    pts_load = sample_top_load(n_load)
    pts_free = sample_top_free(n_free)
    
    return pts_load, pts_free

def sample_interface(n, z_val):
    # z = z_val
    x = torch.rand(n, 1) * config.Lx
    y = torch.rand(n, 1) * config.Ly
    z = torch.ones(n, 1) * z_val
    e = torch.rand(n, 1) * (config.E_RANGE[1] - config.E_RANGE[0]) + config.E_RANGE[0]
    return torch.cat([x, y, z, e], dim=1)

def sample_bottom(n):
    """Sample points on bottom surface (zeta=0)."""
    x_bot = torch.rand(n, 1) * config.Lx
    y_bot = torch.rand(n, 1) * config.Ly
    zeta_bot = torch.zeros(n, 1)
    (e_min, e_max), (h_min, h_max) = _get_param_ranges()
    e_bot = torch.rand(n, 1) * (e_max - e_min) + e_min
    h_bot = torch.rand(n, 1) * (h_max - h_min) + h_min
    return torch.cat([x_bot, y_bot, zeta_bot, e_bot, h_bot], dim=1)

def get_data(prev_data=None, residuals=None):
    """Generate collocation points with optional residual-based sampling.
    
    Args:
        prev_data: Previous training data dict (optional)
        residuals: Dict of residuals for each data type (optional)
                  Keys: 'interior', 'sides', 'top_load', 'top_free', 'bottom'
                  Values: Tensor of residual magnitudes
    
    Returns:
        Dictionary of training data
    """
    z_min, z_max = config.Layer_Interfaces[0], config.Layer_Interfaces[1]
    
    # Decide whether to use residual-based sampling (50% uniform, 50% residual-based)
    use_residual = (prev_data is not None and residuals is not None)
    
    n_patch = int(config.N_INTERIOR * config.UNDER_PATCH_FRACTION)
    if n_patch < 0:
        n_patch = 0
    if n_patch > config.N_INTERIOR:
        n_patch = config.N_INTERIOR
    n_interior = config.N_INTERIOR - n_patch
    
    if use_residual:
        n_uniform = n_interior // 2
        n_residual = n_interior - n_uniform
        
        # Interior: half uniform, half residual-based
        interior_uniform = sample_domain(n_uniform, z_min, z_max)
        interior_residual = sample_domain_residual_based(
            n_residual, z_min, z_max,
            prev_data['interior'][0], residuals['interior']
        )
        interior = torch.cat([interior_uniform, interior_residual], dim=0)
        if n_patch > 0:
            interior_patch = sample_domain_under_patch(n_patch, z_min, z_max)
            interior = torch.cat([interior, interior_patch], dim=0)
        
        # BC Sides: half uniform, half residual-based
        n_uniform_bc = config.N_SIDES // 2
        n_residual_bc = config.N_SIDES - n_uniform_bc
        bc_uniform = sample_boundaries(n_uniform_bc, z_min, z_max)
        bc_residual = sample_boundaries_residual_based(
            n_residual_bc, z_min, z_max,
            prev_data['sides'][0], residuals['sides']
        )
        bc_sides = torch.cat([bc_uniform, bc_residual], dim=0)
        
        # Top Load: half uniform, half residual-based
        n_uniform_load = config.N_TOP_LOAD // 2
        n_residual_load = config.N_TOP_LOAD - n_uniform_load
        load_uniform = sample_top_load(n_uniform_load)
        load_residual = sample_surface_residual_based(
            n_residual_load, 1.0, # zeta=1
            prev_data['top_load'], residuals['top_load'],
            constrain_load_patch=True, is_load_patch=True
        )
        top_load = torch.cat([load_uniform, load_residual], dim=0)
        
        # Top Free: half uniform, half residual-based
        n_uniform_free = config.N_TOP_FREE // 2
        n_residual_free = config.N_TOP_FREE - n_uniform_free
        free_uniform = sample_top_free(n_uniform_free)
        free_residual = sample_surface_residual_based(
            n_residual_free, 1.0, # zeta=1
            prev_data['top_free'], residuals['top_free'],
            constrain_load_patch=True, is_load_patch=False
        )
        top_free = torch.cat([free_uniform, free_residual], dim=0)
        
        # Bottom: half uniform, half residual-based
        n_uniform_bot = config.N_BOTTOM // 2
        n_residual_bot = config.N_BOTTOM - n_uniform_bot
        bot_uniform = sample_bottom(n_uniform_bot)
        bot_residual = sample_surface_residual_based(
            n_residual_bot, 0.0, # zeta=0
            prev_data['bottom'], residuals['bottom']
        )
        bot_free = torch.cat([bot_uniform, bot_residual], dim=0)
        
    else:
        # Uniform sampling (initial or when no residuals provided)
        interior_uniform = sample_domain(n_interior, z_min, z_max)
        if n_patch > 0:
            interior_patch = sample_domain_under_patch(n_patch, z_min, z_max)
            interior = torch.cat([interior_uniform, interior_patch], dim=0)
        else:
            interior = interior_uniform
        bc_sides = sample_boundaries(config.N_SIDES, z_min, z_max)
        top_load = sample_top_load(config.N_TOP_LOAD)
        top_free = sample_top_free(config.N_TOP_FREE)
        bot_free = sample_bottom(config.N_BOTTOM)
    
    return {
        'interior': [interior],
        'sides': [bc_sides],
        'top_load': top_load,
        'top_free': top_free,
        'bottom': bot_free
    }
