
import torch
import numpy as np
import pinn_config as config

def sample_domain(n, z_min, z_max):
    # Uniform sampling
    x = torch.rand(n, 1) * config.Lx
    y = torch.rand(n, 1) * config.Ly
    z = torch.rand(n, 1) * (z_max - z_min) + z_min
    return torch.cat([x, y, z], dim=1)

def sample_boundaries(n, z_min, z_max):
    # 4 Side faces: x=0, x=Lx, y=0, y=Ly
    # Split n among 4 faces
    n_face = n // 4
    
    # x=0
    y1 = torch.rand(n_face, 1) * config.Ly
    z1 = torch.rand(n_face, 1) * (z_max - z_min) + z_min
    x1 = torch.zeros(n_face, 1)
    p1 = torch.cat([x1, y1, z1], dim=1)
    
    # x=Lx
    y2 = torch.rand(n_face, 1) * config.Ly
    z2 = torch.rand(n_face, 1) * (z_max - z_min) + z_min
    x2 = torch.ones(n_face, 1) * config.Lx
    p2 = torch.cat([x2, y2, z2], dim=1)
    
    # y=0
    x3 = torch.rand(n_face, 1) * config.Lx
    z3 = torch.rand(n_face, 1) * (z_max - z_min) + z_min
    y3 = torch.zeros(n_face, 1)
    p3 = torch.cat([x3, y3, z3], dim=1)
    
    # y=Ly
    x4 = torch.rand(n_face, 1) * config.Lx
    z4 = torch.rand(n_face, 1) * (z_max - z_min) + z_min
    y4 = torch.ones(n_face, 1) * config.Ly
    p4 = torch.cat([x4, y4, z4], dim=1)
    
    return torch.cat([p1, p2, p3, p4], dim=0)

def sample_top(n):
    # z=H
    # Determine Loaded Patch vs Free
    # Loaded: Lx/3 < x < 2Lx/3 AND Ly/3 < y < 2Ly/3
    
    # We'll sample uniform and classify, or explicit sample?
    # Better to explicitly sample both sets to ensure coverage
    
    n_load = n // 2
    n_free = n - n_load
    
    # Loaded Patch
    xl = torch.rand(n_load, 1) * (config.Lx/3) + config.Lx/3
    yl = torch.rand(n_load, 1) * (config.Ly/3) + config.Ly/3
    zl = torch.ones(n_load, 1) * config.H
    pts_load = torch.cat([xl, yl, zl], dim=1)
    
    # Free Top (rest of domain)
    # Simple rejection sampling or composition of rectangles
    # Total area = Lx*Ly. Patch area = 1/9 Lx*Ly.
    # Rejection is easy.
    pts_free_list = []
    count = 0
    while count < n_free:
        batch = 1000
        x = torch.rand(batch, 1) * config.Lx
        y = torch.rand(batch, 1) * config.Ly
        
        in_patch = (x > config.Lx/3) & (x < 2*config.Lx/3) & \
                   (y > config.Ly/3) & (y < 2*config.Ly/3)
        
        mask_free = ~in_patch.squeeze()
        xf, yf = x[mask_free], y[mask_free]
        if len(xf) > 0:
            zf = torch.ones(len(xf), 1) * config.H
            batch_pts = torch.cat([xf, yf, zf], dim=1)
            pts_free_list.append(batch_pts)
            count += len(xf)
        
    pts_free = torch.cat(pts_free_list, dim=0)[:n_free]
    
    return pts_load, pts_free

def sample_interface(n, z_val):
    # z = z_val
    x = torch.rand(n, 1) * config.Lx
    y = torch.rand(n, 1) * config.Ly
    z = torch.ones(n, 1) * z_val
    return torch.cat([x, y, z], dim=1)

def get_data():
    # Interior points for single layer
    interior = sample_domain(config.N_INTERIOR, config.Layer_Interfaces[0], config.Layer_Interfaces[1])
    
    # Clamped sides for single layer
    bc_sides = sample_boundaries(config.N_BOUNDARY, config.Layer_Interfaces[0], config.Layer_Interfaces[1])
    
    # Top Surface
    top_load, top_free = sample_top(config.N_BOUNDARY)
    
    # Bottom Surface - Free
    # z=0
    x_bot = torch.rand(config.N_BOUNDARY, 1) * config.Lx
    y_bot = torch.rand(config.N_BOUNDARY, 1) * config.Ly
    z_bot = torch.zeros(config.N_BOUNDARY, 1)
    bot_free = torch.cat([x_bot, y_bot, z_bot], dim=1)
    
    return {
        'interior': [interior],
        'sides': [bc_sides],
        'top_load': top_load,
        'top_free': top_free,
        'bottom': bot_free
    }
