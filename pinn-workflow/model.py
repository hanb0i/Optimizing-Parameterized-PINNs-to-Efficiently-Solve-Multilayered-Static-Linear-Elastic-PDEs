import torch
import torch.nn as nn
import numpy as np
import pinn_config as config


class FourierFeatures(nn.Module):
    def __init__(self, input_dim, mapping_size, scale):
        super().__init__()
        self.B = nn.Parameter(torch.randn(input_dim, mapping_size) * scale, requires_grad=False)

    def forward(self, x):
        # x: (N, 3)
        x_norm = x.clone()
        x_norm[:, 2] = x_norm[:, 2]  # z is already normalized to [0, 1]
        x_proj = (2.0 * np.pi * x_norm) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class LayerNet(nn.Module):
    def __init__(self, hidden_layers=2, hidden_units=32, activation=nn.Tanh(),
                 fourier_dim=0, fourier_scale=1.0):
        super().__init__()
        layers = []
        # Input: x, y, z (and E, thickness, restitution, friction, v0 as extra channels)
        self.fourier = FourierFeatures(3, fourier_dim, fourier_scale) if fourier_dim > 0 else None
        
        # Base input dimension
        # If fourier: [sin(..), cos(..)] (size 2*fourier_dim) + 5 normalized params
        # Else: x, y, z_hat + 5 normalized params (size 8)
        base_dim = (2 * fourier_dim + 5) if fourier_dim > 0 else 8
        
        # We add 3 extra physics-informed features: (H/t), (H/t)^2, (H/t)^3
        # These help the network resolve the 1/t^3 scaling naturally from the PDE data.
        current_dim = base_dim + 3
        
        layers.append(nn.Linear(current_dim, hidden_units))
        layers.append(activation)
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(activation)
            
        # Output: u_x, u_y, u_z (3 dims)
        layers.append(nn.Linear(hidden_units, 3))
        
        self.net = nn.Sequential(*layers)
        
        # Weight initialization
        self._init_weights()
        
    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        # x shape: (N, 8) -> [x, y, z, E, thickness, restitution, friction, v0]
        
        # Use torch.cat to preserve gradient flow
        # x_scaled = [x, y, z*10, E_norm]
        
        x_coord = x[:, 0:1]
        y_coord = x[:, 1:2]
        z_coord = x[:, 2:3]
        e_param = x[:, 3:4]
        t_param = x[:, 4:5]
        r_param = x[:, 5:6]
        mu_param = x[:, 6:7]
        v0_param = x[:, 7:8]
        
        e_min, e_max = config.E_RANGE
        e_span = (e_max - e_min) if (e_max - e_min) != 0 else 1.0
        e_norm = (e_param - e_min) / e_span

        t_min, t_max = config.THICKNESS_RANGE
        t_span = (t_max - t_min) if (t_max - t_min) != 0 else 1.0
        t_norm = (t_param - t_min) / t_span

        r_min, r_max = config.RESTITUTION_RANGE
        r_span = (r_max - r_min) if (r_max - r_min) != 0 else 1.0
        r_norm = (r_param - r_min) / r_span

        mu_min, mu_max = config.FRICTION_RANGE
        mu_span = (mu_max - mu_min) if (mu_max - mu_min) != 0 else 1.0
        mu_norm = (mu_param - mu_min) / mu_span

        v0_min, v0_max = config.IMPACT_VELOCITY_RANGE
        v0_span = (v0_max - v0_min) if (v0_max - v0_min) != 0 else 1.0
        v0_norm = (v0_param - v0_min) / v0_span

        t_safe = torch.clamp(t_param, min=1e-6)
        z_hat = z_coord / t_safe
        
        # Physics-Informed Features: scaling relative to baseline H
        # This gives the network 'basis functions' that match the expected physics (bending ~ 1/t^3)
        # using H=0.1 as reference to keep features ~ O(1)
        # However, for t=0.05, ratio=2, ratio^3=8. This is large but not insane.
        # But if the network is initialized with small weights, an input of 8 might effectively saturate units or dominate gradients.
        
        h_ref = getattr(config, 'H', 0.1)
        ratio = h_ref / t_safe 
        
        # Raw physics scaling features as requested.
        feat_inv1 = ratio
        feat_inv2 = ratio ** 2
        feat_inv3 = ratio ** 3
        
        extra_feats = torch.cat([feat_inv1, feat_inv2, feat_inv3], dim=1)
        
        if self.fourier is not None:
            xyz = torch.cat([x_coord, y_coord, z_hat], dim=1)
            ff = self.fourier(xyz)
            x_scaled = torch.cat([ff, e_norm, t_norm, r_norm, mu_norm, v0_norm, extra_feats], dim=1)
        else:
            x_scaled = torch.cat([x_coord, y_coord, z_hat, e_norm, t_norm, r_norm, mu_norm, v0_norm, extra_feats], dim=1)
        
        u_raw = self.net(x_scaled)
        
        if config.USE_HARD_SIDE_BC:
            # Hard Constraint for Clamped Sides (x=0, x=1, y=0, y=1)
            # Mask M(x,y) = x(1-x)y(1-y)
            # Normalized so max value is ~1 (at center x=0.5, y=0.5, val=0.0625 -> *16)
            x_c = x[:, 0:1]
            y_c = x[:, 1:2]
            
            # We assume domain is [0,1]x[0,1] based on config.
            # If config changed Lx, Ly, this should be dynamic, but for now hardcoded matches config.
            mask = x_c * (1.0 - x_c) * y_c * (1.0 - y_c) * 16.0
            u_raw = u_raw * mask
        
        return u_raw

class MultiLayerPINN(nn.Module):
    def __init__(self):
        super().__init__()
        # Single network for homogeneous material
        self.layer = LayerNet(
            hidden_layers=getattr(config, 'LAYERS', 4),
            hidden_units=getattr(config, 'NEURONS', 64),
            fourier_dim=config.FOURIER_DIM, 
            fourier_scale=config.FOURIER_SCALE
        )
        
    def forward(self, x, layer_idx=0):
        # x: (N, 8) -> [x, y, z, E, thickness, restitution, friction, v0]
        
        # 1. Compute the normalized potential/shape 'v'
        v = self.layer(x)
        
        # Return v directly. Physics layer will handle u = v / E
        return v

    def predict_all(self, x):
        # Consistent prediction
        return self.forward(x)

    def set_hard_bc(self, use_hard):
        config.USE_HARD_SIDE_BC = bool(use_hard)
