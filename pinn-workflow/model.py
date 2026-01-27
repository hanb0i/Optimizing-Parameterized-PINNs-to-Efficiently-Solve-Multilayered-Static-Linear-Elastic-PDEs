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
        # Normalize z coordinate by the configured thickness so z is O(1) like x,y.
        x_norm = x.clone()
        z_scale = 1.0 / max(float(config.H), 1e-12)
        x_norm[:, 2] = x_norm[:, 2] * z_scale
        
        # x_proj: (N, mapping_size)
        x_proj = (2. * np.pi * x_norm) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class LayerNet(nn.Module):
    def __init__(self, hidden_layers=2, hidden_units=32, activation=nn.Tanh(), 
                 fourier_dim=0, fourier_scale=1.0):
        super().__init__()
        layers = []
        # Input: x, y, z (3 coords)
        input_dim = 3
        self.use_fourier = fourier_dim > 0
        
        if self.use_fourier:
            layers.append(FourierFeatures(input_dim, fourier_dim, fourier_scale))
            # FourierFeatures output is 2 * fourier_dim
            current_dim = 2 * fourier_dim
        else:
            current_dim = input_dim
        
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
        # x shape: (N, 3)
        if self.use_fourier:
            x_in = x
        else:
            # Scale z coordinate by thickness to match x,y range [0,1]
            z_scale = 1.0 / max(float(config.H), 1e-12)
            x_in = torch.cat([x[:, 0:1], x[:, 1:2], x[:, 2:3] * z_scale], dim=1)
        
        u_raw = self.net(x_in)
        
        if config.USE_HARD_SIDE_BC:
            # Hard Constraint for Clamped Sides (x=0, x=1, y=0, y=1)
            # Mask M(x,y) = x(1-x)y(1-y)
            # Normalized so max value is ~1 (at center x=0.5, y=0.5, val=0.0625 -> *16)
            x_c = x[:, 0:1] / max(float(config.Lx), 1e-12)
            y_c = x[:, 1:2] / max(float(config.Ly), 1e-12)
            mask = x_c * (1.0 - x_c) * y_c * (1.0 - y_c) * 16.0
            return u_raw * mask
        
        return u_raw

class MultiLayerPINN(nn.Module):
    def __init__(self):
        super().__init__()
        # Single network for homogeneous material
        # Use parameters from config
        self.layer = LayerNet(fourier_dim=config.FOURIER_DIM, fourier_scale=config.FOURIER_SCALE)
        
    def forward(self, x, layer_idx=0):
        # layer_idx kept for compatibility but not used
        return self.layer(x)

    def predict_all(self, x):
        # Direct prediction for single layer
        return self.layer(x)

    def set_hard_bc(self, use_hard):
        config.USE_HARD_SIDE_BC = bool(use_hard)
