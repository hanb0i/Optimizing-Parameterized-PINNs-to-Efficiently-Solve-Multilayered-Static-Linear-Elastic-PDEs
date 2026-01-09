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
        # Normalize coordinates to [0, 1] based on config geometry.
        x_norm = torch.cat(
            [
                x[:, 0:1] / config.Lx,
                x[:, 1:2] / config.Ly,
                x[:, 2:3] / config.H,
            ],
            dim=1,
        )
        
        # x_proj: (N, mapping_size)
        x_proj = (2.0 * np.pi * x_norm) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class LayerNet(nn.Module):
    def __init__(self, hidden_layers=2, hidden_units=32, activation=nn.Tanh(), 
                 fourier_dim=0, fourier_scale=1.0):
        super().__init__()
        layers = []
        # Input: x, y, z (3 coords)
        input_dim = 3
        
        if fourier_dim > 0:
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
        # Normalize coordinates to [0, 1] for conditioning.
        x_scaled = torch.cat(
            [
                x[:, 0:1] / config.Lx,
                x[:, 1:2] / config.Ly,
                x[:, 2:3] / config.H,
            ],
            dim=1,
        )
        
        u_raw = self.net(x_scaled)
        
        # Hard Constraint for Clamped Sides (x=0, x=Lx, y=0, y=Ly)
        # Mask M(x,y) = x(1-x)y(1-y)
        # Normalized so max value is ~1 (at center x=0.5, y=0.5, val=0.0625 -> *16)
        x_c = x[:, 0:1] / config.Lx
        y_c = x[:, 1:2] / config.Ly
        mask = x_c * (1.0 - x_c) * y_c * (1.0 - y_c) * 16.0
        
        # Apply mask
        return u_raw * mask

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
