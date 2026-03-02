import torch
import torch.nn as nn
import numpy as np
import pinn_config as config


class LayerNet(nn.Module):
    def __init__(self, hidden_layers=2, hidden_units=32, activation=nn.Tanh()):
        super().__init__()
        layers = []
        # Fourier Features
        self.fourier_dim = getattr(config, 'FOURIER_DIM', 0)
        self.fourier_scale = getattr(config, 'FOURIER_SCALE', 1.0)
        
        # Spatial inputs to Fourier encode: x, y, z_hat (3 dims)
        fourier_input_dim = 3 
        
        if self.fourier_dim > 0:
            B = torch.randn(self.fourier_dim, fourier_input_dim) * self.fourier_scale
            self.register_buffer('B', B)
            # Total input dim: 
            #   Coords (3): x, y, z_hat
            #   Physical summaries (2): e_eq_n, t_tot_n
            #   Impact params (3): r_n, mu_n, v0_n
            #   Fourier (2*dim)
            current_dim = 8 + 2 * self.fourier_dim
        else:
            current_dim = 8
        
        layers.append(nn.Linear(current_dim, hidden_units))
        layers.append(activation)
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(activation)
            
        layers.append(nn.Linear(hidden_units, 3))
        self.net = nn.Sequential(*layers)
        self._init_weights()
        
    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        # x is (N, 12): [x, y, z, E1, E2, E3, t1, t2, t3, r, mu, v0]
        x_coord = x[:, 0:1]
        y_coord = x[:, 1:2]
        z_coord = x[:, 2:3]
        e1, e2, e3 = x[:, 3:4], x[:, 4:5], x[:, 5:6]
        t1, t2, t3 = x[:, 6:7], x[:, 7:8], x[:, 8:9]
        r_param = x[:, 9:10]
        mu_param = x[:, 10:11]
        v0_param = x[:, 11:12]
        
        # Bending stiffness (EI)
        t_tot = t1 + t2 + t3
        z1 = t1 / 2.0
        z2 = t1 + t2 / 2.0
        z3 = t1 + t2 + t3 / 2.0
        
        num = e1 * t1 * z1 + e2 * t2 * z2 + e3 * t3 * z3
        den = e1 * t1 + e2 * t2 + e3 * t3
        z_bar = num / den
        
        I1 = (t1**3) / 12.0
        I2 = (t2**3) / 12.0
        I3 = (t3**3) / 12.0
        
        d1 = z1 - z_bar
        d2 = z2 - z_bar
        d3 = z3 - z_bar
        
        EI = e1 * (I1 + t1 * d1**2) + e2 * (I2 + t2 * d2**2) + e3 * (I3 + t3 * d3**2)
        
        # Equivalent homogeneous E
        t_safe = torch.clamp(t_tot, min=1e-6)
        e_eq = 12.0 * EI / (t_safe**3)
        e_eq = torch.clamp(e_eq, min=1e-6)
        
        # Normalization for E_eq and t_total
        e_min, e_max = config.E_RANGE
        e_span = (e_max - e_min) if (e_max - e_min) != 0 else 1.0
        e_eq_n = (e_eq - e_min) / e_span

        t_min, t_max = config.THICKNESS_RANGE
        t_span = (t_max*3 - t_min*3) if (t_max - t_min) != 0 else 1.0 # Max total
        t_tot_n = (t_tot - t_min*3) / t_span

        r_min, r_max = config.RESTITUTION_RANGE
        r_span = (r_max - r_min) if (r_max - r_min) != 0 else 1.0
        r_n = (r_param - r_min) / r_span

        mu_min, mu_max = config.FRICTION_RANGE
        mu_span = (mu_max - mu_min) if (mu_max - mu_min) != 0 else 1.0
        mu_n = (mu_param - mu_min) / mu_span

        v0_min, v0_max = config.IMPACT_VELOCITY_RANGE
        v0_span = (v0_max - v0_min) if (v0_max - v0_min) != 0 else 1.0
        v0_n = (v0_param - v0_min) / v0_span

        z_hat = z_coord / t_safe
        
        h_ref = getattr(config, 'H', 0.1)
        
        # Base input to net: 8 dimensions
        base_features = [
            x_coord, y_coord, z_hat, 
            e_eq_n, t_tot_n, 
            r_n, mu_n, v0_n
        ]
        
        if self.fourier_dim > 0:
            spatial_input = torch.cat([x_coord, y_coord, z_hat], dim=1)
            x_proj = (2.0 * torch.pi * spatial_input) @ self.B.T
            fourier_features = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=1)
            inp = torch.cat(base_features + [fourier_features], dim=1)
        else:
            inp = torch.cat(base_features, dim=1)
            
        u_raw = self.net(inp)
        
        # Power scaling: u = v / E_eq^p * (H_ref / t_tot)^alpha
        e_pow = getattr(config, 'E_COMPLIANCE_POWER', 1.0)
        alpha = getattr(config, 'THICKNESS_COMPLIANCE_ALPHA', 0.0)
        t_scale = (h_ref / t_safe) ** alpha
        
        scale_pinn = getattr(config, 'OUTPUT_SCALE_Z', 10.0)
        u_scaled = (u_raw / (e_eq ** e_pow)) * t_scale * scale_pinn
            
        if config.USE_HARD_SIDE_BC:
            mask_side = x_coord * (1.0 - x_coord) * y_coord * (1.0 - y_coord) * 16.0
            u_scaled = u_scaled * mask_side
        
        return u_scaled

class MultiLayerPINN(nn.Module):
    def __init__(self):
        super().__init__()
        lay = getattr(config, 'LAYERS', 4)
        hid = getattr(config, 'NEURONS', 64)
        self.layer1 = LayerNet(hidden_layers=lay, hidden_units=hid)
        self.layer2 = LayerNet(hidden_layers=lay, hidden_units=hid)
        self.layer3 = LayerNet(hidden_layers=lay, hidden_units=hid)
        
    def forward(self, x, layer_idx=0):
        if layer_idx == 1:
            return self.layer1(x)
        elif layer_idx == 2:
            return self.layer2(x)
        elif layer_idx == 3:
            return self.layer3(x)
            
        z_coord = x[:, 2:3]
        t1, t2 = x[:, 6:7], x[:, 7:8]
        
        m1 = (z_coord < t1).float()
        m2 = ((z_coord >= t1) & (z_coord < t1 + t2)).float()
        m3 = (z_coord >= t1 + t2).float()
        
        u1 = self.layer1(x)
        u2 = self.layer2(x)
        u3 = self.layer3(x)
        
        return m1 * u1 + m2 * u2 + m3 * u3

    def predict_all(self, x):
        return self.forward(x)

    def set_hard_bc(self, use_hard):
        config.USE_HARD_SIDE_BC = bool(use_hard)
