import torch
import torch.nn as nn
import pinn_config as config

class LayerNet(nn.Module):
<<<<<<< HEAD
<<<<<<< HEAD
    def __init__(self, hidden_layers=3, hidden_units=64, activation=nn.Tanh(), 
=======
    def __init__(self, hidden_layers=2, hidden_units=32, activation=nn.Tanh(), 
>>>>>>> a204439ef0cee6b426c4e683743f2eee33c9b01a
                 fourier_dim=0, fourier_scale=1.0):
=======
    def __init__(self, hidden_layers=2, hidden_units=32, activation=nn.Tanh()):
>>>>>>> 3176abcf323e43483e790d268adaa1838f1907f2
        super().__init__()
        layers = []
        # Input: x, y, z (3 coords)
        input_dim = 3
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
        # Scale z coordinate by 10 to match x,y range [0,1]
        x_scaled = torch.cat([x[:, 0:1], x[:, 1:2], x[:, 2:3] * 10.0], dim=1)
        
        u_raw = self.net(x_scaled)
        
        # Hard Constraint for Clamped Sides (x=0, x=1, y=0, y=1)
        # Mask M(x,y) = x(1-x)y(1-y)
        # Normalized so max value is ~1 (at center x=0.5, y=0.5, val=0.0625 -> *16)
        x_c = x[:, 0:1]
        y_c = x[:, 1:2]
        
<<<<<<< HEAD
        # We assume domain is [0,1]x[0,1] based on config.
        # If config changed Lx, Ly, this should be dynamic, but for now hardcoded matches config.
        mask = x_c * (1.0 - x_c) * y_c * (1.0 - y_c) * 4.0
=======
        mask = x_c * (1.0 - x_c) * y_c * (1.0 - y_c) * 16.0
>>>>>>> 3176abcf323e43483e790d268adaa1838f1907f2
        
        # Apply mask
        return u_raw * mask * config.OUTPUT_SCALE

class MultiLayerPINN(nn.Module):
    def __init__(self):
        super().__init__()
        # Single network for homogeneous material
        self.layer = LayerNet()
        
    def forward(self, x, layer_idx=0):
        # layer_idx kept for compatibility but not used
        return self.layer(x)

    def predict_all(self, x):
        # Direct prediction for single layer
        return self.layer(x)
