import torch
import torch.nn as nn
import pinn_config as config


def _as_layered_input(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize inputs to the 2-layer layout:
      [x, y, z, E1, t1, E2, t2, r, mu, v0]
    """
    if x.shape[1] == 10:
        return x
    if x.shape[1] == 12:
        # [x,y,z,E1,E2,E3,t1,t2,t3,r,mu,v0] -> drop E3/t3.
        return torch.cat(
            [
                x[:, 0:3],
                x[:, 3:4],
                x[:, 6:7],
                x[:, 4:5],
                x[:, 7:8],
                x[:, 9:12],
            ],
            dim=1,
        )
    if x.shape[1] == 8:
        # [x,y,z,E,t,r,mu,v0] -> split t and E across 2 layers.
        t = x[:, 4:5]
        t_half = 0.5 * t
        e = x[:, 3:4]
        return torch.cat([x[:, 0:3], e, t_half, e, t_half, x[:, 5:8]], dim=1)
    if x.shape[1] == 4:
        # [x,y,z,E] -> assume total thickness = H and default impact params.
        device = x.device
        t_half = torch.full((x.shape[0], 1), float(getattr(config, "H", 0.1)) / 2.0, device=device)
        e = x[:, 3:4]
        r_ref = float(getattr(config, "RESTITUTION_REF", 0.5))
        mu_ref = float(getattr(config, "FRICTION_REF", 0.3))
        v0_ref = float(getattr(config, "IMPACT_VELOCITY_REF", 1.0))
        r = torch.full((x.shape[0], 1), r_ref, device=device)
        mu = torch.full((x.shape[0], 1), mu_ref, device=device)
        v0 = torch.full((x.shape[0], 1), v0_ref, device=device)
        return torch.cat([x[:, 0:3], e, t_half, e, t_half, r, mu, v0], dim=1)
    raise ValueError(f"Unexpected input dimension {x.shape[1]} (expected 4/8/10/12).")


class LayerNet(nn.Module):
    def __init__(self, hidden_layers=2, hidden_units=32, activation=nn.Tanh()):
        super().__init__()
        self.fourier_dim = getattr(config, "FOURIER_DIM", 0)
        self.fourier_scale = getattr(config, "FOURIER_SCALE", 1.0)
        self.use_mixed = bool(getattr(config, "USE_MIXED_FORMULATION", False))

        base_dim = 15
        if self.fourier_dim > 0:
            B = torch.randn(self.fourier_dim, 3) * self.fourier_scale
            self.register_buffer("B", B)
            current_dim = base_dim + 2 * self.fourier_dim
        else:
            current_dim = base_dim

        layers = [nn.Linear(current_dim, hidden_units), activation]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(activation)

        out_dim = 9 if self.use_mixed else 3
        layers.append(nn.Linear(hidden_units, out_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = _as_layered_input(x)
        x_coord = x[:, 0:1]
        y_coord = x[:, 1:2]
        z_coord = x[:, 2:3]
        e1 = x[:, 3:4]
        t1 = torch.clamp(x[:, 4:5], min=1e-6)
        e2 = x[:, 5:6]
        t2 = torch.clamp(x[:, 6:7], min=1e-6)
        r_param = x[:, 7:8]
        mu_param = x[:, 8:9]
        v0_param = x[:, 9:10]

        t_total = torch.clamp(t1 + t2, min=1e-6)
        z_hat = z_coord / t_total
        d_hat = (z_coord - t1) / t_total
        beta = float(getattr(config, "INTERFACE_FEATURE_BETA", 200.0))
        s_hat = torch.sigmoid(beta * d_hat)

        e_min, e_max = config.E_RANGE
        e_span = float(e_max - e_min) if float(e_max - e_min) != 0.0 else 1.0
        e1_n = (e1 - float(e_min)) / e_span
        e2_n = (e2 - float(e_min)) / e_span

        t_min, t_max = config.THICKNESS_RANGE
        t_span = float(t_max - t_min) if float(t_max - t_min) != 0.0 else 1.0
        t1_n = (t1 - float(t_min)) / t_span
        t2_n = (t2 - float(t_min)) / t_span

        r_min, r_max = config.RESTITUTION_RANGE
        r_span = float(r_max - r_min) if float(r_max - r_min) != 0.0 else 1.0
        r_n = (r_param - float(r_min)) / r_span

        mu_min, mu_max = config.FRICTION_RANGE
        mu_span = float(mu_max - mu_min) if float(mu_max - mu_min) != 0.0 else 1.0
        mu_n = (mu_param - float(mu_min)) / mu_span

        v0_min, v0_max = config.IMPACT_VELOCITY_RANGE
        v0_span = float(v0_max - v0_min) if float(v0_max - v0_min) != 0.0 else 1.0
        v0_n = (v0_param - float(v0_min)) / v0_span

        h_ref = float(getattr(config, "H", 0.1))
        inv1 = h_ref / t_total
        inv2 = inv1 ** 2
        inv3 = inv1 ** 3

        base_features = [
            x_coord,
            y_coord,
            z_hat,
            e1_n,
            t1_n,
            e2_n,
            t2_n,
            r_n,
            mu_n,
            v0_n,
            d_hat,
            s_hat,
            inv1,
            inv2,
            inv3,
        ]

        if self.fourier_dim > 0:
            spatial_input = torch.cat([x_coord, y_coord, z_hat], dim=1)
            x_proj = (2.0 * torch.pi * spatial_input) @ self.B.T
            fourier_features = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=1)
            return torch.cat(base_features + [fourier_features], dim=1)

        return torch.cat(base_features, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = _as_layered_input(x)
        feats = self.forward_features(x_in)
        y_raw = self.net(feats)

        e1 = x_in[:, 3:4]
        t1 = torch.clamp(x_in[:, 4:5], min=1e-6)
        e2 = x_in[:, 5:6]
        t2 = torch.clamp(x_in[:, 6:7], min=1e-6)
        t_total = torch.clamp(t1 + t2, min=1e-6)
        e_eff = (e1 * t1 + e2 * t2) / t_total
        e_eff = torch.clamp(e_eff, min=1e-6)

        e_pow = float(getattr(config, "E_COMPLIANCE_POWER", 1.0))
        alpha = float(getattr(config, "THICKNESS_COMPLIANCE_ALPHA", 0.0))
        h_ref = float(getattr(config, "H", 0.1))
        t_scale = (h_ref / t_total) ** alpha
        scale_pinn = float(getattr(config, "OUTPUT_SCALE_Z", 10.0))
        scale = (1.0 / (e_eff ** e_pow)) * t_scale * scale_pinn

        if self.use_mixed and y_raw.shape[1] > 3:
            u_raw = y_raw[:, 0:3] * scale
            rest = y_raw[:, 3:]
        else:
            u_raw = y_raw * scale
            rest = None

        if bool(getattr(config, "USE_HARD_SIDE_BC", False)):
            x_coord = x_in[:, 0:1]
            y_coord = x_in[:, 1:2]
            mask_side = x_coord * (1.0 - x_coord) * y_coord * (1.0 - y_coord) * 16.0
            u_raw = u_raw * mask_side

        if rest is not None:
            return torch.cat([u_raw, rest], dim=1)
        return u_raw


class MultiLayerPINN(nn.Module):
    def __init__(self):
        super().__init__()
        lay = int(getattr(config, "LAYERS", 4))
        hid = int(getattr(config, "NEURONS", 64))
        self.layer = LayerNet(hidden_layers=lay, hidden_units=hid)

    def forward(self, x, layer_idx=None):
        return self.layer(x)

    def predict_all(self, x):
        return self.forward(x)

    def set_hard_bc(self, use_hard):
        config.USE_HARD_SIDE_BC = bool(use_hard)
