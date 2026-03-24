import torch
import torch.nn as nn
import pinn_config as config


def _adapt_first_layer_weight(src_w, tgt_w):
    if src_w.shape[0] != tgt_w.shape[0]:
        return None

    adapted = torch.zeros_like(tgt_w)

    if src_w.shape[1] == 12 and tgt_w.shape[1] >= 12:
        adapted[:, :12] = src_w[:, :12]
        return adapted

    # Legacy semantic layout:
    # [x, y, z_hat, E, t, r, mu, v0, extra1, extra2, extra3]
    # Current semantic layout before thickness parameterization:
    # [x, y, z_hat, E1, E2, r, mu, v0, sd, |sd|, soft]
    # New semantic layout:
    # [x, y, z_hat, E1, E2, t, r, mu, v0, sd, |sd|, soft, zeta^3, bend]
    if src_w.shape[1] == 11 and tgt_w.shape[1] >= 14:
        adapted[:, 0:3] = src_w[:, 0:3]
        adapted[:, 3:5] = src_w[:, 3:5]
        adapted[:, 6:9] = src_w[:, 5:8]
        adapted[:, 9:12] = src_w[:, 8:11]
        return adapted

    if src_w.shape[1] == 11:
        adapted[:, 0:3] = src_w[:, 0:3]
        adapted[:, 3:4] = 0.5 * src_w[:, 3:4]
        adapted[:, 4:5] = 0.5 * src_w[:, 3:4]
        adapted[:, 5:8] = src_w[:, 5:8]
        return adapted

    if src_w.shape[1] == 10 and tgt_w.shape[1] >= 14:
        adapted[:, 0:3] = src_w[:, 0:3]
        adapted[:, 3:4] = 0.5 * src_w[:, 3:4]
        adapted[:, 4:5] = 0.5 * src_w[:, 3:4]
        adapted[:, 6:8] = src_w[:, 5:7]
        return adapted

    if src_w.shape[1] == 10:
        adapted[:, 0:3] = src_w[:, 0:3]
        adapted[:, 3:4] = 0.5 * src_w[:, 3:4]
        adapted[:, 4:5] = 0.5 * src_w[:, 3:4]
        adapted[:, 5:7] = src_w[:, 5:7]
        return adapted

    if src_w.shape[1] == 8:
        adapted[:, 0:3] = src_w[:, 0:3]
        adapted[:, 3:4] = 0.5 * src_w[:, 3:4]
        adapted[:, 4:5] = 0.5 * src_w[:, 3:4]
        return adapted

    if src_w.shape[1] == 4:
        adapted[:, 0:3] = src_w[:, 0:3]
        adapted[:, 3:4] = 0.5 * src_w[:, 3:4]
        adapted[:, 4:5] = 0.5 * src_w[:, 3:4]
        return adapted

    return None


def adapt_legacy_state_dict(state_dict, target_state_dict, remap_same_shape=False):
    adapted = dict(state_dict)
    w_key = "layer.net.0.weight"
    if w_key not in adapted or w_key not in target_state_dict:
        return adapted

    src_w = adapted[w_key]
    tgt_w = target_state_dict[w_key]
    if src_w.shape == tgt_w.shape:
        if remap_same_shape:
            remapped = _adapt_first_layer_weight(src_w, tgt_w)
            if remapped is not None:
                adapted[w_key] = remapped
        return adapted

    remapped = _adapt_first_layer_weight(src_w, tgt_w)
    if remapped is not None:
        adapted[w_key] = remapped
    return adapted


class LayerNet(nn.Module):
    def __init__(self, hidden_layers=2, hidden_units=32, activation=nn.Tanh()):
        super().__init__()
        layers = []
        # Input: x, y, z_hat + 6 normalized params (E1, E2, t, r, mu, v0)
        # + 5 derived thickness/interface features.
        current_dim = 9 + 5
        
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
        x_coord = x[:, 0:1]
        y_coord = x[:, 1:2]
        z_coord = x[:, 2:3]
        e1_param = x[:, 3:4]
        e2_param = x[:, 4:5]
        t_param = x[:, 5:6]
        r_param = x[:, 6:7]
        mu_param = x[:, 7:8]
        v0_param = x[:, 8:9]
        
        e_min, e_max = config.E_RANGE
        e_span = (e_max - e_min) if (e_max - e_min) != 0 else 1.0
        e1_norm = (e1_param - e_min) / e_span
        e2_norm = (e2_param - e_min) / e_span

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

        t_total_safe = torch.clamp(t_param, min=1e-6)
        z_hat = z_coord / t_total_safe
        t1 = 0.5 * t_param
        sd_norm = (z_coord - t1) / t_total_safe
        zeta = 2.0 * z_hat - 1.0
        beta = float(getattr(config, "INTERFACE_FEATURE_BETA", 20.0))
        soft = torch.sigmoid(beta * sd_norm)
        bend_scale = (float(config.H) / t_total_safe) ** 3
        bend_max = (float(config.H) / max(float(t_min), 1e-6)) ** 3
        bend_min = (float(config.H) / max(float(t_max), 1e-6)) ** 3
        bend_span = max(bend_max - bend_min, 1e-6)
        bend_norm = (bend_scale - bend_min) / bend_span
        extra_feats = torch.cat([sd_norm, torch.abs(sd_norm), soft, zeta ** 3, bend_norm], dim=1)
        
        x_scaled = torch.cat(
            [x_coord, y_coord, z_hat, e1_norm, e2_norm, t_norm, r_norm, mu_norm, v0_norm, extra_feats],
            dim=1
        )
        u_raw = self.net(x_scaled)
        
        if config.USE_HARD_SIDE_BC:
            x_c = x[:, 0:1]
            y_c = x[:, 1:2]
            mask = x_c * (1.0 - x_c) * y_c * (1.0 - y_c) * 16.0
            u_raw = u_raw * mask
        
        return u_raw

class MultiLayerPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = LayerNet(
            hidden_layers=getattr(config, 'LAYERS', 4),
            hidden_units=getattr(config, 'NEURONS', 64),
        )
        
    def forward(self, x, layer_idx=0):
        return self.layer(x)

    def predict_all(self, x):
        return self.forward(x)

    def set_hard_bc(self, use_hard):
        config.USE_HARD_SIDE_BC = bool(use_hard)
