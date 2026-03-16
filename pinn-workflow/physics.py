
import torch
import torch.autograd as autograd
import pinn_config as config


def _as_layered_input(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize inputs to the 2-layer layout:
      [x, y, z, E1, t1, E2, t2, r, mu, v0]
    """
    if x.shape[1] == 10:
        return x
    if x.shape[1] == 12:
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
        t = x[:, 4:5]
        t_half = 0.5 * t
        e = x[:, 3:4]
        return torch.cat([x[:, 0:3], e, t_half, e, t_half, x[:, 5:8]], dim=1)
    if x.shape[1] == 4:
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

def compliance_scale(E, t):
    e_safe = torch.clamp(E, min=1e-8)
    t_safe = torch.clamp(t, min=1e-8)
    h_ref = float(getattr(config, "H", 1.0))
    alpha = float(getattr(config, "THICKNESS_COMPLIANCE_ALPHA", 0.0))
    e_pow = float(getattr(config, "E_COMPLIANCE_POWER", 1.0))
    if alpha == 0.0:
        return 1.0 / (e_safe ** e_pow)
    return (1.0 / (e_safe ** e_pow)) * (h_ref / t_safe) ** alpha

def v_to_u(v, E, t):
    return v * compliance_scale(E, t)

def load_mask(x):
    x_coord = x[:, 0]
    y_coord = x[:, 1]
    
    x_min, x_max = config.LOAD_PATCH_X
    y_min, y_max = config.LOAD_PATCH_Y
    
    in_patch = (
        (x_coord >= x_min)
        & (x_coord <= x_max)
        & (y_coord >= y_min)
        & (y_coord <= y_max)
    )
    mask = torch.where(in_patch, torch.ones_like(x_coord), torch.zeros_like(x_coord))
    
    return mask


def _to_u(v: torch.Tensor) -> torch.Tensor:
    if v.shape[1] > 3:
        return v[:, 0:3]
    return v

def gradient(u, x):
    # u: (N, 3), x: (N, 3)
    # Returns du/dx: (N, 3, 3)
    # [ [dux/dx, dux/dy, dux/dz],
    #   [duy/dx, duy/dy, duy/dz],
    #   [duz/dx, duz/dy, duz/dz] ]
    
    grad_u = torch.zeros(x.shape[0], 3, 3, device=x.device)
    
    for i in range(3): # u_x, u_y, u_z
        u_i = u[:, i].unsqueeze(1)
        grad_i = autograd.grad(
            u_i, x, 
            grad_outputs=torch.ones_like(u_i),
            create_graph=True, 
            retain_graph=True
        )[0]
        # Extract only spatial gradients (first 3 columns: dx, dy, dz)
        grad_u[:, i, :] = grad_i[:, :3]
        
    return grad_u

def strain(grad_u):
    # epsilon = 0.5 * (grad_u + grad_u^T)
    return 0.5 * (grad_u + grad_u.transpose(1, 2))

def stress(eps, lm, mu):
    # sigma = lambda * tr(eps) * I + 2 * mu * eps
    trace_eps = torch.einsum('bii->b', eps).unsqueeze(1).unsqueeze(2) # (N, 1, 1)
    eye = torch.eye(3, device=eps.device).unsqueeze(0).repeat(eps.shape[0], 1, 1)
    
    sigma = lm * trace_eps * eye + 2 * mu * eps
    return sigma

def divergence(sigma, x):
    # sigma: (N, 3, 3), x: (N, 3)
    # div_sigma: (N, 3) vector
    # We need d(sigma_ij)/dx_j
    
    div = torch.zeros(x.shape[0], 3, device=x.device)
    
    # Row 0: d(sig_xx)/dx + d(sig_xy)/dy + d(sig_xz)/dz
    # etc.
    
    for i in range(3): # For each component of force equilibrium
        # We need d(sigma_i0)/dx + d(sigma_i1)/dy + d(sigma_i2)/dz
        div_i = 0
        for j in range(3):
            sig_ij = sigma[:, i, j].unsqueeze(1)
            grad_sig_ij = autograd.grad(
                sig_ij, x,
                grad_outputs=torch.ones_like(sig_ij),
                create_graph=True,
                retain_graph=True
            )[0]
            div_i += grad_sig_ij[:, j]
        div[:, i] = div_i
        
    return div

def get_material_properties(x):
    # x layout: [x, y, z, E1, t1, E2, t2, r, mu, v0]
    x = _as_layered_input(x)
    z = x[:, 2:3]
    e1 = x[:, 3:4]
    t1 = x[:, 4:5]
    e2 = x[:, 5:6]
    t2 = x[:, 6:7]
    nu_ref = config.nu_vals[0]

    mask1 = (z < t1).float()
    mask2 = (z >= t1).float()
    E = mask1 * e1 + mask2 * e2
    nu = torch.full_like(E, nu_ref)

    return E, nu

def compute_loss(model, data, device, weights=None):
    total_loss = 0
    losses = {}
    if weights is None:
        weights = config.WEIGHTS
    
    # --- 1. PDE Residuals (Interior) ---
    x_int = data['interior'][0].to(device).detach().clone().requires_grad_(True)
    E_int, nu_int = get_material_properties(x_int)
    lm_int, mu_int = get_lame_params_torch(E_int, nu_int)
    
    u_int = _to_u(model(x_int))
    grad_u = gradient(u_int, x_int)
    sig = stress(strain(grad_u), lm_int.unsqueeze(2), mu_int.unsqueeze(2))
    div_sigma = divergence(sig, x_int)
    
    pde_res = div_sigma * getattr(config, "PDE_LENGTH_SCALE", 1.0)
    pde_loss = torch.mean(pde_res**2)
    losses['pde'] = pde_loss
    total_loss += weights.get('pde', 1.0) * pde_loss
    
    # --- 2. Dirichlet BCs (Sides) ---
    x_side = data['sides'][0].to(device)
    u_side = _to_u(model(x_side))
    side_loss = torch.mean(u_side**2)
    losses['bc_sides'] = side_loss
    total_loss += weights.get('bc', 1.0) * side_loss
    
    # --- 3. Traction BCs (Top & Bottom) ---
    # Top Load
    x_top_l = data['top_load'].to(device).detach().clone().requires_grad_(True)
    E_tl, nu_tl = get_material_properties(x_top_l)
    lm_tl, mu_tl = get_lame_params_torch(E_tl, nu_tl)
    u_tl = _to_u(model(x_top_l))
    sig_tl = stress(strain(gradient(u_tl, x_top_l)), lm_tl.unsqueeze(2), mu_tl.unsqueeze(2))
    
    T_tl = sig_tl[:, :, 2]
    # Normal is [0,0,1]
    target_tl = torch.zeros_like(T_tl)
    target_tl[:, 2] = -config.p0 # load_mask is 1 since data is sampled under patch
    load_loss = torch.mean((T_tl - target_tl)**2)
    losses['load'] = load_loss
    total_loss += weights.get('load', 1.0) * load_loss
    
    # Top Free
    x_top_f = data['top_free'].to(device).detach().clone().requires_grad_(True)
    E_tf, nu_tf = get_material_properties(x_top_f)
    lm_tf, mu_tf = get_lame_params_torch(E_tf, nu_tf)
    u_tf = _to_u(model(x_top_f))
    sig_tf = stress(strain(gradient(u_tf, x_top_f)), lm_tf.unsqueeze(2), mu_tf.unsqueeze(2))
    T_tf = sig_tf[:, :, 2] # Normal [0,0,1]
    free_top_loss = torch.mean(T_tf**2)
    losses['free_top'] = free_top_loss
    total_loss += weights.get('bc', 1.0) * free_top_loss
    
    # Bottom Free
    x_bot = data['bottom'].to(device).detach().clone().requires_grad_(True)
    E_bot, nu_bot = get_material_properties(x_bot)
    lm_bot, mu_bot = get_lame_params_torch(E_bot, nu_bot)
    u_bot = _to_u(model(x_bot))
    sig_bot = stress(strain(gradient(u_bot, x_bot)), lm_bot.unsqueeze(2), mu_bot.unsqueeze(2))
    T_bot = -sig_bot[:, :, 2] # Normal [0,0,-1]
    free_bot_loss = torch.mean(T_bot**2)
    losses['free_bot'] = free_bot_loss
    total_loss += weights.get('bc', 1.0) * free_bot_loss
    
    # --- 4. Interface Continuity (single-net, 2-layer) ---
    x_if = data.get("interface1", None)
    if x_if is not None:
        x_if = x_if.to(device).detach().clone().requires_grad_(True)
        x_if = _as_layered_input(x_if)

        e1 = x_if[:, 3:4]
        e2 = x_if[:, 5:6]
        nu_ref = nu_int[0]

        u_if = _to_u(model(x_if))
        grad_if = gradient(u_if, x_if)
        eps_if = strain(grad_if)

        lm1, mu1 = get_lame_params_torch(e1, nu_ref)
        lm2, mu2 = get_lame_params_torch(e2, nu_ref)
        t1 = stress(eps_if, lm1.unsqueeze(2), mu1.unsqueeze(2))[:, :, 2]
        t2 = stress(eps_if, lm2.unsqueeze(2), mu2.unsqueeze(2))[:, :, 2]

        loss_if_t = torch.mean((t1 - t2) ** 2)
        losses["interface_traction_1"] = loss_if_t
        losses["interface_traction"] = loss_if_t
        total_loss += weights.get("interface_traction", 50.0) * loss_if_t

        loss_if_u = torch.zeros((), device=device)
        losses["interface_u_1"] = loss_if_u
        total_loss += weights.get("interface_u", 0.0) * loss_if_u
    else:
        losses["interface_traction_1"] = torch.zeros((), device=device)
        losses["interface_u_1"] = torch.zeros((), device=device)

    losses['energy'] = torch.tensor(0.0).to(device)
    losses['total'] = total_loss
    return total_loss, losses

def get_lame_params_torch(E, nu):
    lm = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lm, mu

def compute_residuals(model, data, device):
    residuals = {}

    # --- PDE Residuals (Interior) ---
    x_int = data['interior'][0].to(device).detach().clone().requires_grad_(True)
    x_int = _as_layered_input(x_int)
    
    E_local, nu_local = get_material_properties(x_int)
    t_local = x_int[:, 4:5] + x_int[:, 6:7]
    
    lm = (E_local * nu_local) / ((1 + nu_local) * (1 - 2 * nu_local))
    mu = E_local / (2 * (1 + nu_local))
    lm = lm.unsqueeze(2)
    mu = mu.unsqueeze(2)
    
    v_int = model(x_int)
    u = _to_u(v_int)
    grad_u = gradient(u, x_int)
    eps = strain(grad_u)
    sig = stress(eps, lm, mu)
    div_sigma = divergence(sig, x_int)
    
    residual = -div_sigma * getattr(config, "PDE_LENGTH_SCALE", 1.0)
    residual_mag = torch.sqrt(torch.sum(residual**2, dim=1))
    residuals['interior'] = residual_mag.cpu()
    
    # --- BC Sides Residuals ---
    x_side = data['sides'][0].to(device)
    x_side = _as_layered_input(x_side)
    E_side, _ = get_material_properties(x_side)
    v_side = model(x_side)
    u_side = _to_u(v_side)
    bc_residual = torch.sqrt(torch.sum(u_side**2, dim=1))
    residuals['sides'] = bc_residual.cpu()
    
    # --- Top Load Residuals ---
    x_top_load = data['top_load'].to(device).detach().clone().requires_grad_(True)
    x_top_load = _as_layered_input(x_top_load)
    
    E_local_load, nu_local_load = get_material_properties(x_top_load)
    lm = (E_local_load * nu_local_load) / ((1 + nu_local_load) * (1 - 2 * nu_local_load))
    mu = E_local_load / (2 * (1 + nu_local_load))
    lm = lm.unsqueeze(2)
    mu = mu.unsqueeze(2)
    
    v_top = model(x_top_load)
    u_top = _to_u(v_top)
    grad_u_top = gradient(u_top, x_top_load)
    sig_top = stress(strain(grad_u_top), lm, mu)
    T = sig_top[:, :, 2]
    mask = load_mask(x_top_load).unsqueeze(1)
    target_load = -config.p0 * mask
    target = torch.cat([
        torch.zeros_like(target_load),
        torch.zeros_like(target_load),
        target_load,
    ], dim=1)
    load_residual = torch.sqrt(torch.sum((T - target) ** 2, dim=1))
    residuals['top_load'] = load_residual.cpu()
    
    # --- Top Free Residuals ---
    x_top_free = data['top_free'].to(device).detach().clone().requires_grad_(True)
    x_top_free = _as_layered_input(x_top_free)
    
    E_local_free, nu_local_free = get_material_properties(x_top_free)
    lm_free = (E_local_free * nu_local_free) / ((1 + nu_local_free) * (1 - 2 * nu_local_free))
    mu_free = E_local_free / (2 * (1 + nu_local_free))
    lm_free = lm_free.unsqueeze(2)
    mu_free = mu_free.unsqueeze(2)
    
    v_top_free = model(x_top_free)
    u_top_free = _to_u(v_top_free)
    grad_u_free = gradient(u_top_free, x_top_free)
    sig_top_free = stress(strain(grad_u_free), lm_free, mu_free)
    T_free = sig_top_free[:, :, 2]
    free_residual = torch.sqrt(torch.sum(T_free**2, dim=1))
    residuals['top_free'] = free_residual.cpu()
    
    # --- Bottom Residuals ---
    x_bot = data['bottom'].to(device).detach().clone().requires_grad_(True)
    x_bot = _as_layered_input(x_bot)
    
    E_local_bot, nu_local_bot = get_material_properties(x_bot)
    lm_bot = (E_local_bot * nu_local_bot) / ((1 + nu_local_bot) * (1 - 2 * nu_local_bot))
    mu_bot = E_local_bot / (2 * (1 + nu_local_bot))
    lm_bot = lm_bot.unsqueeze(2)
    mu_bot = mu_bot.unsqueeze(2)
    
    v_bot = model(x_bot)
    u_bot = _to_u(v_bot)
    grad_u_bot = gradient(u_bot, x_bot)
    sig_bot = stress(strain(grad_u_bot), lm_bot, mu_bot)
    T_bot = -sig_bot[:, :, 2]
    bot_residual = torch.sqrt(torch.sum(T_bot**2, dim=1))
    residuals['bottom'] = bot_residual.cpu()

    return residuals
