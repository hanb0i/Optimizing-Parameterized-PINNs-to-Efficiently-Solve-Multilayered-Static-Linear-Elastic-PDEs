
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _layer_ids_from_z_centers(z_centers, t_layers):
    """Assign each element to the layer containing its centroid.

    Layers are ordered bottom-to-top. Adjacent layers share mesh nodes at their
    interface, so displacement continuity is enforced by the global DOFs. The
    assembled equilibrium equations transmit equal-and-opposite interface
    tractions; only material stiffness changes by layer.
    """
    interfaces = np.cumsum(np.array(t_layers, dtype=float))
    layer_ids = np.searchsorted(interfaces, z_centers, side="right")
    return np.clip(layer_ids, 0, len(t_layers) - 1)


def _hex8_stiffness(dx, dy, dz, E, nu):
    lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    gp = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
    ke = np.zeros((24, 24))
    c_diag = [lam + 2 * mu, lam + 2 * mu, lam + 2 * mu, mu, mu, mu]
    c_mat = np.zeros((6, 6))
    c_mat[0:3, 0:3] = lam
    np.fill_diagonal(c_mat, c_diag)

    for r_val in gp:
        for s_val in gp:
            for t_val in gp:
                inv_j = np.diag([2 / dx, 2 / dy, 2 / dz])
                det_j = dx * dy * dz / 8.0
                b_mat = np.zeros((6, 24))
                node_signs = [
                    [-1, -1, -1],
                    [1, -1, -1],
                    [1, 1, -1],
                    [-1, 1, -1],
                    [-1, -1, 1],
                    [1, -1, 1],
                    [1, 1, 1],
                    [-1, 1, 1],
                ]
                for node_idx, (xi, eta, zeta) in enumerate(node_signs):
                    dN_dxi = 0.125 * xi * (1 + eta * s_val) * (1 + zeta * t_val)
                    dN_deta = 0.125 * eta * (1 + xi * r_val) * (1 + zeta * t_val)
                    dN_dzeta = 0.125 * zeta * (1 + xi * r_val) * (1 + eta * s_val)
                    d_global = inv_j @ np.array([dN_dxi, dN_deta, dN_dzeta])
                    nx_val, ny_val, nz_val = d_global
                    col = 3 * node_idx
                    b_mat[0, col] = nx_val
                    b_mat[1, col + 1] = ny_val
                    b_mat[2, col + 2] = nz_val
                    b_mat[3, col + 1] = nz_val
                    b_mat[3, col + 2] = ny_val
                    b_mat[4, col] = nz_val
                    b_mat[4, col + 2] = nx_val
                    b_mat[5, col] = ny_val
                    b_mat[5, col + 1] = nx_val
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    ke += b_mat.T @ c_mat @ b_mat * det_j
    return ke


def _assemble_and_solve(cfg, ke_per_element):
    Lx, Ly, H = cfg["geometry"]["Lx"], cfg["geometry"]["Ly"], cfg["geometry"]["H"]
    ne_x = int(cfg["geometry"].get("ne_x", 30))
    ne_y = int(cfg["geometry"].get("ne_y", 30))
    ne_z = int(cfg["geometry"].get("ne_z", 10))

    dx, dy, dz = Lx / ne_x, Ly / ne_y, H / ne_z
    nx, ny, nz = ne_x + 1, ne_y + 1, ne_z + 1
    n_dof = nx * ny * nz * 3

    print("Assembling...")
    el_indices = np.arange(ne_x * ne_y * ne_z)
    ek, ej, ei = np.unravel_index(el_indices, (ne_z, ne_y, ne_x))

    n0 = (ei) + (ej) * nx + (ek) * nx * ny
    n1 = (ei + 1) + (ej) * nx + (ek) * nx * ny
    n2 = (ei + 1) + (ej + 1) * nx + (ek) * nx * ny
    n3 = (ei) + (ej + 1) * nx + (ek) * nx * ny
    n4 = (ei) + (ej) * nx + (ek + 1) * nx * ny
    n5 = (ei + 1) + (ej) * nx + (ek + 1) * nx * ny
    n6 = (ei + 1) + (ej + 1) * nx + (ek + 1) * nx * ny
    n7 = (ei) + (ej + 1) * nx + (ek + 1) * nx * ny

    conn = np.stack([n0, n1, n2, n3, n4, n5, n6, n7], axis=1)

    dof_indices = np.zeros((conn.shape[0], 24), dtype=int)
    for i in range(8):
        dof_indices[:, i * 3 : i * 3 + 3] = conn[:, i : i + 1] * 3 + np.array([0, 1, 2])

    dof_rows = np.repeat(dof_indices, 24, axis=1).ravel()
    dof_cols = np.tile(dof_indices, (1, 24)).ravel()

    vals = ke_per_element.reshape(-1)
    K = sp.coo_matrix((vals, (dof_rows, dof_cols)), shape=(n_dof, n_dof)).tocsr()

    F = np.zeros(n_dof)
    p0 = cfg["load_patch"]["pressure"]

    k = nz - 1
    x_nodes = np.linspace(0, Lx, nx)
    y_nodes = np.linspace(0, Ly, ny)

    patch_x_min = cfg["load_patch"]["x_start"] * Lx
    patch_x_max = cfg["load_patch"]["x_end"] * Lx
    patch_y_min = cfg["load_patch"]["y_start"] * Ly
    patch_y_max = cfg["load_patch"]["y_end"] * Ly

    def load_mask(x, y):
        if x < patch_x_min or x > patch_x_max or y < patch_y_min or y > patch_y_max:
            return 0.0

        x_norm = (x - patch_x_min) / (patch_x_max - patch_x_min)
        y_norm = (y - patch_y_min) / (patch_y_max - patch_y_min)
        return 16.0 * x_norm * (1.0 - x_norm) * y_norm * (1.0 - y_norm)

    # Top traction load integrated over element faces with 2x2 Gauss quadrature.
    # This is more mesh-consistent than assigning dx*dy loads directly to nodes
    # and uses the same smooth patch profile as the PINN load residual.
    gp = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
    face_det_j = dx * dy / 4.0
    top_k = nz - 1
    for elem_j in range(ne_y):
        for elem_i in range(ne_x):
            face_nodes = [
                elem_i + elem_j * nx + top_k * nx * ny,
                (elem_i + 1) + elem_j * nx + top_k * nx * ny,
                (elem_i + 1) + (elem_j + 1) * nx + top_k * nx * ny,
                elem_i + (elem_j + 1) * nx + top_k * nx * ny,
            ]
            x0 = elem_i * dx
            y0 = elem_j * dy
            for r_val in gp:
                for s_val in gp:
                    shape = np.array(
                        [
                            0.25 * (1.0 - r_val) * (1.0 - s_val),
                            0.25 * (1.0 + r_val) * (1.0 - s_val),
                            0.25 * (1.0 + r_val) * (1.0 + s_val),
                            0.25 * (1.0 - r_val) * (1.0 + s_val),
                        ]
                    )
                    x_q = x0 + 0.5 * (r_val + 1.0) * dx
                    y_q = y0 + 0.5 * (s_val + 1.0) * dy
                    traction_z = -p0 * load_mask(x_q, y_q)
                    for node_id, n_val in zip(face_nodes, shape):
                        F[3 * node_id + 2] += traction_z * n_val * face_det_j

    fixed_dofs = []
    for j in range(ny):
        for k in range(nz):
            n_start = 0 + j * nx + k * nx * ny
            n_end = (nx - 1) + j * nx + k * nx * ny
            fixed_dofs.extend([3 * n_start, 3 * n_start + 1, 3 * n_start + 2])
            fixed_dofs.extend([3 * n_end, 3 * n_end + 1, 3 * n_end + 2])

    for i in range(nx):
        for k in range(nz):
            n_start = i + 0 * nx + k * nx * ny
            n_end = i + (ny - 1) * nx + k * nx * ny
            fixed_dofs.extend([3 * n_start, 3 * n_start + 1, 3 * n_start + 2])
            fixed_dofs.extend([3 * n_end, 3 * n_end + 1, 3 * n_end + 2])

    fixed_dofs = np.unique(fixed_dofs)

    penalty = 1e12
    K = K + sp.coo_matrix(
        (np.ones(len(fixed_dofs)) * penalty, (fixed_dofs, fixed_dofs)),
        shape=(n_dof, n_dof),
    ).tocsr()

    print("Solving...")
    u = spla.spsolve(K, F)

    u_grid = np.zeros((nx, ny, nz, 3))
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                idx = i + j * nx + k * nx * ny
                u_grid[i, j, k] = u[3 * idx : 3 * idx + 3]

    return x_nodes, y_nodes, np.linspace(0, H, nz), u_grid


def solve_fem(cfg):
    print("Initializing FEA Solver...")
    Lx, Ly, H = cfg["geometry"]["Lx"], cfg["geometry"]["Ly"], cfg["geometry"]["H"]
    ne_x = int(cfg["geometry"].get("ne_x", 30))
    ne_y = int(cfg["geometry"].get("ne_y", 30))
    ne_z = int(cfg["geometry"].get("ne_z", 10))
    dx, dy, dz = Lx / ne_x, Ly / ne_y, H / ne_z

    material = cfg["material"]
    ke = _hex8_stiffness(dx, dy, dz, material["E"], material["nu"])
    ke_per_element = np.tile(ke.ravel(), ne_x * ne_y * ne_z).reshape(-1, 24 * 24)
    return _assemble_and_solve(cfg, ke_per_element)


def solve_two_layer_fem(cfg):
    print("Initializing Two-Layer FEA Solver...")
    Lx, Ly, H = cfg["geometry"]["Lx"], cfg["geometry"]["Ly"], cfg["geometry"]["H"]
    ne_x = int(cfg["geometry"].get("ne_x", 30))
    ne_y = int(cfg["geometry"].get("ne_y", 30))
    ne_z = int(cfg["geometry"].get("ne_z", 10))
    dx, dy, dz = Lx / ne_x, Ly / ne_y, H / ne_z

    material = cfg["material"]
    nu = material["nu"]
    e_layers = material["E_layers"]
    t_layers = material.get("t_layers", [0.5 * H, 0.5 * H])
    if len(e_layers) != 2 or len(t_layers) != 2:
        raise ValueError("solve_two_layer_fem expects exactly two layers.")

    ke_bottom = _hex8_stiffness(dx, dy, dz, float(e_layers[0]), nu)
    ke_top = _hex8_stiffness(dx, dy, dz, float(e_layers[1]), nu)

    ek, _, _ = np.unravel_index(np.arange(ne_x * ne_y * ne_z), (ne_z, ne_y, ne_x))
    z_centers = (ek + 0.5) * dz
    # Element material lookup: bottom layer is index 0, top layer is index 1.
    # Interface continuity is not imposed with duplicate constraints; it follows
    # from shared nodes and the single global displacement vector.
    layer_ids = _layer_ids_from_z_centers(z_centers, t_layers)
    ke_flat_bottom = ke_bottom.ravel()
    ke_flat_top = ke_top.ravel()
    ke_per_element = np.where(layer_ids[:, None] == 0, ke_flat_bottom[None, :], ke_flat_top[None, :])

    return _assemble_and_solve(cfg, ke_per_element)


def solve_three_layer_fem(cfg):
    print("Initializing Three-Layer FEA Solver...")
    Lx, Ly, H = cfg["geometry"]["Lx"], cfg["geometry"]["Ly"], cfg["geometry"]["H"]
    ne_x = int(cfg["geometry"].get("ne_x", 30))
    ne_y = int(cfg["geometry"].get("ne_y", 30))
    ne_z = int(cfg["geometry"].get("ne_z", 10))
    dx, dy, dz = Lx / ne_x, Ly / ne_y, H / ne_z

    material = cfg["material"]
    nu = material["nu"]
    e_layers = material["E_layers"]
    t_layers = material.get("t_layers", [H / 3.0, H / 3.0, H / 3.0])
    if len(e_layers) != 3 or len(t_layers) != 3:
        raise ValueError("solve_three_layer_fem expects exactly three layers.")

    ke_by_layer = np.stack(
        [_hex8_stiffness(dx, dy, dz, float(e_layers[i]), nu).ravel() for i in range(3)],
        axis=0,
    )

    ek, _, _ = np.unravel_index(np.arange(ne_x * ne_y * ne_z), (ne_z, ne_y, ne_x))
    z_centers = (ek + 0.5) * dz
    # Element material lookup: layers are ordered bottom-to-top as
    # [E1, E2, E3] with interfaces at cumulative [t1, t1+t2].
    # Shared interface nodes carry one displacement value, and the assembled
    # stiffness matrix enforces traction balance weakly across the interface.
    layer_ids = _layer_ids_from_z_centers(z_centers, t_layers)
    ke_per_element = ke_by_layer[layer_ids]

    return _assemble_and_solve(cfg, ke_per_element)
