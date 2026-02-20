import os
import sys
import argparse

import numpy as np
import torch

# Avoid .pyc writes in locked-down environments.
sys.dont_write_bytecode = True

# Matplotlib needs writable cache + temp dirs in this environment.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_local_tmp = os.path.join(REPO_ROOT, ".tmp")
_mpl_cache = os.path.join(REPO_ROOT, ".cache", "matplotlib")
os.makedirs(_local_tmp, exist_ok=True)
os.makedirs(_mpl_cache, exist_ok=True)
os.environ.setdefault("TMPDIR", _local_tmp)
os.environ.setdefault("TEMP", _local_tmp)
os.environ.setdefault("TMP", _local_tmp)
os.environ.setdefault("MPLCONFIGDIR", _mpl_cache)

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

PINN_WORKFLOW_DIR = os.path.join(REPO_ROOT, "pinn-workflow")
if PINN_WORKFLOW_DIR not in sys.path:
    sys.path.insert(0, PINN_WORKFLOW_DIR)

import pinn_config as config
import model
from tessellated_geometry import load_stl_surface, affine_map_surface_to_bounds, sample_boundary


def _default_stl_path() -> str:
    candidates = [
        os.path.join(PINN_WORKFLOW_DIR, "stl", "sphere.stl"),
        os.path.join(PINN_WORKFLOW_DIR, "stl", "unit_plate_1x1x0p1.stl"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Could not find an STL under `pinn-workflow/stl/`.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize PiNN predictions on an STL CAD surface.")
    parser.add_argument(
        "--stl",
        default=None,
        help="Path to STL. If omitted, uses `pinn_config.CAD_STL_PATH` when set; otherwise auto-picks `sphere.stl` if present.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Do not map STL into the (0..Lx, 0..Ly, 0..H) box; use STL coordinates as-is.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for PNGs. Default: impact_pipeline_outputs/cad_viz/",
    )
    parser.add_argument(
        "--deform-scale",
        type=float,
        default=0.5,
        help="Scale factor for visualizing deformed geometry.",
    )
    return parser.parse_args()


def _set_box_aspect_from_points(ax, xyz: np.ndarray) -> None:
    mins = np.min(xyz, axis=0)
    maxs = np.max(xyz, axis=0)
    spans = np.maximum(1e-9, maxs - mins)
    spans = spans / float(np.max(spans))
    ax.set_box_aspect(tuple(spans.tolist()))


def _render_mesh(
    triangles_xyz: np.ndarray,
    face_values: np.ndarray,
    out_path: str,
    title: str,
    cmap_name: str = "viridis",
) -> None:
    tri = np.asarray(triangles_xyz, dtype=np.float64)
    vals = np.asarray(face_values, dtype=np.float64).reshape(-1)
    if tri.ndim != 3 or tri.shape[1:] != (3, 3):
        raise ValueError(f"Expected triangles shaped (T,3,3), got {tri.shape}")
    if len(vals) != tri.shape[0]:
        raise ValueError(f"Expected {tri.shape[0]} face values, got {len(vals)}")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = -1.0, 1.0

    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    face_colors = cmap(norm(vals))

    coll = Poly3DCollection(tri, facecolors=face_colors, linewidths=0.05, edgecolors=(0, 0, 0, 0.15))
    coll.set_alpha(1.0)
    ax.add_collection3d(coll)

    pts = tri.reshape(-1, 3)
    ax.set_xlim(float(np.min(pts[:, 0])), float(np.max(pts[:, 0])))
    ax.set_ylim(float(np.min(pts[:, 1])), float(np.max(pts[:, 1])))
    ax.set_zlim(float(np.min(pts[:, 2])), float(np.max(pts[:, 2])))
    _set_box_aspect_from_points(ax, pts)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(vals)
    fig.colorbar(sm, ax=ax, shrink=0.6, label="Uz (pred)")

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _load_model(model_path: str, device: torch.device) -> model.MultiLayerPINN:
    pinn = model.MultiLayerPINN().to(device)
    if os.path.exists(model_path):
        sd = torch.load(model_path, map_location=device, weights_only=True)
        target_sd = pinn.state_dict()
        w_key = "layer.net.0.weight"
        if w_key in sd and w_key in target_sd:
            src_w = sd[w_key]
            tgt_w = target_sd[w_key]
            if src_w.shape != tgt_w.shape and src_w.shape[0] == tgt_w.shape[0]:
                if src_w.shape[1] == 8 and tgt_w.shape[1] == 11:
                    adapted = torch.zeros_like(tgt_w)
                    adapted[:, 0:5] = src_w[:, 0:5]
                    adapted[:, 8:11] = src_w[:, 5:8]
                    sd[w_key] = adapted
                elif src_w.shape[1] == 10 and tgt_w.shape[1] == 11:
                    adapted = torch.zeros_like(tgt_w)
                    adapted[:, 0:7] = src_w[:, 0:7]
                    adapted[:, 8:11] = src_w[:, 7:10]
                    sd[w_key] = adapted
        pinn.load_state_dict(sd, strict=False)
        print(f"Model loaded from {model_path}")
    else:
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    pinn.eval()
    return pinn


def _u_from_v(v: np.ndarray, E_val: float, thickness: float) -> np.ndarray:
    alpha = float(getattr(config, "THICKNESS_COMPLIANCE_ALPHA", 0.0))
    t_scale = 1.0 if alpha == 0.0 else (float(config.H) / max(1e-8, float(thickness))) ** alpha
    e_pow = float(getattr(config, "E_COMPLIANCE_POWER", 1.0))
    return (v / (float(E_val) ** e_pow)) * t_scale


def main():
    device = torch.device("cpu")

    args = _parse_args()

    # Configure CAD mode (this script sets CAD mode explicitly so it runs without editing config)
    config.GEOMETRY_MODE = "cad"
    config.CAD_SAMPLER = "tessellation"

    if args.stl is not None:
        config.CAD_STL_PATH = args.stl
    elif config.CAD_STL_PATH is None:
        config.CAD_STL_PATH = _default_stl_path()

    if not os.path.exists(str(config.CAD_STL_PATH)):
        raise FileNotFoundError(f"STL not found: {config.CAD_STL_PATH}")

    config.CAD_NORMALIZE_TO_CONFIG_BOUNDS = not args.no_normalize
    if config.CAD_NORMALIZE_TO_CONFIG_BOUNDS:
        stl_base = os.path.basename(str(config.CAD_STL_PATH)).lower()
        # Prevent "sphere.stl" from being squashed into a thin plate by H=0.1 defaults.
        if "sphere" in stl_base:
            cube = max(float(config.Lx), float(config.Ly), float(config.H))
            config.Lx = cube
            config.Ly = cube
            config.H = cube

    out_dir = args.out_dir or os.path.join(REPO_ROOT, "impact_pipeline_outputs", "cad_viz")
    os.makedirs(out_dir, exist_ok=True)

    ckpt_candidates = [
        os.path.join(PINN_WORKFLOW_DIR, "pinn_model.pth"),
        os.path.join(REPO_ROOT, "pinn_model.pth"),
    ]
    ckpt = next((p for p in ckpt_candidates if os.path.exists(p)), None)
    if ckpt is None:
        raise FileNotFoundError("Could not find `pinn_model.pth` in repo root or `pinn-workflow/`.")

    pinn = _load_model(ckpt, device)
    # Hard side masks assume a unit box boundary; disable for CAD visualization.
    config.USE_HARD_SIDE_BC = False
    pinn.set_hard_bc(False)

    # For visualization, sample additional boundary points directly on the STL surface.
    surface = load_stl_surface(config.CAD_STL_PATH)
    if config.CAD_NORMALIZE_TO_CONFIG_BOUNDS:
        surface = affine_map_surface_to_bounds(
            surface,
            (0.0, 0.0, 0.0),
            (float(config.Lx), float(config.Ly), float(config.H)),
        )
    bnd = sample_boundary(surface, nr_points=20000)
    xyz = np.concatenate([bnd["x"], bnd["y"], bnd["z"]], axis=1)

    # Use reference parameters for a single forward pass
    E_val = float(getattr(config, "E_vals", [1.0])[0])
    thickness = float(getattr(config, "H", 0.1))
    r_ref = float(getattr(config, "RESTITUTION_REF", 0.5))
    mu_ref = float(getattr(config, "FRICTION_REF", 0.3))
    v0_ref = float(getattr(config, "IMPACT_VELOCITY_REF", 1.0))

    params = np.stack(
        [
            np.full(len(xyz), E_val),
            np.full(len(xyz), thickness),
            np.full(len(xyz), r_ref),
            np.full(len(xyz), mu_ref),
            np.full(len(xyz), v0_ref),
        ],
        axis=1,
    )
    pts = np.concatenate([xyz, params], axis=1).astype(np.float32)

    with torch.no_grad():
        v = pinn(torch.tensor(pts, dtype=torch.float32).to(device)).cpu().numpy()
    u = _u_from_v(v, E_val, thickness)

    uz = u[:, 2]
    uz_min, uz_max = float(np.min(uz)), float(np.max(uz))
    print(f"CAD surface Uz range: [{uz_min:.6f}, {uz_max:.6f}]")

    # 3D scatter on CAD surface, colored by Uz
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=uz, cmap="viridis", s=2)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="Uz (pred)")
    ax.set_title(f"CAD STL surface sampled points colored by PiNN Uz ({os.path.basename(str(config.CAD_STL_PATH))})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    _set_box_aspect_from_points(ax, xyz)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cad_surface_uz_scatter.png"), dpi=200)
    plt.close(fig)

    # Deformed visualization (scaled for visibility)
    scale = 2.0
    xyz_def = xyz.copy()
    xyz_def[:, 2] = xyz_def[:, 2] + scale * uz
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(xyz_def[:, 0], xyz_def[:, 1], xyz_def[:, 2], c=uz, cmap="viridis", s=2)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="Uz (pred)")
    ax.set_title(
        f"CAD surface (deformed) colored by PiNN Uz (scale={scale}) ({os.path.basename(str(config.CAD_STL_PATH))})"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    _set_box_aspect_from_points(ax, xyz_def)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cad_surface_uz_deformed_scatter.png"), dpi=200)
    plt.close(fig)

    # FEA-like: render the actual STL triangle mesh, colored by predicted Uz
    tri = surface.triangles.astype(np.float64, copy=False)  # (T,3,3)
    tri_pts = tri.reshape(-1, 3)
    tri_params = np.stack(
        [
            np.full(len(tri_pts), E_val),
            np.full(len(tri_pts), thickness),
            np.full(len(tri_pts), r_ref),
            np.full(len(tri_pts), mu_ref),
            np.full(len(tri_pts), v0_ref),
        ],
        axis=1,
    )
    tri_in = np.concatenate([tri_pts, tri_params], axis=1).astype(np.float32)
    with torch.no_grad():
        v_tri = pinn(torch.tensor(tri_in, dtype=torch.float32).to(device)).cpu().numpy()
    u_tri = _u_from_v(v_tri, E_val, thickness).reshape(tri.shape[0], 3, 3)
    uz_face = np.mean(u_tri[:, :, 2], axis=1)  # (T,)

    _render_mesh(
        triangles_xyz=tri,
        face_values=uz_face,
        out_path=os.path.join(out_dir, "cad_mesh_uz.png"),
        title=f"STL mesh colored by PiNN Uz ({os.path.basename(str(config.CAD_STL_PATH))})",
    )

    tri_def = tri + float(args.deform_scale) * u_tri
    _render_mesh(
        triangles_xyz=tri_def,
        face_values=uz_face,
        out_path=os.path.join(out_dir, "cad_mesh_uz_deformed.png"),
        title=f"Deformed STL mesh (scale={args.deform_scale}) colored by PiNN Uz ({os.path.basename(str(config.CAD_STL_PATH))})",
    )

    # Top surface (load vs free) quick sanity plot
    z = xyz[:, 2]
    z_min, z_max = float(np.min(z)), float(np.max(z))
    z_span = max(1e-12, z_max - z_min)
    load_cap = float(getattr(config, "CAD_LOAD_Z_FRAC", 0.02)) * z_span
    z_load_thr = z_max - load_cap
    is_top = z >= z_load_thr

    x = xyz[:, 0]
    y = xyz[:, 1]
    x0, x1 = float(config.LOAD_PATCH_X[0]), float(config.LOAD_PATCH_X[1])
    y0, y1 = float(config.LOAD_PATCH_Y[0]), float(config.LOAD_PATCH_Y[1])
    in_patch = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    is_top_load = is_top & in_patch
    is_top_free = is_top & (~in_patch)

    top_xyz = xyz[is_top]
    top_params = np.stack(
        [
            np.full(len(top_xyz), E_val),
            np.full(len(top_xyz), thickness),
            np.full(len(top_xyz), r_ref),
            np.full(len(top_xyz), mu_ref),
            np.full(len(top_xyz), v0_ref),
        ],
        axis=1,
    )
    top_pts = np.concatenate([top_xyz, top_params], axis=1).astype(np.float32)
    with torch.no_grad():
        v_top = pinn(torch.tensor(top_pts, dtype=torch.float32).to(device)).cpu().numpy()
    u_top = _u_from_v(v_top, E_val, thickness)
    uz_top = u_top[:, 2]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(top_xyz[:, 0], top_xyz[:, 1], top_xyz[:, 2], c=uz_top, cmap="viridis", s=3)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="Uz (pred)")
    ax.set_title("Top surface points (load + free) colored by PiNN Uz")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    _set_box_aspect_from_points(ax, top_xyz)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cad_top_surface_uz.png"), dpi=200)
    plt.close(fig)

    # Load/free mask plot (shows where the "pressure patch" is on the CAD surface)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    colors = np.full(len(xyz), 0.5, dtype=np.float64)
    colors[is_top_free] = 0.0
    colors[is_top_load] = 1.0
    sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, cmap="coolwarm", s=2, vmin=0.0, vmax=1.0)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="0=top_free, 1=top_load (else mid)")
    ax.set_title("CAD surface regions used for BCs (top load patch vs top free)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    _set_box_aspect_from_points(ax, xyz)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cad_surface_bc_regions.png"), dpi=200)
    plt.close(fig)

    print(f"Wrote CAD visualization outputs to: {out_dir}")


if __name__ == "__main__":
    main()
