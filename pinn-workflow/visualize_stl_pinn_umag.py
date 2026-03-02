#!/usr/bin/env python3
"""
Visualize an STL surface mesh colored by PINN-predicted displacement magnitude |u|.

Note: this is a *visualization utility*. Unless you explicitly trained the PINN with
contact constraints on that surface, evaluating the network on an arbitrary STL (e.g.
a sphere) is not "enforcement" of physics on the STL; it's just sampling the network.
"""

from __future__ import annotations

import argparse
import os
import struct
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

import model as pinn_model
import pinn_config as config


Tri = np.ndarray  # shape (3, 3)


def _read_ascii_stl(text: str) -> List[Tri]:
    verts: List[Tuple[float, float, float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.lower().startswith("vertex"):
            continue
        parts = line.split()
        if len(parts) != 4:
            continue
        verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if len(verts) % 3 != 0:
        raise ValueError(f"ASCII STL parse error: vertex count {len(verts)} is not divisible by 3.")
    tris: List[Tri] = []
    for i in range(0, len(verts), 3):
        tri = np.array([verts[i], verts[i + 1], verts[i + 2]], dtype=np.float32)
        tris.append(tri)
    return tris


def _read_binary_stl(blob: bytes) -> List[Tri]:
    if len(blob) < 84:
        raise ValueError("Binary STL too small.")
    n_tri = struct.unpack("<I", blob[80:84])[0]
    offset = 84
    stride = 50
    expected = offset + n_tri * stride
    if len(blob) < expected:
        raise ValueError(f"Binary STL truncated: expected at least {expected} bytes, got {len(blob)}.")
    tris: List[Tri] = []
    for _ in range(n_tri):
        # normal (3 floats) + v1,v2,v3 (9 floats) + attr (uint16)
        chunk = blob[offset : offset + stride]
        floats = struct.unpack("<12f", chunk[:48])
        v1 = floats[3:6]
        v2 = floats[6:9]
        v3 = floats[9:12]
        tris.append(np.array([v1, v2, v3], dtype=np.float32))
        offset += stride
    return tris


def load_stl(path: Path) -> List[Tri]:
    blob = path.read_bytes()
    # Heuristic: ASCII STLs start with "solid". Binary can also start with "solid",
    # but most binaries have non-text header; we attempt binary first when it looks binary.
    head = blob[:80].lower()
    if head.startswith(b"solid") and b"\n" in blob[:200]:
        try:
            return _read_ascii_stl(blob.decode("utf-8", errors="strict"))
        except Exception:
            # Fall back to binary
            return _read_binary_stl(blob)
    return _read_binary_stl(blob)


def main() -> int:
    parser = argparse.ArgumentParser(description="Color an STL mesh by PINN |u| prediction.")
    parser.add_argument("--stl", required=True, help="Path to STL file.")
    parser.add_argument("--out", default=None, help="Output PNG path (default: pinn-workflow/visualization/<name>_umag.png).")
    parser.add_argument("--ckpt", default=None, help="Checkpoint path (default: pinn-workflow/pinn_model.pth if present).")
    parser.add_argument("--device", default=None, help="Torch device (e.g. cpu, cuda, mps).")

    # Conditioning inputs (constants)
    parser.add_argument("--E", type=float, default=1.0, help="Young's modulus input channel (constant).")
    parser.add_argument("--t", type=float, default=1.0, help="Thickness input channel (constant).")
    parser.add_argument("--r", type=float, default=float(getattr(config, "RESTITUTION_REF", 0.5)), help="Restitution input channel.")
    parser.add_argument("--mu", type=float, default=float(getattr(config, "FRICTION_REF", 0.3)), help="Friction input channel.")
    parser.add_argument("--v0", type=float, default=float(getattr(config, "IMPACT_VELOCITY_REF", 1.0)), help="Impact velocity input channel.")
    parser.add_argument("--max_faces", type=int, default=20000, help="Limit faces plotted (for speed).")

    args = parser.parse_args()
    # Matplotlib writes cache/config files and may default to an interactive backend.
    # Force a headless-friendly backend and a writable config dir in the workspace.
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).resolve().parent / ".mplconfig").resolve()))
    stl_path = Path(args.stl)
    if not stl_path.exists():
        raise FileNotFoundError(f"STL not found: {stl_path}")

    out_path: Path
    if args.out:
        out_path = Path(args.out)
    else:
        out_dir = Path(__file__).resolve().parent / "visualization"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stl_path.stem}_umag.png"

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else (
            torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        )
    )

    # Load model
    pinn = pinn_model.MultiLayerPINN().to(device)
    ckpt = args.ckpt
    if ckpt is None:
        # Prefer pinn-workflow/pinn_model.pth when running from repo root.
        candidates = [
            Path("pinn-workflow") / "pinn_model.pth",
            Path(__file__).resolve().parent / "pinn_model.pth",
            Path("pinn_model.pth"),
        ]
        for c in candidates:
            if c.exists():
                ckpt = str(c)
                break
    if ckpt is None or not Path(ckpt).exists():
        raise FileNotFoundError("PINN checkpoint not found. Provide --ckpt or create pinn_model.pth.")
    pinn.load_state_dict(torch.load(ckpt, map_location=device))
    pinn.eval()

    tris = load_stl(stl_path)
    if not tris:
        raise ValueError("No triangles found in STL.")

    if args.max_faces and len(tris) > args.max_faces:
        tris = tris[: args.max_faces]

    tri_arr = np.stack(tris, axis=0)  # (F, 3, 3)
    centroids = tri_arr.mean(axis=1)  # (F, 3)

    # Build model inputs: [x,y,z,E,t,r,mu,v0]
    N = centroids.shape[0]
    x = torch.from_numpy(centroids.astype(np.float32)).to(device)
    E = torch.full((N, 1), float(args.E), device=device)
    t = torch.full((N, 1), float(args.t), device=device)
    r = torch.full((N, 1), float(args.r), device=device)
    mu = torch.full((N, 1), float(args.mu), device=device)
    v0 = torch.full((N, 1), float(args.v0), device=device)
    inputs = torch.cat([x, E, t, r, mu, v0], dim=1)

    with torch.no_grad():
        u = pinn(inputs, 0)
        umag = torch.linalg.vector_norm(u, ord=2, dim=1).detach().cpu().numpy()

    # Plot
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    norm = Normalize(vmin=float(np.min(umag)), vmax=float(np.max(umag)))
    cmap = plt.get_cmap("jet")
    facecolors = cmap(norm(umag))

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    coll = Poly3DCollection(tri_arr, facecolors=facecolors, edgecolor=(0, 0, 0, 0.08), linewidths=0.2)
    ax.add_collection3d(coll)

    ax.set_title(f"STL mesh colored by PINN umag ({stl_path.name})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    mins = centroids.min(axis=0)
    maxs = centroids.max(axis=0)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(umag)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.65, pad=0.10)
    cbar.set_label("|U| (pred)")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
