from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PINN_WORKFLOW_DIR = os.path.join(REPO_ROOT, "pinn-workflow")
FEA_SOLVER_DIR = os.path.join(REPO_ROOT, "fea-workflow", "solver")
if PINN_WORKFLOW_DIR not in sys.path:
    sys.path.insert(0, PINN_WORKFLOW_DIR)
if FEA_SOLVER_DIR not in sys.path:
    sys.path.insert(0, FEA_SOLVER_DIR)

import pinn_config as config
import model
import physics
import fem_solver


@dataclass(frozen=True)
class Case:
    H: float
    frac: float
    E1: float
    E2: float


def _infer_arch_from_state_dict(sd: dict) -> tuple[int, int]:
    w0 = sd.get("layers.0.net.0.weight", None)
    if w0 is None:
        return int(getattr(config, "LAYERS", 4)), int(getattr(config, "NEURONS", 64))
    neurons = int(w0.shape[0])
    linear_indices = set()
    for k in sd.keys():
        if not (isinstance(k, str) and k.startswith("layers.0.net.") and k.endswith(".weight")):
            continue
        try:
            idx = int(k.split(".")[3])
        except Exception:
            continue
        linear_indices.add(idx)
    hidden_layers = max(1, len(linear_indices) - 1) if linear_indices else int(getattr(config, "LAYERS", 4))
    return hidden_layers, neurons


def _select_device(device_str: str | None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _relative_l2(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    num = float(np.linalg.norm(a - b))
    den = float(np.linalg.norm(b))
    return num / max(den, eps)


def _parse_only_cases(s: str | None) -> set[int] | None:
    if not s:
        return None
    out: set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def _load_cases_json(path: str) -> list[Case]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError("cases_json must be a list of objects like {H, frac, E1, E2}.")
    out: list[Case] = []
    for item in raw:
        out.append(Case(H=float(item["H"]), frac=float(item["frac"]), E1=float(item["E1"]), E2=float(item["E2"])))
    return out


def _sample_cases(seed: int, n: int) -> list[Case]:
    rng = np.random.default_rng(int(seed))
    e_min, e_max = map(float, getattr(config, "E_RANGE", (1.0, 10.0)))
    t_min, t_max = map(float, getattr(config, "THICKNESS_RANGE", (float(getattr(config, "H", 0.1)), float(getattr(config, "H", 0.1)))))
    frac_min = float(getattr(config, "LAYER_THICKNESS_FRACTION_MIN", 0.05))
    frac_min = max(1e-4, min(frac_min, 0.49))
    out: list[Case] = []
    for _ in range(int(n)):
        H = float(rng.uniform(t_min, t_max))
        frac = float(rng.uniform(frac_min, 1.0 - frac_min))
        E1 = float(rng.uniform(e_min, e_max))
        E2 = float(rng.uniform(e_min, e_max))
        out.append(Case(H=H, frac=frac, E1=E1, E2=E2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate side-by-side plots: 2-layer PINN vs layered FEA (top + XZ).")
    ap.add_argument("--model", required=True, help="PINN checkpoint path.")
    ap.add_argument("--device", default=None, help="cpu|cuda|mps (auto if omitted).")
    ap.add_argument("--out_dir", default="compare_random_cases", help="Output directory for PNGs.")

    ap.add_argument("--cases_json", default=None, help="Optional JSON list of cases [{H, frac, E1, E2}, ...].")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cases", type=int, default=20, help="Number of random cases if --cases_json is not set.")
    ap.add_argument("--only_cases", default=None, help="Comma/range list, e.g. '0,2,5-7'.")
    ap.add_argument("--err_thresh_pct", type=float, default=5.0)

    ap.add_argument("--ne_x", type=int, default=16)
    ap.add_argument("--ne_y", type=int, default=16)
    ap.add_argument("--ne_z", type=int, default=16)
    ap.add_argument("--nu", type=float, default=float(getattr(config, "NU_FIXED", 0.3)))
    ap.add_argument("--p0", type=float, default=float(getattr(config, "p0", 1.0)))
    ap.add_argument("--use_soft_mask", type=int, default=int(getattr(config, "USE_SOFT_LOAD_MASK", True)))
    args = ap.parse_args()

    if int(getattr(config, "NUM_LAYERS", 2)) != 2:
        raise ValueError(f"Expected NUM_LAYERS=2 for this script (got {getattr(config, 'NUM_LAYERS', None)}).")

    os.makedirs(args.out_dir, exist_ok=True)
    only = _parse_only_cases(args.only_cases)

    import matplotlib

    if os.environ.get("MPLBACKEND") is None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = _select_device(args.device)
    pinn = model.MultiLayerPINN().to(device)
    if args.model and os.path.exists(args.model):
        sd = torch.load(args.model, map_location=device, weights_only=False)
        inferred_layers, inferred_neurons = _infer_arch_from_state_dict(sd)
        config.LAYERS = int(inferred_layers)
        config.NEURONS = int(inferred_neurons)
        pinn = model.MultiLayerPINN().to(device)
        pinn.load_state_dict(sd, strict=False)
        print(f"Loaded model: {args.model} (layers={config.LAYERS}, neurons={config.NEURONS})")
    else:
        raise FileNotFoundError(f"Model checkpoint not found: {args.model!r}")
    pinn.eval()

    if args.cases_json:
        cases = _load_cases_json(args.cases_json)
    else:
        cases = _sample_cases(int(args.seed), int(args.cases))

    x0, x1 = map(float, getattr(config, "LOAD_PATCH_X", (1.0 / 3.0, 2.0 / 3.0)))
    y0, y1 = map(float, getattr(config, "LOAD_PATCH_Y", (1.0 / 3.0, 2.0 / 3.0)))
    Lx = float(getattr(config, "Lx", 1.0))
    Ly = float(getattr(config, "Ly", 1.0))

    r_ref = float(getattr(config, "RESTITUTION_REF", 0.5))
    mu_ref = float(getattr(config, "FRICTION_REF", 0.3))
    v0_ref = float(getattr(config, "IMPACT_VELOCITY_REF", 1.0))

    thresh = float(args.err_thresh_pct) / 100.0
    pass_peak = 0
    pass_l2 = 0
    evaluated = 0

    for k, c in enumerate(cases):
        if only is not None and k not in only:
            continue

        H = float(c.H)
        f = float(c.frac)
        t1 = H * f
        t2 = H - t1
        E1 = float(c.E1)
        E2 = float(c.E2)

        cfg = {
            "geometry": {"Lx": Lx, "Ly": Ly, "H": H},
            "mesh": {"ne_x": int(args.ne_x), "ne_y": int(args.ne_y), "ne_z": int(args.ne_z)},
            "layers": [
                {"t": float(t1), "E": float(E1), "nu": float(args.nu)},
                {"t": float(t2), "E": float(E2), "nu": float(args.nu)},
            ],
            "load_patch": {
                "pressure": float(args.p0),
                "x_start": x0 / Lx,
                "x_end": x1 / Lx,
                "y_start": y0 / Ly,
                "y_end": y1 / Ly,
            },
            "use_soft_mask": bool(int(args.use_soft_mask)),
        }

        x_nodes, y_nodes, z_nodes, u_fea = fem_solver.solve_fem(cfg)
        X, Y, Z = np.meshgrid(np.array(x_nodes), np.array(y_nodes), np.array(z_nodes), indexing="ij")
        u_true = np.array(u_fea, dtype=np.float32)

        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32, copy=False)
        params = np.array([E1, t1, E2, t2, r_ref, mu_ref, v0_ref], dtype=np.float32)[None, :]
        x_in = np.concatenate([pts, np.repeat(params, pts.shape[0], axis=0)], axis=1)

        with torch.no_grad():
            x_t = torch.tensor(x_in, dtype=torch.float32, device=device)
            v = pinn(x_t)
            u_pred = physics.decode_u(v, x_t).cpu().numpy().astype(np.float32).reshape(u_true.shape)

        uz_fea_top = u_true[:, :, -1, 2]
        uz_pinn_top = u_pred[:, :, -1, 2]
        peak_fea = float(np.min(uz_fea_top))
        peak_pinn = float(np.min(uz_pinn_top))
        peak_rel = abs(peak_pinn - peak_fea) / max(abs(peak_fea), 1e-12)
        l2_rel = _relative_l2(uz_pinn_top, uz_fea_top)
        pass_peak += int(peak_rel <= thresh)
        pass_l2 += int(l2_rel <= thresh)
        evaluated += 1
        print(
            f"case{k}: H={H:.4f} t1/T={f:.2f} E1={E1:.2f} E2={E2:.2f} | peak_rel={peak_rel*100:.2f}% l2_rel={l2_rel*100:.2f}%"
        )

        # --- Plots: top u_z ---
        vmin = float(min(uz_fea_top.min(), uz_pinn_top.min()))
        vmax = float(max(uz_fea_top.max(), uz_pinn_top.max()))
        err_top = np.abs(uz_pinn_top - uz_fea_top)
        err_max = float(err_top.max())

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        im0 = axes[0].contourf(X[:, :, 0], Y[:, :, 0], uz_fea_top, 50, cmap="jet", vmin=vmin, vmax=vmax)
        axes[0].set_title("FEA $u_z$ (top)")
        plt.colorbar(im0, ax=axes[0])
        im1 = axes[1].contourf(X[:, :, 0], Y[:, :, 0], uz_pinn_top, 50, cmap="jet", vmin=vmin, vmax=vmax)
        axes[1].set_title("PINN $u_z$ (top)")
        plt.colorbar(im1, ax=axes[1])
        im2 = axes[2].contourf(X[:, :, 0], Y[:, :, 0], err_top, 50, cmap="magma", vmin=0.0, vmax=err_max if err_max > 0 else 1.0)
        axes[2].set_title("|error| (top)")
        plt.colorbar(im2, ax=axes[2])
        for ax in axes:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "k-", lw=1.0)
        fig.suptitle(f"case{k}: peak_rel={peak_rel*100:.2f}%  l2_rel={l2_rel*100:.2f}%", fontsize=12)
        fig.tight_layout()
        out_top = os.path.join(args.out_dir, f"case{k}_top.png")
        fig.savefig(out_top, dpi=150)
        plt.close(fig)

        # --- Plots: XZ slice @ mid y ---
        mid_y = len(y_nodes) // 2
        Xs = X[:, mid_y, :]
        Zs = Z[:, mid_y, :]
        uz_fea_xz = u_true[:, mid_y, :, 2]
        uz_pinn_xz = u_pred[:, mid_y, :, 2]
        vmin = float(min(uz_fea_xz.min(), uz_pinn_xz.min()))
        vmax = float(max(uz_fea_xz.max(), uz_pinn_xz.max()))
        err_xz = np.abs(uz_pinn_xz - uz_fea_xz)
        err_max = float(err_xz.max())

        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
        im0 = axes2[0].contourf(Xs, Zs, uz_fea_xz, 50, cmap="jet", vmin=vmin, vmax=vmax)
        axes2[0].set_title("FEA $u_z$ (XZ)")
        plt.colorbar(im0, ax=axes2[0])
        im1 = axes2[1].contourf(Xs, Zs, uz_pinn_xz, 50, cmap="jet", vmin=vmin, vmax=vmax)
        axes2[1].set_title("PINN $u_z$ (XZ)")
        plt.colorbar(im1, ax=axes2[1])
        im2 = axes2[2].contourf(Xs, Zs, err_xz, 50, cmap="magma", vmin=0.0, vmax=err_max if err_max > 0 else 1.0)
        axes2[2].set_title("|error| (XZ)")
        plt.colorbar(im2, ax=axes2[2])
        for ax in axes2:
            ax.set_xlabel("x")
            ax.set_ylabel("z")
            ax.axhline(float(t1), color="k", lw=1.0)
        fig2.suptitle(f"case{k}: peak_rel={peak_rel*100:.2f}%  l2_rel={l2_rel*100:.2f}%", fontsize=12)
        fig2.tight_layout()
        out_xz = os.path.join(args.out_dir, f"case{k}_xz.png")
        fig2.savefig(out_xz, dpi=150)
        plt.close(fig2)

        print(f"  wrote: {out_top}")
        print(f"  wrote: {out_xz}")

    n = max(1, int(evaluated))
    print(f"\nPeak pass-rate: {pass_peak}/{n} ({(pass_peak/n)*100:.1f}%) at <= {args.err_thresh_pct:.1f}%")
    print(f"L2 pass-rate:   {pass_l2}/{n} ({(pass_l2/n)*100:.1f}%) at <= {args.err_thresh_pct:.1f}%")


if __name__ == "__main__":
    main()

