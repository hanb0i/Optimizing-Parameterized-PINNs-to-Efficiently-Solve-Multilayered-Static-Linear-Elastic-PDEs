from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch

sys.dont_write_bytecode = True

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
    ap = argparse.ArgumentParser(description="Failure map: where PINN disagrees with layered FEA.")
    ap.add_argument("--model", required=True, help="PINN checkpoint path.")
    ap.add_argument("--device", default=None, help="cpu|cuda|mps (auto if omitted).")
    ap.add_argument("--out", default="failure_map.png", help="Output PNG path.")
    ap.add_argument("--save_metrics", default=None, help="Optional JSON path to save computed metrics per case.")

    ap.add_argument("--cases_json", default=None, help="Optional JSON list of cases [{H, frac, E1, E2}, ...].")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cases", type=int, default=20, help="Number of random cases if --cases_json is not set.")
    ap.add_argument("--err_thresh_pct", type=float, default=5.0)
    ap.add_argument("--vmax_pct", type=float, default=40.0, help="Color scale max (percent).")

    ap.add_argument("--ne_x", type=int, default=12)
    ap.add_argument("--ne_y", type=int, default=12)
    ap.add_argument("--ne_z", type=int, default=12)
    ap.add_argument("--nu", type=float, default=float(getattr(config, "NU_FIXED", 0.3)))
    ap.add_argument("--p0", type=float, default=float(getattr(config, "p0", 1.0)))
    ap.add_argument("--use_soft_mask", type=int, default=int(getattr(config, "USE_SOFT_LOAD_MASK", True)))
    args = ap.parse_args()

    if int(getattr(config, "NUM_LAYERS", 2)) != 2:
        raise ValueError(f"Expected NUM_LAYERS=2 for this script (got {getattr(config, 'NUM_LAYERS', None)}).")

    import matplotlib

    if os.environ.get("MPLBACKEND") is None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    device = _select_device(args.device)
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model!r}")
    sd = torch.load(args.model, map_location=device, weights_only=False)
    inferred_layers, inferred_neurons = _infer_arch_from_state_dict(sd)
    config.LAYERS = int(inferred_layers)
    config.NEURONS = int(inferred_neurons)
    pinn = model.MultiLayerPINN().to(device)
    pinn.load_state_dict(sd, strict=False)
    pinn.eval()
    print(f"Loaded model: {args.model} (layers={config.LAYERS}, neurons={config.NEURONS})")

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
    frac_list: list[float] = []
    log_ratio_list: list[float] = []
    peak_pct_list: list[float] = []
    l2_pct_list: list[float] = []

    metrics_rows: list[dict] = []

    for k, c in enumerate(cases):
        H = float(c.H)
        frac = float(c.frac)
        t1 = H * frac
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

        frac_list.append(frac)
        log_ratio_list.append(float(np.log10(max(E1, 1e-12) / max(E2, 1e-12))))
        peak_pct_list.append(float(peak_rel * 100.0))
        l2_pct_list.append(float(l2_rel * 100.0))

        metrics_rows.append(
            {
                "case": int(k),
                "H": float(H),
                "frac": float(frac),
                "E1": float(E1),
                "E2": float(E2),
                "peak_rel_pct": float(peak_rel * 100.0),
                "l2_rel_pct": float(l2_rel * 100.0),
                "pass_peak": bool(peak_rel <= thresh),
                "pass_l2": bool(l2_rel <= thresh),
            }
        )
        print(
            f"case{k}: frac={frac:.2f} log10(E1/E2)={log_ratio_list[-1]:+.2f} | peak={peak_pct_list[-1]:.2f}% l2={l2_pct_list[-1]:.2f}%"
        )

    if args.save_metrics:
        with open(args.save_metrics, "w", encoding="utf-8") as f:
            json.dump(metrics_rows, f, indent=2, sort_keys=True)
        print(f"Wrote metrics: {args.save_metrics}")

    frac_arr = np.array(frac_list, dtype=float)
    log_ratio_arr = np.array(log_ratio_list, dtype=float)
    peak_pct = np.array(peak_pct_list, dtype=float)
    l2_pct = np.array(l2_pct_list, dtype=float)

    pass_peak = peak_pct <= float(args.err_thresh_pct)
    pass_l2 = l2_pct <= float(args.err_thresh_pct)

    vmax = max(1.0, float(args.vmax_pct))
    norm = Normalize(vmin=0.0, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    for ax, metric_pct, pass_mask, title in [
        (axes[0], peak_pct, pass_peak, "Peak error (%)"),
        (axes[1], l2_pct, pass_l2, "Relative L2 error (%)"),
    ]:
        sc = ax.scatter(
            frac_arr,
            log_ratio_arr,
            c=np.clip(metric_pct, 0.0, vmax),
            cmap="magma",
            norm=norm,
            s=90,
            edgecolors=np.where(pass_mask, "lime", "cyan"),
            linewidths=1.2,
        )
        for i in range(len(frac_arr)):
            ax.annotate(str(i), (frac_arr[i], log_ratio_arr[i]), fontsize=8, xytext=(4, 2), textcoords="offset points")
        ax.set_title(title)
        ax.set_xlabel("t1/T (fraction of total thickness)")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("log10(E1/E2)")
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
    cbar.set_label("error (%) (clipped)")

    model_tag = os.path.basename(args.model)
    fig.suptitle(
        f"Failure map (threshold={args.err_thresh_pct:.1f}%) | {model_tag} | n={len(cases)}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=170)
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()

