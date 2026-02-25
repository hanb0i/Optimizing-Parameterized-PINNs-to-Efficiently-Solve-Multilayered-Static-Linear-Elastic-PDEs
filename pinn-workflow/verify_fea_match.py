import argparse
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PINN_WORKFLOW_DIR = os.path.join(REPO_ROOT, "pinn-workflow")
if PINN_WORKFLOW_DIR not in sys.path:
    sys.path.insert(0, PINN_WORKFLOW_DIR)

import pinn_config as config
import model
import physics


def _infer_arch_from_state_dict(sd: dict) -> tuple[int, int]:
    """
    Infer (hidden_layers, hidden_units) from a LayerNet state dict.
    Expects keys like: layers.0.net.0.weight (hidden_units, 15)
    and final: layers.0.net.{last}.weight (3, hidden_units).
    """
    w0 = sd.get("layers.0.net.0.weight", None)
    if w0 is None:
        return int(getattr(config, "LAYERS", 4)), int(getattr(config, "NEURONS", 64))
    hidden_units = int(w0.shape[0])
    # Count Linear layers inside net by scanning indices.
    # In nn.Sequential, Linear weights appear at even indices: 0,2,4,...,2*hidden_layers
    linear_indices = set()
    for k in sd.keys():
        if not k.startswith("layers.0.net.") or not k.endswith(".weight"):
            continue
        try:
            idx = int(k.split(".")[3])
        except Exception:
            continue
        linear_indices.add(idx)
    if not linear_indices:
        return int(getattr(config, "LAYERS", 4)), hidden_units
    n_linear = len(linear_indices)
    # n_linear = hidden_layers + 1 => hidden_layers = n_linear - 1
    hidden_layers = max(1, int(n_linear - 1))
    return hidden_layers, hidden_units


def _build_inputs_from_fea(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    *,
    E1: float,
    t1: float,
    E2: float,
    t2: float,
    E3: float,
    t3: float,
    restitution: float,
    friction: float,
    impact_velocity: float,
) -> np.ndarray:
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    n = pts.shape[0]
    out = np.zeros((n, 12), dtype=np.float32)
    out[:, 0:3] = pts.astype(np.float32)
    out[:, 3] = float(E1)
    out[:, 4] = float(t1)
    out[:, 5] = float(E2)
    out[:, 6] = float(t2)
    out[:, 7] = float(E3)
    out[:, 8] = float(t3)
    out[:, 9] = float(restitution)
    out[:, 10] = float(friction)
    out[:, 11] = float(impact_velocity)
    return out


def _metrics(u_pred: np.ndarray, u_true: np.ndarray) -> dict:
    err = u_pred - u_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "max_abs": float(np.max(np.abs(err))),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate trained 3-layer PiNN against fea_solution.npy (no supervision).")
    ap.add_argument("--fea", default="fea_solution.npy", help="Path to FEA npy (dict with x,y,z,u).")
    ap.add_argument("--model", default="pinn_model.pth", help="Path to trained PINN checkpoint.")
    ap.add_argument("--layers", type=int, default=None, help="Override hidden layers (default: infer from checkpoint).")
    ap.add_argument("--neurons", type=int, default=None, help="Override hidden units (default: infer from checkpoint).")
    ap.add_argument(
        "--hard_clamp_sides",
        type=int,
        default=None,
        help="If set, overrides pinn_config.HARD_CLAMP_SIDES for evaluation (0/1).",
    )

    ap.add_argument("--E1", type=float, default=1.0)
    ap.add_argument("--E2", type=float, default=1.0)
    ap.add_argument("--E3", type=float, default=1.0)
    ap.add_argument("--t1", type=float, default=float(getattr(config, "H", 0.1)) / 3.0)
    ap.add_argument("--t2", type=float, default=float(getattr(config, "H", 0.1)) / 3.0)
    ap.add_argument("--t3", type=float, default=float(getattr(config, "H", 0.1)) / 3.0)
    ap.add_argument("--r", type=float, default=float(getattr(config, "RESTITUTION_REF", 0.5)))
    ap.add_argument("--mu", type=float, default=float(getattr(config, "FRICTION_REF", 0.3)))
    ap.add_argument("--v0", type=float, default=float(getattr(config, "IMPACT_VELOCITY_REF", 1.0)))
    args = ap.parse_args()

    if not os.path.exists(args.fea):
        raise FileNotFoundError(args.fea)
    if not os.path.exists(args.model):
        raise FileNotFoundError(args.model)

    sd = torch.load(args.model, map_location="cpu", weights_only=False)
    inferred_layers, inferred_neurons = _infer_arch_from_state_dict(sd)
    config.LAYERS = int(args.layers) if args.layers is not None else int(inferred_layers)
    config.NEURONS = int(args.neurons) if args.neurons is not None else int(inferred_neurons)
    if args.hard_clamp_sides is not None:
        config.HARD_CLAMP_SIDES = bool(int(args.hard_clamp_sides))

    fem = np.load(args.fea, allow_pickle=True).item()
    X, Y, Z, U = fem["x"], fem["y"], fem["z"], fem["u"]
    u_true = U.reshape(-1, 3).astype(np.float64)

    t_total = float(args.t1 + args.t2 + args.t3)
    u_in = _build_inputs_from_fea(
        X,
        Y,
        Z,
        E1=args.E1,
        t1=args.t1,
        E2=args.E2,
        t2=args.t2,
        E3=args.E3,
        t3=args.t3,
        restitution=args.r,
        friction=args.mu,
        impact_velocity=args.v0,
    )

    device = torch.device("cpu")
    pinn = model.MultiLayerPINN().to(device)
    pinn.load_state_dict(sd, strict=False)
    pinn.eval()

    with torch.no_grad():
        x_tensor = torch.tensor(u_in, dtype=torch.float32, device=device)
        v = pinn(x_tensor)
        u_pred = physics.decode_u(v, x_tensor).cpu().numpy().astype(np.float64)

    m_all = _metrics(u_pred, u_true)

    H = float(getattr(config, "H", t_total))
    z = Z.ravel().astype(np.float64)
    top = np.isclose(z, H)
    x = X.ravel()
    y = Y.ravel()
    x0, x1 = map(float, config.LOAD_PATCH_X)
    y0, y1 = map(float, config.LOAD_PATCH_Y)
    patch = top & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    free_top = top & (~patch)

    def _maybe(mask: np.ndarray, label: str) -> None:
        if int(mask.sum()) == 0:
            print(f"{label}: n=0")
            return
        mm = _metrics(u_pred[mask], u_true[mask])
        uz_p = u_pred[mask, 2]
        uz_t = u_true[mask, 2]
        print(
            f"{label}: n={int(mask.sum())} mae={mm['mae']:.6f} rmse={mm['rmse']:.6f} max={mm['max_abs']:.6f} "
            f"| uz_pred(min/mean/max)=({uz_p.min():.4f},{uz_p.mean():.4f},{uz_p.max():.4f}) "
            f"uz_fea(min/mean/max)=({uz_t.min():.4f},{uz_t.mean():.4f},{uz_t.max():.4f})"
        )

    print(f"ALL: mae={m_all['mae']:.6f} rmse={m_all['rmse']:.6f} max={m_all['max_abs']:.6f}")
    _maybe(top, "TOP")
    _maybe(patch, "TOP_PATCH")
    _maybe(free_top, "TOP_FREE")


if __name__ == "__main__":
    main()
