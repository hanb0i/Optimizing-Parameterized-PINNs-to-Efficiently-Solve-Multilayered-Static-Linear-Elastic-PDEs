from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from glob import glob
from typing import Iterable

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PINN_WORKFLOW_DIR = os.path.join(REPO_ROOT, "pinn-workflow")
if PINN_WORKFLOW_DIR not in sys.path:
    sys.path.insert(0, PINN_WORKFLOW_DIR)

import pinn_config as config
import model
import physics


def _infer_arch_from_state_dict(sd: dict) -> tuple[int, int]:
    # neurons from first linear layer
    first = sd.get("layers.0.net.0.weight", None)
    if first is None or not hasattr(first, "shape"):
        raise ValueError("Could not infer architecture (missing layers.0.net.0.weight).")
    neurons = int(first.shape[0])

    # hidden layer count from number of Linear layers minus output layer
    linear_indices = set()
    for k in sd.keys():
        if not isinstance(k, str):
            continue
        if not k.startswith("layers.0.net.") or not k.endswith(".weight"):
            continue
        parts = k.split(".")
        if len(parts) >= 4 and parts[3].isdigit():
            linear_indices.add(int(parts[3]))
    num_linears = len(sorted(linear_indices))
    hidden_layers = max(1, num_linears - 1)
    return hidden_layers, neurons


@dataclass(frozen=True)
class Metrics:
    global_mae_pct: float
    top_uz_mae_pct: float
    patch_top_uz_mae_pct: float
    peak_uz_top: float

    @property
    def score(self) -> float:
        # Minimize the worst of the two user-facing metrics.
        return float(max(self.global_mae_pct, self.patch_top_uz_mae_pct))


def _compute_metrics(
    pinn: torch.nn.Module,
    pts12: torch.Tensor,
    U_fea: np.ndarray,
    X_fea: np.ndarray,
    Y_fea: np.ndarray,
) -> Metrics:
    with torch.no_grad():
        v = pinn(pts12)
        u_flat = physics.decode_u(v, pts12).cpu().numpy()
    U_pinn = u_flat.reshape(U_fea.shape)

    err_all = (U_pinn - U_fea).reshape(-1, 3)
    mae_all = float(np.mean(np.abs(err_all)))
    denom_all = float(np.max(np.abs(U_fea)))
    global_mae_pct = (mae_all / denom_all) * 100.0 if denom_all > 0 else float("nan")

    uz_fea_top = U_fea[:, :, -1, 2]
    uz_top = U_pinn[:, :, -1, 2]
    abs_diff = np.abs(uz_fea_top - uz_top)
    denom_top = float(np.max(np.abs(uz_fea_top)))
    top_uz_mae_pct = (float(np.mean(abs_diff)) / denom_top) * 100.0 if denom_top > 0 else float("nan")

    x0, x1 = map(float, getattr(config, "LOAD_PATCH_X", [1.0 / 3.0, 2.0 / 3.0]))
    y0, y1 = map(float, getattr(config, "LOAD_PATCH_Y", [1.0 / 3.0, 2.0 / 3.0]))
    X0 = X_fea[:, :, 0]
    Y0 = Y_fea[:, :, 0]
    patch = (X0 >= x0) & (X0 <= x1) & (Y0 >= y0) & (Y0 <= y1)
    if np.any(patch) and denom_top > 0:
        patch_top_uz_mae_pct = (float(np.mean(abs_diff[patch])) / denom_top) * 100.0
    else:
        patch_top_uz_mae_pct = float("nan")

    return Metrics(
        global_mae_pct=global_mae_pct,
        top_uz_mae_pct=top_uz_mae_pct,
        patch_top_uz_mae_pct=patch_top_uz_mae_pct,
        peak_uz_top=float(np.min(uz_top)),
    )


def _iter_checkpoints(patterns: Iterable[str]) -> list[str]:
    out: list[str] = []
    for pat in patterns:
        out.extend(glob(pat))
    out = [p for p in out if os.path.isfile(p)]
    out.sort()
    return out


def main():
    fea_path = os.environ.get("FEA_NPY", "fea_solution_layered_E1_1_E2_5_E3_10_equal.npy")
    if not os.path.exists(fea_path):
        raise FileNotFoundError(f"FEA file not found: {fea_path}")

    # Ensure eval is consistent: hard side BC on with fully-hard mask.
    config.USE_HARD_SIDE_BC = True
    config.HARD_SIDE_BC_POWER = 1.0

    # Fixed layered parameters (match file naming).
    h = float(getattr(config, "H", 0.1))
    E1, E2, E3 = 1.0, 5.0, 10.0
    t1 = t2 = t3 = h / 3.0
    r = float(getattr(config, "RESTITUTION_REF", 0.5))
    mu = float(getattr(config, "FRICTION_REF", 0.3))
    v0 = float(getattr(config, "IMPACT_VELOCITY_REF", 1.0))

    fea = np.load(fea_path, allow_pickle=True).item()
    X_fea = fea["x"]
    Y_fea = fea["y"]
    Z_fea = fea["z"]
    U_fea = fea["u"]

    pts = np.stack([X_fea.ravel(), Y_fea.ravel(), Z_fea.ravel()], axis=1).astype(np.float32)
    params = np.array([E1, t1, E2, t2, E3, t3, r, mu, v0], dtype=np.float32)[None, :]
    params = np.repeat(params, repeats=pts.shape[0], axis=0)
    pts12_np = np.concatenate([pts, params], axis=1)
    pts12 = torch.tensor(pts12_np, dtype=torch.float32, device="cpu")

    patterns = sys.argv[1:] if len(sys.argv) > 1 else ["*.pth"]
    checkpoints = _iter_checkpoints(patterns)
    if not checkpoints:
        print("No checkpoints found.")
        return

    rows = []
    for path in checkpoints:
        try:
            sd = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            sd = torch.load(path, map_location="cpu")
        try:
            layers, neurons = _infer_arch_from_state_dict(sd)
        except Exception:
            continue

        config.LAYERS = int(layers)
        config.NEURONS = int(neurons)
        pinn = model.MultiLayerPINN().cpu()
        try:
            pinn.load_state_dict(sd, strict=True)
        except Exception:
            continue
        pinn.eval()

        m = _compute_metrics(pinn, pts12, U_fea, X_fea, Y_fea)
        rows.append((path, m))

    rows.sort(key=lambda x: (x[1].score, x[1].global_mae_pct, x[1].patch_top_uz_mae_pct))
    print(f"Evaluated {len(rows)} checkpoints on {os.path.basename(fea_path)}.")
    print("Top 10 by score=max(global_MAE%, patch_top_u_z_MAE%):")
    for path, m in rows[:10]:
        print(
            f"{path}  score={m.score:6.2f}  global={m.global_mae_pct:6.2f}%  "
            f"top_u_z={m.top_uz_mae_pct:6.2f}%  patch_top={m.patch_top_uz_mae_pct:6.2f}%  peak={m.peak_uz_top:7.3f}"
        )


if __name__ == "__main__":
    main()

