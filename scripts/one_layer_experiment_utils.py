"""Shared helpers for one-layer PINN/FEM experiment scripts."""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
ONE_LAYER_DIR = REPO_ROOT / "one-layer"
if not ONE_LAYER_DIR.exists():
    ONE_LAYER_DIR = REPO_ROOT / "one-layer-workflow"
FEA_DIR = REPO_ROOT / "fea-workflow" / "solver"
GRAPHS_DATA_DIR = REPO_ROOT / "graphs" / "data"
GRAPHS_FIG_DIR = REPO_ROOT / "graphs" / "figures"

for _path in (ONE_LAYER_DIR, FEA_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fem_solver  # noqa: E402
import model  # noqa: E402
import pinn_config as config  # noqa: E402


@dataclass(frozen=True)
class OneLayerCase:
    case_id: str
    E: float
    thickness: float


def ensure_output_dirs() -> None:
    GRAPHS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_FIG_DIR.mkdir(parents=True, exist_ok=True)


def select_device() -> torch.device:
    requested = os.getenv("PINN_DEVICE")
    if requested:
        return torch.device(requested)
    if os.getenv("PINN_FORCE_CPU", "0") == "1":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _adapt_state_dict(state: dict, target: dict) -> dict:
    state = dict(state)
    w_key = "layer.net.0.weight"
    if w_key not in state or w_key not in target:
        return state
    src_w = state[w_key]
    tgt_w = target[w_key]
    if src_w.shape == tgt_w.shape or src_w.shape[0] != tgt_w.shape[0]:
        return state
    if src_w.shape[1] == 8 and tgt_w.shape[1] == 11:
        adapted = torch.zeros_like(tgt_w)
        adapted[:, 0:5] = src_w[:, 0:5]
        adapted[:, 8:11] = src_w[:, 5:8]
        state[w_key] = adapted
    elif src_w.shape[1] == 10 and tgt_w.shape[1] == 11:
        adapted = torch.zeros_like(tgt_w)
        adapted[:, 0:7] = src_w[:, 0:7]
        adapted[:, 8:11] = src_w[:, 7:10]
        state[w_key] = adapted
    return state


def load_pinn(device: torch.device, model_path: str | os.PathLike | None = None):
    model_path = Path(model_path or os.getenv("PINN_MODEL_PATH") or ONE_LAYER_DIR / "pinn_model.pth")
    pinn = model.MultiLayerPINN().to(device)
    state = torch.load(str(model_path), map_location=device, weights_only=True)
    state = _adapt_state_dict(state, pinn.state_dict())
    pinn.load_state_dict(state, strict=False)
    pinn.eval()
    return pinn, model_path


def u_from_v(v: np.ndarray, pts: np.ndarray) -> np.ndarray:
    e_vals = pts[:, 3:4]
    t_vals = pts[:, 4:5]
    e_pow = float(getattr(config, "E_COMPLIANCE_POWER", 1.0))
    alpha = float(getattr(config, "THICKNESS_COMPLIANCE_ALPHA", 0.0))
    scale = float(getattr(config, "DISPLACEMENT_COMPLIANCE_SCALE", 1.0))
    h_ref = float(getattr(config, "H", 1.0))
    return scale * v / (e_vals**e_pow) * (h_ref / np.clip(t_vals, 1e-8, None)) ** alpha


def make_points(x: np.ndarray, y: np.ndarray, z: np.ndarray, case: OneLayerCase) -> np.ndarray:
    r_ref = float(getattr(config, "RESTITUTION_REF", 0.5))
    mu_ref = float(getattr(config, "FRICTION_REF", 0.3))
    v0_ref = float(getattr(config, "IMPACT_VELOCITY_REF", 1.0))
    return np.stack(
        [
            x,
            y,
            z,
            np.full_like(x, case.E, dtype=float),
            np.full_like(x, case.thickness, dtype=float),
            np.full_like(x, r_ref, dtype=float),
            np.full_like(x, mu_ref, dtype=float),
            np.full_like(x, v0_ref, dtype=float),
        ],
        axis=1,
    )


def predict_displacement(pinn, device: torch.device, pts: np.ndarray, batch_size: int = 32768) -> np.ndarray:
    out = []
    with torch.no_grad():
        for start in range(0, len(pts), batch_size):
            batch_pts = pts[start : start + batch_size]
            v = pinn(torch.tensor(batch_pts, dtype=torch.float32, device=device)).detach().cpu().numpy()
            out.append(u_from_v(v, batch_pts))
    return np.concatenate(out, axis=0)


def fem_cfg(case: OneLayerCase, ne_x: int, ne_y: int, ne_z: int) -> dict:
    return {
        "geometry": {"Lx": config.Lx, "Ly": config.Ly, "H": case.thickness, "ne_x": ne_x, "ne_y": ne_y, "ne_z": ne_z},
        "material": {"E": case.E, "nu": config.nu_vals[0]},
        "load_patch": {
            "pressure": config.p0,
            "x_start": config.LOAD_PATCH_X[0] / config.Lx,
            "x_end": config.LOAD_PATCH_X[1] / config.Lx,
            "y_start": config.LOAD_PATCH_Y[0] / config.Ly,
            "y_end": config.LOAD_PATCH_Y[1] / config.Ly,
        },
    }


def solve_fem_case(case: OneLayerCase, ne_x: int, ne_y: int, ne_z: int):
    start = time.perf_counter()
    x_nodes, y_nodes, z_nodes, u_grid = fem_solver.solve_fem(fem_cfg(case, ne_x, ne_y, ne_z))
    elapsed = time.perf_counter() - start
    return np.asarray(x_nodes), np.asarray(y_nodes), np.asarray(z_nodes), np.asarray(u_grid), elapsed


def mae_pct(pred: np.ndarray, ref: np.ndarray) -> float:
    denom = float(np.max(np.abs(ref)))
    return 100.0 * float(np.mean(np.abs(pred - ref))) / denom if denom > 0 else 0.0


def max_pct(pred: np.ndarray, ref: np.ndarray) -> float:
    denom = float(np.max(np.abs(ref)))
    return 100.0 * float(np.max(np.abs(pred - ref))) / denom if denom > 0 else 0.0


def evaluate_case_grid(pinn, device: torch.device, case: OneLayerCase, ne_x: int, ne_y: int, ne_z: int) -> dict:
    x_nodes, y_nodes, z_nodes, u_fem, fem_seconds = solve_fem_case(case, ne_x, ne_y, ne_z)
    xg, yg, zg = np.meshgrid(x_nodes, y_nodes, z_nodes, indexing="ij")
    pts = make_points(xg.ravel(), yg.ravel(), zg.ravel(), case)
    start = time.perf_counter()
    u_pinn = predict_displacement(pinn, device, pts).reshape(u_fem.shape)
    pinn_eval_seconds = time.perf_counter() - start
    top_pred = u_pinn[:, :, -1, 2]
    top_ref = u_fem[:, :, -1, 2]
    return {
        "case": case,
        "x_nodes": x_nodes,
        "y_nodes": y_nodes,
        "z_nodes": z_nodes,
        "u_fem": u_fem,
        "u_pinn": u_pinn,
        "fem_seconds": fem_seconds,
        "pinn_eval_seconds": pinn_eval_seconds,
        "top_uz_mae_pct": mae_pct(top_pred, top_ref),
        "top_uz_max_pct": max_pct(top_pred, top_ref),
        "volume_mae_pct": mae_pct(u_pinn, u_fem),
        "volume_max_pct": max_pct(u_pinn, u_fem),
        "peak_fem_uz": float(np.min(top_ref)),
        "peak_pinn_uz": float(np.min(top_pred)),
        "n_eval_points": int(u_fem.shape[0] * u_fem.shape[1] * u_fem.shape[2]),
    }


def random_interior_cases(n_cases: int, seed: int) -> list[OneLayerCase]:
    rng = np.random.default_rng(seed)
    e_min, e_max = [float(v) for v in getattr(config, "E_RANGE", [1.0, 10.0])]
    t_min, t_max = [float(v) for v in getattr(config, "THICKNESS_RANGE", [0.05, 0.15])]
    cases = []
    for idx in range(n_cases):
        E, thickness = rng.uniform([e_min, t_min], [e_max, t_max])
        cases.append(OneLayerCase(f"one_layer_random_{idx:03d}", float(E), float(thickness)))
    return cases


def supervised_parameter_grid() -> set[tuple[float, float]]:
    out = set()
    for E in getattr(config, "DATA_E_VALUES", []):
        for thickness in getattr(config, "DATA_THICKNESS_VALUES", []):
            out.add((round(float(E), 10), round(float(thickness), 10)))
    return out


def is_supervised_parameter_case(case: OneLayerCase) -> bool:
    return (round(float(case.E), 10), round(float(case.thickness), 10)) in supervised_parameter_grid()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def rows_to_csv(path: Path, fieldnames: Iterable[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
