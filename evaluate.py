"""
Phase 7 — Physics validation, repeated impacts with damage, and active-learning loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from config import Bounds, DamageConfig, PipelineConfig
from dataset import generate_dataset, mu_dict_to_tensor, mu_tensor_to_dict, sample_mu
from metrics import MetricsConfig, compute_metrics
from surrogate import SurrogateBundle, train_surrogate


MuType = Union[torch.Tensor, Mapping[str, float]]
PhysicsRunner = Callable[[MuType], Mapping[str, Any]]


def evaluate_mu(mu: MuType, physics_runner: PhysicsRunner, metrics_cfg: MetricsConfig) -> Dict[str, float]:
    sim_out = physics_runner(mu)
    metrics = compute_metrics(sim_out, metrics_cfg)
    return {k: float(v.detach().cpu().item()) for k, v in metrics.items()}


def compare_designs(
    baseline_mu: MuType,
    candidate_mus: Sequence[MuType],
    physics_runner: PhysicsRunner,
    metrics_cfg: MetricsConfig,
    *,
    keys: Sequence[str] = ("y_strain_energy", "y_accel_peak", "y_disp_peak"),
) -> List[Dict[str, Any]]:
    base = evaluate_mu(baseline_mu, physics_runner, metrics_cfg)
    rows: List[Dict[str, Any]] = []
    for i, mu in enumerate(candidate_mus):
        m = evaluate_mu(mu, physics_runner, metrics_cfg)
        row: Dict[str, Any] = {"candidate": i}
        for k in keys:
            row[f"baseline_{k}"] = base.get(k)
            row[f"candidate_{k}"] = m.get(k)
        rows.append(row)
    return rows


@dataclass
class DamageState:
    D: float = 0.0


def apply_damage_to_mu(mu: MuType, bounds: Bounds, dcfg: DamageConfig, state: DamageState) -> MuType:
    """
    Default damage model: scale any configured "modulus" parameters by (1 - clamp(D)).

    This is intentionally modular because mu semantics are application-specific.
    """

    if not dcfg.modulus_param_names:
        return mu

    scale = 1.0 - float(min(max(state.D, 0.0), float(dcfg.d_max))) * float(dcfg.modulus_scale_max)
    scale = float(max(scale, 0.0))

    if isinstance(mu, dict):
        mu2 = dict(mu)
        for k in dcfg.modulus_param_names:
            if k in mu2:
                mu2[k] = float(mu2[k]) * scale
        return mu2

    mu_t = mu.clone()
    for k in dcfg.modulus_param_names:
        if k in bounds.names:
            idx = bounds.names.index(k)
            mu_t[idx] = mu_t[idx] * scale
    return mu_t


def run_repeated_impacts(
    mu0: MuType,
    bounds: Bounds,
    physics_runner: PhysicsRunner,
    metrics_cfg: MetricsConfig,
    dcfg: DamageConfig,
) -> List[Dict[str, Any]]:
    """
    Sequential impacts with accumulated smooth damage:
      D_{k+1} = D_k + ∫∫ softplus(ε-ε_crit) dx dt
    """

    if not dcfg.enabled:
        raise ValueError("DamageConfig.enabled is False.")

    state = DamageState(D=0.0)
    records: List[Dict[str, Any]] = []

    for k in range(int(dcfg.n_impacts)):
        mu_k = apply_damage_to_mu(mu0, bounds, dcfg, state)

        sim_out = physics_runner(mu_k)
        mcfg = MetricsConfig(
            absorb_mask=metrics_cfg.absorb_mask,
            t_max=metrics_cfg.t_max,
            accel_smooth_window=metrics_cfg.accel_smooth_window,
            include_optional=True,
            include_damage=True,
            damage_eps_crit=float(dcfg.eps_crit),
            damage_softplus_beta=float(dcfg.softplus_beta),
            strain_fn=metrics_cfg.strain_fn,
            protected_index=metrics_cfg.protected_index,
        )
        metrics = compute_metrics(sim_out, mcfg)

        D_inc = float(metrics.get("D_damage", torch.tensor(0.0)).detach().cpu().item())
        state.D += D_inc
        records.append(
            {
                "impact": k,
                "D_inc": D_inc,
                "D_total": state.D,
                "y_strain_energy": float(metrics["y_strain_energy"].detach().cpu().item()),
                "y_accel_peak": float(metrics["y_accel_peak"].detach().cpu().item()),
                "y_disp_peak": float(metrics["y_disp_peak"].detach().cpu().item()),
            }
        )

    return records


def active_learning_refine(
    dataset_n: int,
    bounds: Bounds,
    physics_runner: PhysicsRunner,
    cfg: PipelineConfig,
    candidate_mus: Sequence[Mapping[str, float]],
    surrogate: SurrogateBundle,
) -> Optional[SurrogateBundle]:
    """
    If surrogate discrepancies are large on evaluated candidates, add new points and retrain.
    """

    if not cfg.active_learning.enabled:
        return None

    target_names = ["y_strain_energy", "y_accel_peak", "y_disp_peak"]

    bad_points: List[torch.Tensor] = []
    bad_targets: List[torch.Tensor] = []

    for mu_d in candidate_mus:
        true_metrics = evaluate_mu(mu_d, physics_runner, cfg.metrics)
        y_true = torch.tensor([true_metrics[k] for k in target_names], dtype=torch.float32)
        y_pred = surrogate.predict_raw(mu_d).cpu()

        rel = (y_pred - y_true).abs() / (y_true.abs().clamp_min(float(cfg.active_learning.relerr_floor)))
        if float(rel.max().item()) > float(cfg.active_learning.discrepancy_tol):
            bad_points.append(mu_dict_to_tensor(mu_d, bounds).cpu())
            bad_targets.append(y_true.cpu())

    if not bad_points:
        return None

    n_add = min(int(cfg.active_learning.add_points_per_iter), len(bad_points))
    x_add = torch.stack(bad_points[:n_add], dim=0)
    y_add = torch.stack(bad_targets[:n_add], dim=0)

    base_ds = generate_dataset(dataset_n, bounds, physics_runner, cfg, seed=cfg.surrogate.seed, target_names=target_names)
    x_raw = torch.cat([base_ds.x_raw, x_add], dim=0)
    y_raw = torch.cat([base_ds.y_raw, y_add], dim=0)

    x_min = x_raw.min(dim=0).values
    x_max = x_raw.max(dim=0).values
    y_min = y_raw.min(dim=0).values
    y_max = y_raw.max(dim=0).values

    # Reuse split sizes from the newly created base dataset
    from dataset import build_supervised_dataset

    ds2 = build_supervised_dataset(
        param_names=base_ds.param_names,
        target_names=base_ds.target_names,
        x_raw=x_raw,
        y_raw=y_raw,
        train_frac=cfg.surrogate.train_frac,
        val_frac=cfg.surrogate.val_frac,
        seed=int(cfg.surrogate.seed),
    )

    refined, _, _ = train_surrogate(ds2, bounds, cfg)
    return refined


def discrepancy_report(
    surrogate: SurrogateBundle,
    mus: Sequence[Mapping[str, float]],
    physics_runner: PhysicsRunner,
    metrics_cfg: MetricsConfig,
    *,
    target_names: Sequence[str] = ("y_strain_energy", "y_accel_peak", "y_disp_peak"),
    relerr_floor: float = 1e-6,
) -> Dict[str, Any]:
    """
    Evaluate physics vs surrogate on a set of mu dicts and return summary stats.
    """

    if not mus:
        raise ValueError("mus is empty")

    y_true_rows = []
    y_pred_rows = []
    for mu in mus:
        m = evaluate_mu(mu, physics_runner, metrics_cfg)
        y_true_rows.append(torch.tensor([m[k] for k in target_names], dtype=torch.float32))
        y_pred_rows.append(surrogate.predict_raw(mu).cpu().to(dtype=torch.float32))

    y_true = torch.stack(y_true_rows, dim=0)
    y_pred = torch.stack(y_pred_rows, dim=0)
    rel = (y_pred - y_true).abs() / (y_true.abs().clamp_min(float(relerr_floor)))

    out: Dict[str, Any] = {
        "max_rel": float(rel.max().item()),
        "mean_rel": float(rel.mean().item()),
        "per_target_max_rel": {str(target_names[j]): float(rel[:, j].max().item()) for j in range(rel.shape[1])},
        "per_target_mean_rel": {str(target_names[j]): float(rel[:, j].mean().item()) for j in range(rel.shape[1])},
    }
    return out


def iterative_active_learning(
    *,
    bounds: Bounds,
    physics_runner: PhysicsRunner,
    cfg: PipelineConfig,
    candidates: Sequence[Mapping[str, float]],
    initial_surrogate: SurrogateBundle,
    dataset_n: int,
) -> Tuple[SurrogateBundle, List[Dict[str, Any]]]:
    """
    Run multiple refine iterations until candidate discrepancies fall below tolerance.
    """

    logs: List[Dict[str, Any]] = []
    surrogate = initial_surrogate

    for it in range(int(cfg.active_learning.iters)):
        eval_mus: List[Mapping[str, float]] = list(candidates)
        n_rand = int(cfg.active_learning.eval_random_points)
        if n_rand > 0:
            x_rand = sample_mu(n_rand, bounds, seed=int(cfg.surrogate.seed + 1000 + it)).cpu()
            eval_mus.extend([mu_tensor_to_dict(x_rand[i], list(bounds.names)) for i in range(x_rand.shape[0])])

        rep = discrepancy_report(
            surrogate,
            eval_mus,
            physics_runner,
            cfg.metrics,
            relerr_floor=float(cfg.active_learning.relerr_floor),
        )
        rep["iter"] = it
        logs.append(rep)
        if rep["max_rel"] <= float(cfg.active_learning.discrepancy_tol):
            break
        refined = active_learning_refine(dataset_n, bounds, physics_runner, cfg, candidates, surrogate)
        if refined is None:
            break
        surrogate = refined

    return surrogate, logs
