"""End-to-end pipeline: Surrogate training, optimization, and evaluation."""

import argparse
import os
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import torch

from config import Bounds, PipelineConfig
from dataset import generate_dataset, mu_dict_to_tensor, mu_tensor_to_dict, sample_mu
from evaluate import (
    active_learning_refine,
    compare_designs,
    discrepancy_report,
    evaluate_mu,
    iterative_active_learning,
    run_repeated_impacts,
)
from metrics import MetricsConfig
from optimize import optimize_design
from surrogate import train_surrogate


def _physics_runner_mock():
    class _Runner:
        def __call__(self, mu):
            if isinstance(mu, Mapping):
                mu_t = torch.tensor(list(mu.values()), dtype=torch.float32).view(1, -1)
            else:
                mu_t = mu if mu.dim() > 1 else mu.unsqueeze(0)

            mu_np = mu_t.detach().cpu().numpy().flatten()
            e1, e2, e3, t1, t2, t3 = mu_np[:6]
            r, mu_fric, v0 = mu_np[6:9] if len(mu_np) >= 9 else (0.5, 0.3, 1.0)

            thickness = t1 + t2 + t3
            scale = 1.0 / (1.0 + 0.5 * thickness)
            strain_energy = 10.0 / (e1 + 0.1) * scale + 5.0 / (e2 + 0.1) * scale + 2.5 / (e3 + 0.1) * scale
            accel_peak = 20.0 / (t1 + 0.01) + 10.0 / (t2 + 0.01) + 5.0 / (t3 + 0.01)
            accel_peak += (1.0 - r) * 5.0 + (1.0 - mu_fric) * 3.0 + v0 * 2.0
            disp_peak = 0.5 * strain_energy + 0.1 * accel_peak

            t = np.linspace(0.0, 1.0, 101)
            u_array = disp_peak * np.sin(np.pi * t)
            eps_array = 0.1 * np.sin(2 * np.pi * t)
            accel_array = accel_peak * np.sin(3 * np.pi * t)

            return {
                "t": torch.tensor(t, dtype=torch.float32),
                "u_array": torch.tensor(u_array, dtype=torch.float32),
                "strain": torch.tensor(eps_array, dtype=torch.float32),
                "accel_array": torch.tensor(accel_array, dtype=torch.float32),
                "u_final": torch.tensor(u_array[-1], dtype=torch.float32).unsqueeze(0),
                "accel_final": torch.tensor(accel_peak, dtype=torch.float32).unsqueeze(0),
            }

    return _Runner()


def _maybe_load_fea_injection():
    fea_path = os.environ.get("FEA_SOLUTION_NPY", "")
    if fea_path and os.path.isfile(fea_path):
        data = np.load(fea_path, allow_pickle=True).item()
        u = data.get("u", data.get("displacement", 0.0))
        strain = data.get("strain", data.get("eps", 0.0))
        accel = data.get("accel", data.get("acceleration", 0.0))
        if isinstance(u, (list, tuple)):
            u = np.array(u)
        if isinstance(strain, (list, tuple)):
            strain = np.array(strain)
        if isinstance(accel, (list, tuple)):
            accel = np.array(accel)

        def _inject(mu):
            return {
                "t": torch.tensor(np.linspace(0.0, 1.0, max(len(u), 101)), dtype=torch.float32),
                "u_array": torch.tensor(u, dtype=torch.float32) if u.ndim else torch.tensor([float(u)], dtype=torch.float32),
                "strain": torch.tensor(strain, dtype=torch.float32) if strain.ndim else torch.tensor([float(strain)], dtype=torch.float32),
                "accel_array": torch.tensor(accel, dtype=torch.float32) if accel.ndim else torch.tensor([float(accel)], dtype=torch.float32),
                "u_final": torch.tensor(float(u[-1] if hasattr(u, "__len__") else u), dtype=torch.float32).unsqueeze(0),
                "accel_final": torch.tensor(float(accel[-1] if hasattr(accel, "__len__") else accel), dtype=torch.float32).unsqueeze(0),
            }

        return _inject
    return None


def _make_physics_runner(bounds: Bounds, cfg: PipelineConfig):
    fea_runner = _maybe_load_fea_injection()
    mock_runner = _physics_runner_mock()

    def runner(mu):
        if isinstance(mu, Mapping):
            mu_t = torch.tensor(list(mu.values()), dtype=torch.float32).view(1, -1)
        else:
            mu_t = mu if mu.dim() > 1 else mu.unsqueeze(0)

        raw_mu_dict = mu_tensor_to_dict(mu_t, list(bounds.names))
        mu_dict_keys_lower = {k.lower(): v for k, v in raw_mu_dict.items()}
        t3_val = float(mu_dict_keys_lower.get("t3", 0.0))
        t_total = float(mu_dict_keys_lower.get("t1", 0.0)) + float(mu_dict_keys_lower.get("t2", 0.0)) + t3_val

        tol = 0.001
        is_flat = t_total < tol
        is_3layer = t3_val >= tol
        is_fea_simulated = fea_runner is not None and (is_flat or is_3layer)

        if is_fea_simulated:
            return fea_runner(mu)
        return mock_runner(mu)

    return runner


def _check_bounds_consistency(bounds: Bounds, cfg: PipelineConfig):
    required = set(cfg.metrics.param_names)
    available = set(bounds.names)
    missing = required - available
    if missing:
        raise ValueError(f"Missing parameters in bounds: {missing}")
    for k in required:
        lo, hi = bounds.get(k)
        if lo > hi:
            raise ValueError(f"Bounds invalid for {k}: {lo} > {hi}")


def _log_summarize(bounds: Bounds, cfg: PipelineConfig):
    print("=" * 60)
    print("Pipeline Configuration Summary")
    print("=" * 60)
    print(f"Parameters ({len(bounds.names)}): {bounds.names}")
    print(f"Dataset size: {cfg.dataset.n_points}")
    print(f"Surrogate: hidden_layers={cfg.surrogate.hidden_layers}, hidden_size={cfg.surrogate.hidden_size}")
    print(f"Optimization: lr={cfg.optimize.lr}, steps={cfg.optimize.steps}, restarts={cfg.optimize.restarts}")
    print(f"Active learning: enabled={cfg.active_learning.enabled}")
    print("=" * 60)


def _save_artifacts(args, bounds, surrogate, mtrain, best_mu, baseline_mu, out_dir):
    import json

    meta: Dict[str, Any] = {
        "args": vars(args),
        "bounds": {k: [float(bounds.get(k)[0]), float(bounds.get(k)[1])] for k in bounds.names},
        "best_mu": best_mu,
        "baseline_mu": baseline_mu,
    }

    meta_path = os.path.join(out_dir, "pipeline_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {meta_path}")

    model_path = os.path.join(out_dir, "surrogate_model.pt")
    torch.save(
        {
            "state_dict": surrogate.model.state_dict(),
            "input_names": bounds.names,
            "target_names": ["y_strain_energy", "y_accel_peak", "y_disp_peak"],
        },
        model_path,
    )
    print(f"Saved surrogate model to {model_path}")

    np.save(os.path.join(out_dir, "train_loss.npy"), np.array(mtrain.loss_history, dtype=float))
    np.save(os.path.join(out_dir, "val_loss.npy"), np.array(mtrain.val_loss_history, dtype=float))


def main():
    parser = argparse.ArgumentParser(description="Run surrogate optimization pipeline")
    parser.add_argument("--dataset-n", type=int, default=256, help="Number of samples for dataset")
    parser.add_argument("--opt-steps", type=int, default=200, help="Optimization steps")
    parser.add_argument("--opt-restarts", type=int, default=5, help="Multi-start restarts")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--active-learning", action="store_true", help="Enable active learning")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    cfg = PipelineConfig(
        surrogate={"seed": args.seed},
        dataset={"n_points": args.dataset_n},
        optimize={"steps": args.opt_steps, "restarts": args.opt_restarts},
        active_learning={"enabled": args.active_learning, "seed": args.seed},
    )
    cfg.active_learning.seed = args.seed

    _check_bounds_consistency(cfg.bounds, cfg)
    _log_summarize(cfg.bounds, cfg)

    physics_runner = _make_physics_runner(cfg.bounds, cfg)

    target_names = ["y_strain_energy", "y_accel_peak", "y_disp_peak"]
    ds = generate_dataset(cfg.dataset.n_points, cfg.bounds, physics_runner, cfg, seed=cfg.surrogate.seed, target_names=target_names)

    surrogate, _, mtrain = train_surrogate(ds, cfg.bounds, cfg)
    print(f"Surrogate trained. Best val loss={mtrain.best_val_loss:.6e}")

    opt_cfg = cfg.optimize
    opt_cfg.seed = cfg.surrogate.seed
    best_mu, cand = optimize_design(surrogate, cfg.bounds, opt_cfg)

    baseline_mu: Dict[str, float] = {k: float((cfg.bounds.get(k)[0] + cfg.bounds.get(k)[1]) / 2.0) for k in cfg.bounds.names}

    comparison = compare_designs(baseline_mu, [best_mu], physics_runner, cfg.metrics)
    print("Comparison (baseline vs optimized):")
    for row in comparison:
        print(f"  {row}")

    if args.active_learning:
        print("Running active learning refinement...")
        surrogate, logs = iterative_active_learning(
            bounds=cfg.bounds,
            physics_runner=physics_runner,
            cfg=cfg,
            candidates=[best_mu],
            initial_surrogate=surrogate,
            dataset_n=args.dataset_n,
        )
        print(f"Active learning completed. {len(logs)} iterations.")

    os.makedirs(args.output_dir, exist_ok=True)
    _save_artifacts(args, cfg.bounds, surrogate, mtrain, best_mu, baseline_mu, args.output_dir)

    if cfg.damage.enabled:
        print("Running repeated impact analysis...")
        impact_log = run_repeated_impacts(best_mu, cfg.bounds, physics_runner, cfg.metrics, cfg.damage)
        for rec in impact_log:
            print(rec)

    print(f"\nPipeline completed. Outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
