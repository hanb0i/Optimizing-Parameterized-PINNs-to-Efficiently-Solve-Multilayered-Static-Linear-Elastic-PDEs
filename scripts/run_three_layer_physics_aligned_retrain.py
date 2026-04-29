"""Retrain/evaluate the three-layer PINN on corrected refined FEM supervision.

This is the most physically aligned route for closing the remaining <5% target:
the FEM solver and PINN traction residual now use the same smooth load patch, and
this runner regenerates sparse FEM supervision on the refined benchmark mesh
before training.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "graphs" / "data"
RUNS_DIR = DATA_DIR / "physics_aligned_runs"


def _python() -> str:
    return sys.executable or "python3"


def _run(cmd: list[str], env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running: {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env={**os.environ, **env},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_path.write_text(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit {proc.returncode}. See {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="three_layer_refined_smooth_load")
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-data-points", type=int, default=72000)
    parser.add_argument("--n-cases", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    run_dir = RUNS_DIR / args.run_name
    log_dir = run_dir / "logs"
    eval_csv = run_dir / "random_interior_generalization.csv"
    eval_summary = run_dir / "random_interior_generalization_summary.json"
    ckpt = run_dir / "pinn_model.pth"
    run_dir.mkdir(parents=True, exist_ok=True)

    env = {
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(REPO_ROOT / ".mplconfig"),
        "PYTHONPYCACHEPREFIX": str(REPO_ROOT / ".pycache"),
        "PINN_OUT_DIR": str(run_dir),
        "PINN_MODEL_PATH": str(ckpt),
        "PINN_WARM_START": "0",
        "PINN_SUPERVISION_CACHE": "1",
        "PINN_REGEN_SUPERVISION": "1",
        "PINN_FEM_NE_X": "16",
        "PINN_FEM_NE_Y": "16",
        "PINN_FEM_NE_Z": "8",
        "PINN_N_DATA_POINTS": str(args.n_data_points),
        "PINN_USE_SUPERVISION_DATA": "1",
        "PINN_W_DATA": "400",
        "PINN_E_COMPLIANCE_POWER": "0.95",
        "PINN_THICKNESS_COMPLIANCE_ALPHA": "3",
        "PINN_DISPLACEMENT_COMPLIANCE_SCALE": "1",
        "PINN_PDE_DECOMPOSE_BY_LAYER": "1",
        "PINN_W_INTERFACE_U": "300",
        "PINN_N_INTERFACE": "16000",
        "PINN_INTERFACE_SAMPLE_FRACTION": "0.75",
        "PINN_ADAPTIVE_RESAMPLE_EVERY": "500",
        "PINN_EPOCHS_ADAM": str(args.epochs),
        "PINN_EPOCHS_SOAP": str(args.epochs),
    }
    if args.device:
        env["PINN_DEVICE"] = args.device

    manifest = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt),
        "reason": "Physics-aligned retrain: corrected smooth load patch, refined FEM supervision mesh, regenerated supervision cache.",
        "env": env,
        "evaluation": {
            "seed": args.seed,
            "n_cases": args.n_cases,
            "mesh": {"ne_x": 16, "ne_y": 16, "ne_z": 8},
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if args.skip_train:
        if not ckpt.exists():
            raise FileNotFoundError(f"--skip-train requested but checkpoint is missing: {ckpt}")
        print(f"Skipping training; using existing checkpoint {ckpt}")
    else:
        _run([_python(), "three-layer-workflow/train.py"], env, log_dir / "train.log")

    eval_env = dict(env)
    eval_env["PINN_REGEN_SUPERVISION"] = "0"
    _run(
        [
            _python(),
            "scripts/run_random_interior_generalization.py",
            "--model-path",
            str(ckpt),
            "--n-cases",
            str(args.n_cases),
            "--seed",
            str(args.seed),
            "--ne-x",
            "16",
            "--ne-y",
            "16",
            "--ne-z",
            "8",
            "--out-csv",
            str(eval_csv),
            "--out-summary",
            str(eval_summary),
        ],
        eval_env,
        log_dir / "eval.log",
    )
    print(f"Wrote {eval_summary}")


if __name__ == "__main__":
    main()
