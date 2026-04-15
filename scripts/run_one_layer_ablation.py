"""Run/evaluate one-layer ablation variants.

The one-layer model has no material interface, so the interface-continuity
ablation is emitted as not applicable rather than treated as a fake experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "graphs" / "data"
ONE_LAYER_DIR = REPO_ROOT / "one-layer"
if not ONE_LAYER_DIR.exists():
    ONE_LAYER_DIR = REPO_ROOT / "one-layer-workflow"


def _python() -> str:
    return sys.executable or "python3"


def _run(cmd: list[str], env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
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
        raise RuntimeError(f"Command failed (exit {proc.returncode}). See log: {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs-soap", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--n-eval-cases", type=int, default=4)
    args = parser.parse_args()

    out_csv = DATA_DIR / "one_layer_ablation_results.csv"
    runs_dir = DATA_DIR / "one_layer_ablation_runs"
    logs_dir = DATA_DIR / "one_layer_ablation_logs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    base_env = {
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(REPO_ROOT / ".mplconfig"),
        "XDG_CACHE_HOME": str(REPO_ROOT / ".cache"),
        "PYTHONPYCACHEPREFIX": str(REPO_ROOT / ".pycache"),
        "PINN_WARM_START": "0",
    }
    if args.device:
        base_env["PINN_DEVICE"] = args.device
    if args.epochs_soap is not None:
        base_env["PINN_EPOCHS_ADAM"] = str(args.epochs_soap)
        base_env["PINN_EPOCHS_SOAP"] = str(args.epochs_soap)

    variants = [
        (
            "Pure-physics baseline",
            {
                "PINN_E_COMPLIANCE_POWER": "0",
                "PINN_THICKNESS_COMPLIANCE_ALPHA": "0",
                "PINN_USE_SUPERVISION_DATA": "0",
                "PINN_W_DATA": "0",
            },
        ),
        (
            "+ Compliance-aware scaling",
            {
                "PINN_E_COMPLIANCE_POWER": "0.973",
                "PINN_THICKNESS_COMPLIANCE_ALPHA": "1.234",
                "PINN_USE_SUPERVISION_DATA": "0",
                "PINN_W_DATA": "0",
            },
        ),
        (
            "+ Tuned compliance-aware scaling",
            {},
        ),
        (
            "+ Sparse FEM supervision",
            {
                "PINN_E_COMPLIANCE_POWER": "0",
                "PINN_THICKNESS_COMPLIANCE_ALPHA": "0",
                "PINN_USE_SUPERVISION_DATA": "1",
                "PINN_W_DATA": "1",
                "PINN_N_DATA_POINTS": "9000",
            },
        ),
        (
            "Full one-layer framework",
            {
                "PINN_E_COMPLIANCE_POWER": "0.973",
                "PINN_THICKNESS_COMPLIANCE_ALPHA": "1.234",
                "PINN_USE_SUPERVISION_DATA": "1",
                "PINN_W_DATA": "1",
                "PINN_N_DATA_POINTS": "9000",
            },
        ),
    ]
    calibration_path = DATA_DIR / "one_layer_compliance_calibration.json"
    if calibration_path.exists():
        tuned = json.loads(calibration_path.read_text()).get("tuned_params", {})
        for idx, (variant_name, overrides) in enumerate(variants):
            if variant_name == "+ Tuned compliance-aware scaling":
                variants[idx] = (
                    variant_name,
                    {
                        "PINN_DISPLACEMENT_COMPLIANCE_SCALE": str(tuned.get("PINN_DISPLACEMENT_COMPLIANCE_SCALE", 1.0)),
                        "PINN_E_COMPLIANCE_POWER": str(tuned.get("PINN_E_COMPLIANCE_POWER", 0.973)),
                        "PINN_THICKNESS_COMPLIANCE_ALPHA": str(tuned.get("PINN_THICKNESS_COMPLIANCE_ALPHA", 1.234)),
                        "PINN_USE_SUPERVISION_DATA": "0",
                        "PINN_W_DATA": "0",
                    },
                )
                break

    rows = [
        {
            "variant": "+ Interface continuity enforcement",
            "status": "not_applicable_single_layer",
            "mean_top_uz_mae_pct": "",
            "worst_top_uz_mae_pct": "",
            "mean_volume_mae_pct": "",
            "worst_volume_mae_pct": "",
            "checkpoint": "",
        }
    ]

    for name, overrides in variants:
        slug = name.lower().replace("+", "plus").replace(" ", "_").replace("-", "_")
        run_dir = runs_dir / slug
        ckpt_path = run_dir / "pinn_model.pth"
        eval_summary = run_dir / "generalization_summary.json"
        eval_csv = run_dir / "generalization.csv"
        env = dict(base_env)
        env.update(overrides)
        env["PINN_OUT_DIR"] = str(run_dir)
        env["PINN_MODEL_PATH"] = str(ckpt_path)

        eval_ckpt_path = ckpt_path
        if args.skip_train and ckpt_path.exists():
            status = "evaluated_existing_checkpoint"
        elif args.skip_train and not ckpt_path.exists():
            shared_ckpt = ONE_LAYER_DIR / "pinn_model.pth"
            if name in {"Pure-physics baseline", "+ Compliance-aware scaling", "+ Tuned compliance-aware scaling"} and shared_ckpt.exists():
                eval_ckpt_path = shared_ckpt
                status = "evaluated_shared_pure_physics_checkpoint"
            else:
                rows.append(
                    {
                        "variant": name,
                        "status": "missing_checkpoint",
                        "mean_top_uz_mae_pct": "",
                        "worst_top_uz_mae_pct": "",
                        "mean_volume_mae_pct": "",
                        "worst_volume_mae_pct": "",
                        "checkpoint": str(ckpt_path),
                    }
                )
                continue
        else:
            _run([_python(), str(ONE_LAYER_DIR / "train.py")], env, logs_dir / f"{slug}_train.log")
            status = "trained_and_evaluated"

        _run(
            [
                _python(),
                "scripts/run_one_layer_generalization.py",
                "--model-path",
                str(eval_ckpt_path),
                "--n-cases",
                str(args.n_eval_cases),
                "--out-csv",
                str(eval_csv),
                "--out-summary",
                str(eval_summary),
            ],
            env,
            logs_dir / f"{slug}_eval.log",
        )
        summary = json.loads(eval_summary.read_text())
        rows.append(
            {
                "variant": name,
                "status": status,
                "mean_top_uz_mae_pct": f"{summary['top_uz_mae_pct_mean']:.6f}",
                "worst_top_uz_mae_pct": f"{summary['top_uz_mae_pct_worst']:.6f}",
                "mean_volume_mae_pct": f"{summary['volume_mae_pct_mean']:.6f}",
                "worst_volume_mae_pct": f"{summary['volume_mae_pct_worst']:.6f}",
                "checkpoint": str(eval_ckpt_path),
            }
        )

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "status",
                "mean_top_uz_mae_pct",
                "worst_top_uz_mae_pct",
                "mean_volume_mae_pct",
                "worst_volume_mae_pct",
                "checkpoint",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
