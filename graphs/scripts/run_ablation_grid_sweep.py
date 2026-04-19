#!/usr/bin/env python3
"""Grid-sweep evaluation for all three-layer ablation variants."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from _common import DATA_DIR, REPO_ROOT


def _python() -> str:
    return sys.executable or "python3"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run grid-sweep evaluation on all three-layer ablation variants."
    )
    parser.add_argument("--device", default=None, help="Override PINN_DEVICE (cpu/cuda/mps)")
    args = parser.parse_args()

    out_csv = DATA_DIR / "ablation_grid_sweep_results.csv"
    logs_dir = DATA_DIR / "ablation_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    base_env = {
        "MPLCONFIGDIR": str(REPO_ROOT / ".mplconfig"),
        "PYTHONPYCACHEPREFIX": str(REPO_ROOT / ".pycache"),
        "PINN_FORCE_CPU": "0",
    }
    if args.device:
        base_env["PINN_DEVICE"] = str(args.device)

    calibration_path = DATA_DIR / "three_layer_compliance_calibration.json"

    runs_dir = DATA_DIR / "ablation_runs"

    # Same variants & overrides as run_ablation_three_layer.py
    variants: list[tuple[str, dict[str, str], Path]] = [
        (
            "Base parametric PINN",
            {
                "PINN_E_COMPLIANCE_POWER": "0",
                "PINN_THICKNESS_COMPLIANCE_ALPHA": "0",
                "PINN_PDE_DECOMPOSE_BY_LAYER": "0",
                "PINN_W_INTERFACE_U": "0",
                "PINN_N_INTERFACE": "2000",
                "PINN_INTERFACE_SAMPLE_FRACTION": "0.25",
                "PINN_USE_SUPERVISION_DATA": "0",
                "PINN_W_DATA": "0",
                "PINN_N_DATA_POINTS": "0",
            },
            runs_dir / "base_parametric_pinn" / "pinn_model.pth",
        ),
        (
            "+ Compliance-aware scaling",
            {
                "PINN_E_COMPLIANCE_POWER": "0.95",
                "PINN_THICKNESS_COMPLIANCE_ALPHA": "3",
                "PINN_DISPLACEMENT_COMPLIANCE_SCALE": "1",
                "PINN_PDE_DECOMPOSE_BY_LAYER": "0",
                "PINN_W_INTERFACE_U": "0",
                "PINN_N_INTERFACE": "2000",
                "PINN_INTERFACE_SAMPLE_FRACTION": "0.25",
                "PINN_USE_SUPERVISION_DATA": "0",
                "PINN_W_DATA": "0",
                "PINN_N_DATA_POINTS": "0",
            },
            runs_dir / "plus_compliance_aware_scaling" / "pinn_model.pth",
        ),
        (
            "+ Layerwise PDE decomposition",
            {
                "PINN_E_COMPLIANCE_POWER": "0",
                "PINN_THICKNESS_COMPLIANCE_ALPHA": "0",
                "PINN_PDE_DECOMPOSE_BY_LAYER": "1",
                "PINN_W_INTERFACE_U": "0",
                "PINN_N_INTERFACE": "2000",
                "PINN_INTERFACE_SAMPLE_FRACTION": "0.25",
                "PINN_USE_SUPERVISION_DATA": "0",
                "PINN_W_DATA": "0",
                "PINN_N_DATA_POINTS": "0",
            },
            runs_dir / "plus_layerwise_pde_decomposition" / "pinn_model.pth",
        ),
        (
            "+ Interface continuity enforcement",
            {
                "PINN_E_COMPLIANCE_POWER": "0",
                "PINN_THICKNESS_COMPLIANCE_ALPHA": "0",
                "PINN_PDE_DECOMPOSE_BY_LAYER": "0",
                "PINN_W_INTERFACE_U": "300",
                "PINN_N_INTERFACE": "16000",
                "PINN_INTERFACE_SAMPLE_FRACTION": "0.75",
                "PINN_USE_SUPERVISION_DATA": "0",
                "PINN_W_DATA": "0",
                "PINN_N_DATA_POINTS": "0",
            },
            runs_dir / "plus_interface_continuity_enforcement" / "pinn_model.pth",
        ),
        (
            "+ Sparse FEM supervision",
            {
                "PINN_E_COMPLIANCE_POWER": "0",
                "PINN_THICKNESS_COMPLIANCE_ALPHA": "0",
                "PINN_PDE_DECOMPOSE_BY_LAYER": "0",
                "PINN_W_INTERFACE_U": "0",
                "PINN_N_INTERFACE": "2000",
                "PINN_INTERFACE_SAMPLE_FRACTION": "0.25",
                "PINN_USE_SUPERVISION_DATA": "1",
                "PINN_W_DATA": "400",
                "PINN_N_DATA_POINTS": "36000",
                "PINN_SUPERVISION_THICKNESS_POWER": "3.0",
                "PINN_FEM_NE_X": "10",
                "PINN_FEM_NE_Y": "10",
                "PINN_FEM_NE_Z": "4",
            },
            runs_dir / "plus_sparse_fem_supervision" / "pinn_model.pth",
        ),
        (
            "Full framework",
            {
                "PINN_E_COMPLIANCE_POWER": "0.95",
                "PINN_THICKNESS_COMPLIANCE_ALPHA": "3",
                "PINN_DISPLACEMENT_COMPLIANCE_SCALE": "1",
                "PINN_PDE_DECOMPOSE_BY_LAYER": "1",
                "PINN_W_INTERFACE_U": "300",
                "PINN_W_PDE": "10",
                "PINN_W_DATA": "400",
                "PINN_N_INTERFACE": "16000",
                "PINN_INTERFACE_SAMPLE_FRACTION": "0.75",
                "PINN_USE_SUPERVISION_DATA": "1",
                "PINN_N_DATA_POINTS": "36000",
                "PINN_SUPERVISION_THICKNESS_POWER": "3.0",
                "PINN_FEM_NE_X": "10",
                "PINN_FEM_NE_Y": "10",
                "PINN_FEM_NE_Z": "4",
            },
            runs_dir / "full_framework" / "pinn_model.pth",
        ),
    ]

    rows: list[dict[str, str]] = []

    # Temporarily rename pinn-workflow to avoid import conflicts
    pinn_workflow = REPO_ROOT / "pinn-workflow"
    pinn_workflow_temp = REPO_ROOT / "pinn-workflow-temp"
    renamed = False
    if pinn_workflow.exists():
        pinn_workflow.rename(pinn_workflow_temp)
        renamed = True
        print("Temporarily renamed pinn-workflow -> pinn-workflow-temp")

    try:
        for variant_name, overrides, ckpt_path in variants:
            print("\n" + "=" * 80)
            print(f"Variant: {variant_name}")
            print(f"Checkpoint: {ckpt_path}")

            if not ckpt_path.exists():
                print(f"WARNING: checkpoint not found, skipping.")
                rows.append({
                    "variant": variant_name,
                    "mean_mae": "N/A",
                    "worst_mae": "N/A",
                })
                continue

            env = dict(base_env)
            env.update(overrides)
            env["PINN_MODEL_PATH"] = str(ckpt_path)
            env["PINN_EVAL_OUT_DIR"] = str(runs_dir / variant_name.lower().replace(" ", "_").replace("+", "plus").replace("/", "_").replace("-", "_") / "grid_sweep_viz")

            if variant_name == "Full framework" and calibration_path.exists():
                env["PINN_CALIBRATION_JSON"] = str(calibration_path)

            log_path = logs_dir / f"{variant_name.lower().replace(' ', '_').replace('+', 'plus').replace('/', '_').replace('-', '_')}_grid_sweep.log"

            cmd = [_python(), str(REPO_ROOT / "compare_three_layer_pinn_fem.py")]
            print(f"Running: {' '.join(cmd)}")
            proc = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                env={**os.environ, **env},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            log_path.write_text(proc.stdout)

            if proc.returncode != 0:
                print(f"ERROR: grid sweep failed (exit {proc.returncode}). See log: {log_path}")
                rows.append({
                    "variant": variant_name,
                    "mean_mae": "ERROR",
                    "worst_mae": "ERROR",
                })
                continue

            # Parse stdout for mean and worst MAE
            mean_mae = None
            worst_mae = None
            for line in proc.stdout.splitlines():
                if "Three-layer sweep mean MAE=" in line:
                    mean_mae = line.split("Three-layer sweep mean MAE=")[1].split("%")[0].strip()
                if "Three-layer sweep worst MAE=" in line:
                    worst_mae = line.split("Three-layer sweep worst MAE=")[1].split("%")[0].strip()

            if mean_mae is None or worst_mae is None:
                print(f"WARNING: could not parse MAE values from stdout.")
                print(proc.stdout[-500:])

            rows.append({
                "variant": variant_name,
                "mean_mae": mean_mae or "N/A",
                "worst_mae": worst_mae or "N/A",
            })
            print(f"  mean MAE (%): {mean_mae}")
            print(f"  worst MAE (%): {worst_mae}")

    finally:
        if renamed:
            pinn_workflow_temp.rename(pinn_workflow)
            print("Restored pinn-workflow-temp -> pinn-workflow")

    # Write CSV
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "mean_mae", "worst_mae"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote grid-sweep results to {out_csv}")
    for r in rows:
        print(f"  {r['variant']}: mean={r['mean_mae']}%, worst={r['worst_mae']}%")


if __name__ == "__main__":
    main()
