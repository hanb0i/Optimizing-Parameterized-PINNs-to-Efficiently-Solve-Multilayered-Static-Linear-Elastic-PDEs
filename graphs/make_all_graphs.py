from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "graphs" / "scripts"


def _python() -> str:
    return sys.executable or "python3"


def _run(cmd: list[str]) -> None:
    pretty = " ".join(cmd)
    print(f"\n=== {pretty} ===", flush=True)
    base_env = {
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(REPO_ROOT / ".mplconfig"),
        "XDG_CACHE_HOME": str(REPO_ROOT / ".cache"),
        "PYTHONPYCACHEPREFIX": str(REPO_ROOT / ".pycache"),
    }
    subprocess.run(cmd, cwd=str(REPO_ROOT), env={**os.environ, **base_env}, check=True)


def _run_script(script: str, args: list[str] | None = None) -> None:
    script_path = SCRIPTS_DIR / script
    cmd = [_python(), str(script_path)]
    if args:
        cmd.extend(args)
    _run(cmd)


def _param_names(npz_path: Path) -> list[str] | None:
    if not npz_path.exists():
        return None
    try:
        blob = np.load(str(npz_path), allow_pickle=True)
        return [str(x) for x in list(blob["param_names"])]
    except Exception:
        return None


def _three_layer_surrogate_ok() -> bool:
    out_dir = REPO_ROOT / "pinn-workflow" / "surrogate_workflow" / "outputs"
    ds = out_dir / "phase1_dataset.npz"
    model = out_dir / "surrogate_model.pt"
    return ds.exists() and model.exists() and _param_names(ds) == ["E1", "t1", "E2", "t2", "E3", "t3"]


def _two_layer_surrogate_ok() -> bool:
    out_dir = REPO_ROOT / "pinn-workflow-2layer" / "surrogate_workflow" / "outputs"
    ds = out_dir / "phase1_dataset.npz"
    model = out_dir / "surrogate_model.pt"
    return ds.exists() and model.exists() and _param_names(ds) == ["E1", "t1", "E2", "t2"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all paper figures under `graphs/figures/`.")
    parser.add_argument(
        "--mode",
        choices=["fast", "full", "plot-only"],
        default="fast",
        help="fast: generate missing surrogate artifacts with small defaults; full: use workflow defaults; plot-only: no generation.",
    )
    parser.add_argument("--device", default="cpu", help="Device for surrogate generation (cpu/cuda/mps).")
    parser.add_argument("--surrogate2-n-samples", type=int, default=None, help="Override 2-layer surrogate sample count.")
    parser.add_argument("--surrogate2-max-epochs", type=int, default=None, help="Override 2-layer surrogate max epochs.")
    parser.add_argument("--surrogate3-n-samples", type=int, default=None, help="Override 3-layer surrogate sample count.")
    parser.add_argument("--surrogate3-max-epochs", type=int, default=None, help="Override 3-layer surrogate max epochs.")
    args = parser.parse_args()

    mode = str(args.mode)

    if mode != "plot-only" and not _three_layer_surrogate_ok():
        ns = args.surrogate3_n_samples
        me = args.surrogate3_max_epochs
        if mode == "fast":
            ns = 200 if ns is None else ns
            me = 1200 if me is None else me
        cmd = [_python(), "pinn-workflow/surrogate_workflow/run_phase1.py", "--regenerate", "--device", str(args.device)]
        if ns is not None:
            cmd.extend(["--n-samples", str(int(ns))])
        if me is not None:
            cmd.extend(["--max-epochs", str(int(me))])
        _run(cmd)
        _run([_python(), "pinn-workflow/surrogate_workflow/verify_phase1.py", "--device", str(args.device)])

    if mode != "plot-only" and not _two_layer_surrogate_ok():
        ns = args.surrogate2_n_samples
        me = args.surrogate2_max_epochs
        if mode == "fast":
            ns = 300 if ns is None else ns
            me = 1600 if me is None else me
        cmd = [
            _python(),
            "pinn-workflow-2layer/surrogate_workflow/run_phase1.py",
            "--regenerate",
            "--device",
            str(args.device),
        ]
        if ns is not None:
            cmd.extend(["--n-samples", str(int(ns))])
        if me is not None:
            cmd.extend(["--max-epochs", str(int(me))])
        _run(cmd)
        _run([_python(), "pinn-workflow-2layer/surrogate_workflow/verify_phase1.py", "--device", str(args.device)])

    _run_script("plot_geometry_bc.py")
    _run_script("plot_ablation_table.py")
    _run_script("plot_error_heatmap.py")
    _run_script("plot_surrogate_verification.py")
    _run_script("plot_surrogate_verification_two_layer.py")
    _run_script("plot_surrogate_verification_results.py")
    print("\nDone. Figures are in `graphs/figures/`.")


if __name__ == "__main__":
    main()
