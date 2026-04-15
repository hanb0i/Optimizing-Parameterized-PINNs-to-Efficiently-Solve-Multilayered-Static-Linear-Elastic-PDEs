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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all paper figures under `graphs/figures/`.")
    parser.add_argument(
        "--mode",
        choices=["fast", "full", "plot-only"],
        default="fast",
        help="fast: generate missing artifacts with small defaults; full: use workflow defaults; plot-only: no generation.",
    )
    parser.add_argument("--device", default="cpu", help="Device for generation (cpu/cuda/mps).")
    args = parser.parse_args()

    _run_script("plot_geometry_bc.py")
    _run_script("plot_ablation_table.py")
    _run_script("plot_error_heatmap.py")
    _run_script("plot_generalization_summary.py")
    _run_script("plot_unsupervised_region_summary.py")
    _run_script("plot_efficiency_timing.py")
    _run_script("plot_one_layer_summaries.py")
    print("\nDone. Figures are in `graphs/figures/`.")


if __name__ == "__main__":
    main()
