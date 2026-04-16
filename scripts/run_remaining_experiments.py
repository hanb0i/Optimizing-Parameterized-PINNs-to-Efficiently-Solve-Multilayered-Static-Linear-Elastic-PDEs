"""Run the remaining paper-support experiments in sequence."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ONE_LAYER_DIR = REPO_ROOT / "one-layer"
if not ONE_LAYER_DIR.exists():
    ONE_LAYER_DIR = REPO_ROOT / "one-layer-workflow"


def _python() -> str:
    return sys.executable or "python3"


def _run(cmd: list[str], env: dict[str, str]) -> None:
    print("\n=== " + " ".join(cmd) + " ===", flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), env={**os.environ, **env}, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=str(REPO_ROOT / "graphs" / "data" / "ablation_runs" / "full_framework" / "pinn_model.pth"))
    parser.add_argument("--one-layer-model-path", default=str(ONE_LAYER_DIR / "pinn_model.pth"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--random-cases", type=int, default=8)
    parser.add_argument("--skip-ablation", action="store_true", help="Do not run ablation training/evaluation.")
    parser.add_argument("--ablation-skip-train", action="store_true", help="Reuse existing ablation checkpoints.")
    args = parser.parse_args()

    env = {
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(REPO_ROOT / ".mplconfig"),
        "XDG_CACHE_HOME": str(REPO_ROOT / ".cache"),
        "PYTHONPYCACHEPREFIX": str(REPO_ROOT / ".pycache"),
    }
    if args.device:
        env["PINN_DEVICE"] = args.device

    if not args.skip_ablation:
        cmd = [_python(), "graphs/scripts/run_ablation_three_layer.py"]
        if args.ablation_skip_train:
            cmd.append("--skip-train")
        if args.device:
            cmd.extend(["--device", args.device])
        _run(cmd, env)

    _run([_python(), "scripts/tune_compliance_calibration.py", "--model-path", args.model_path], env)
    _run([_python(), "scripts/tune_three_layer_compliance_calibration.py", "--model-path", args.model_path, "--ridge", "10"], env)
    three_layer_env = dict(env)
    three_layer_cal = REPO_ROOT / "graphs" / "data" / "three_layer_compliance_calibration.json"
    if three_layer_cal.exists():
        three_layer_env["PINN_CALIBRATION_JSON"] = str(three_layer_cal)
    _run(
        [
            _python(),
            "scripts/run_random_interior_generalization.py",
            "--model-path",
            args.model_path,
            "--n-cases",
            str(args.random_cases),
        ],
        three_layer_env,
    )
    _run([_python(), "scripts/run_unsupervised_region_comparison.py", "--model-path", args.model_path], three_layer_env)
    _run([_python(), "scripts/report_efficiency_timing.py", "--model-path", args.model_path], three_layer_env)
    _run([_python(), "scripts/tune_one_layer_compliance_calibration.py", "--model-path", args.one_layer_model_path], env)
    tuned_env = dict(env)
    one_layer_cal = REPO_ROOT / "graphs" / "data" / "one_layer_compliance_calibration.json"
    if one_layer_cal.exists():
        import json

        tuned = json.loads(one_layer_cal.read_text()).get("tuned_params", {})
        for key, value in tuned.items():
            tuned_env[key] = str(value)
    _run([_python(), "scripts/run_one_layer_generalization.py", "--model-path", args.one_layer_model_path, "--n-cases", str(args.random_cases)], tuned_env)
    _run([_python(), "scripts/run_one_layer_unsupervised_region_comparison.py", "--model-path", args.one_layer_model_path], tuned_env)
    _run([_python(), "scripts/report_one_layer_efficiency_timing.py", "--model-path", args.one_layer_model_path], tuned_env)
    _run([_python(), "scripts/run_one_layer_ablation.py", "--skip-train", "--n-eval-cases", str(args.random_cases)], tuned_env)
    _run([_python(), "graphs/make_all_graphs.py", "--mode", "plot-only"], env)


if __name__ == "__main__":
    main()
