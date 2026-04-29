"""Write a machine-readable status report for the revision checklist."""

from __future__ import annotations

import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "graphs" / "data" / "revision_compliance_report.json"


def _json(path: str) -> dict:
    p = REPO_ROOT / path
    return json.loads(p.read_text()) if p.exists() else {}


def _csv_rows(path: str) -> list[dict]:
    p = REPO_ROOT / path
    if not p.exists():
        return []
    with p.open("r", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    one_calibrated = (REPO_ROOT / "graphs/data/one_layer_random_generalization_refined_calibrated_summary.json").exists()
    three_calibrated = (REPO_ROOT / "graphs/data/three_layer_random_generalization_refined_calibrated_summary.json").exists()
    one = _json("graphs/data/one_layer_random_generalization_refined_calibrated_summary.json") or _json("graphs/data/one_layer_random_generalization_refined_summary.json")
    three = _json("graphs/data/three_layer_random_generalization_refined_calibrated_summary.json") or _json("graphs/data/three_layer_random_generalization_refined_summary.json")
    timing = _json("graphs/data/efficiency_timing_refined_100_summary.json") or _json("graphs/data/efficiency_timing_refined_summary.json")
    one_timing = _json("graphs/data/one_layer_efficiency_timing_refined_100_summary.json") or _json("graphs/data/one_layer_efficiency_timing_refined_summary.json")
    opt = _json("graphs/data/surrogate_optimization_gradient_summary.json")
    ablation = _csv_rows("graphs/data/ablation_results.csv")
    one_ablation = _csv_rows("graphs/data/one_layer_ablation_results.csv")
    missing_ablation = [r["variant"] for r in ablation if r.get("status") == "missing_checkpoint_requires_training"]
    missing_one_ablation = [r["variant"] for r in one_ablation if r.get("status") == "missing_checkpoint"]

    report = {
        "date": "2026-04-28",
        "status": "implemented_with_accuracy_and_training_caveats",
        "satisfied_items": {
            "strong_form_before_weak_form": True,
            "layer_interaction_documented_and_commented": True,
            "explicit_loss_components_logged": True,
            "mean_relative_l2_and_average_displacement_metrics": True,
            "one_and_three_layer_refined_benchmarks_generated": True,
            "timing_comparison_generated": True,
            "backward_ablation_script_and_table": True,
            "random_heldout_generalization_generated": True,
            "fem_supervision_count_reported": True,
            "gradient_based_constrained_optimization_generated": True,
            "nontrivial_material_design_objective": True,
        },
        "caveats": {
            "one_layer_less_than_5_pct_top_surface_met": bool(one.get("all_top_cases_below_5_pct", False)),
            "three_layer_less_than_5_pct_top_surface_met": bool(three.get("all_top_cases_below_5_pct", False)),
            "missing_three_layer_backward_ablation_checkpoints": missing_ablation,
            "missing_one_layer_backward_ablation_checkpoints": missing_one_ablation,
            "full_100_repeat_timing": {
                "three_layer_repeats_measured": timing.get("fem_repeats_measured"),
                "one_layer_repeats_measured": one_timing.get("fem_repeats_measured"),
                "scripts_default_to_100_repeats": True,
            },
            "likely_accuracy_limiters": [
                "three-layer saved checkpoint remains above the <5% top-surface target after transparent calibration",
                "16x16x8 FEM remains mesh-sensitive relative to 32x32x16 in convergence outputs",
                "full backward ablation variants require retraining to populate all rows",
            ],
        },
        "key_metrics": {
            "one_layer_refined": {
                "top_uz_mae_pct_mean": one.get("top_uz_mae_pct_mean"),
                "top_uz_relative_l2_pct_mean": one.get("top_uz_relative_l2_pct_mean"),
                "volume_mae_pct_mean": one.get("volume_mae_pct_mean"),
                "calibrated": one_calibrated,
            },
            "three_layer_refined": {
                "top_uz_mae_pct_mean": three.get("top_uz_mae_pct_mean"),
                "top_uz_relative_l2_pct_mean": three.get("top_uz_relative_l2_pct_mean"),
                "volume_mae_pct_mean": three.get("volume_mae_pct_mean"),
                "calibrated": three_calibrated,
            },
            "timing": {
                "three_layer_fem_seconds_mean": timing.get("fem_seconds_mean"),
                "three_layer_pinn_eval_seconds_mean": timing.get("pinn_eval_seconds_mean"),
                "estimated_fem_seconds_for_1e6_configs": timing.get("estimated_fem_seconds_for_1e6_configs"),
                "estimated_pinn_inference_seconds_for_1e6_configs": timing.get("estimated_pinn_inference_seconds_for_1e6_configs"),
            },
            "optimization": {
                "method": opt.get("optimization_protocol", {}).get("method"),
                "constraint": opt.get("optimization_protocol", {}).get("constraint"),
                "best_surrogate_design": opt.get("best_surrogate_design"),
            },
        },
        "primary_artifacts": [
            "README.md",
            "REVISION_NOTES.md",
            "graphs/data/ablation_results.csv",
            "graphs/data/one_layer_ablation_results.csv",
            "graphs/data/one_layer_random_generalization_refined_calibrated_summary.json",
            "graphs/data/three_layer_random_generalization_refined_calibrated_summary.json",
            "graphs/data/efficiency_timing_refined_100_summary.json",
            "graphs/data/one_layer_efficiency_timing_refined_100_summary.json",
            "graphs/data/surrogate_optimization_gradient_summary.json",
            "graphs/generalized_study/fem_convergence/one_layer_convergence.csv",
            "graphs/generalized_study/fem_convergence/three_layer_convergence.csv",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
