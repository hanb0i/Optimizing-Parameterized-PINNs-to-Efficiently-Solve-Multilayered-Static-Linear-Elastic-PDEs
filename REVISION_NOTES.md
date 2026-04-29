# Revision Notes and Reproducibility Status

Date: 2026-04-28

## What Was Updated

- Added the strong-form linear-elasticity PDE before the FEM weak form in `README.md`.
- Documented multilayer material lookup, interface displacement continuity, and traction balance in `README.md` and `fea-workflow/solver/README.md`.
- Added code comments at FEM and PINN layer-assignment/material-lookup sites.
- Fixed a FEM/PINN loading mismatch: both now use the same smooth load-patch profile.
- Replaced nodal top-load assignment in FEM with 2×2 Gauss quadrature on top element faces.
- Added explicit per-component loss logging to `loss_history_components.csv` for future training runs.
- Added relative L2/integral error, mean displacement error, and average displacement comparison metrics to one-layer and three-layer evaluators.
- Changed default benchmark/generalization mesh from coarse `8×8×4` or `10×10×4` usage to `16×16×8`.
- Updated timing scripts to run 100 repeats by default and report measured repeat count plus `1,000,000`-configuration estimates.
- Changed ablation scripts to backward ablations that start from the full model and remove one component at a time.
- Replaced brute-force/random optimization with multi-start gradient optimization over the PINN surrogate.
- Added the material-design constraint `min(E1,E2,E3) < 5`.
- Changed the design objective to weighted load-patch displacement plus material-cost and stiffness-contrast penalties.
- Added `scripts/run_three_layer_physics_aligned_retrain.py`, the recommended path for closing the remaining three-layer gap by regenerating refined FEM supervision and retraining against the corrected smooth load physics.

## Generated Outputs

The following outputs were regenerated. Files with `_100` use 100 repeats per listed timing configuration.

- `graphs/data/one_layer_random_generalization_refined.csv`
- `graphs/data/one_layer_random_generalization_refined_summary.json`
- `graphs/data/three_layer_random_generalization_refined.csv`
- `graphs/data/three_layer_random_generalization_refined_summary.json`
- `graphs/data/one_layer_efficiency_timing_refined.csv`
- `graphs/data/one_layer_efficiency_timing_refined_summary.json`
- `graphs/data/efficiency_timing_refined.csv`
- `graphs/data/efficiency_timing_refined_summary.json`
- `graphs/data/one_layer_efficiency_timing_refined_100.csv`
- `graphs/data/one_layer_efficiency_timing_refined_100_summary.json`
- `graphs/data/efficiency_timing_refined_100.csv`
- `graphs/data/efficiency_timing_refined_100_summary.json`
- `graphs/data/one_layer_compliance_calibration_refined.json`
- `graphs/data/three_layer_compliance_calibration_refined_optimized.json`
- `graphs/data/one_layer_random_generalization_refined_calibrated.csv`
- `graphs/data/one_layer_random_generalization_refined_calibrated_summary.json`
- `graphs/data/three_layer_random_generalization_refined_calibrated.csv`
- `graphs/data/three_layer_random_generalization_refined_calibrated_summary.json`
- `graphs/data/surrogate_optimization_gradient_candidates.csv`
- `graphs/data/surrogate_optimization_gradient_topk.csv`
- `graphs/data/surrogate_optimization_gradient_confirmation.csv`
- `graphs/data/surrogate_optimization_gradient_summary.json`
- `graphs/generalized_study/fem_convergence/one_layer_convergence.csv`
- `graphs/generalized_study/fem_convergence/three_layer_convergence.csv`

## Current Accuracy Limitation

After the load-profile fix and transparent compliance recalibration, the refined
one-layer check on eight held-out cases produced top-surface MAE mean `2.39%`.
All tested one-layer top-surface cases are below 5%.

After the load-profile fix and transparent compliance recalibration, the refined
three-layer held-out check produced top-surface MAE mean `6.83%`. Some tested
three-layer cases remain above 5%, so this empirical target is not yet fully met
for the saved checkpoint.

The FEM convergence study shows that `16×16×8` is still materially different
from `32×32×16` for the one-layer average displacement metric
(`top_avg_rel_err ≈ 27.1%`) and is closer but not fully converged for the
three-layer metric (`top_avg_rel_err ≈ 10.3%`). The remaining benchmark gap is
therefore a combination of FEM resolution sensitivity and saved checkpoints that
were trained before the FEM/PINN load-profile correction. A defensible <5%
top-surface comparison will require retraining/recalibration against the
corrected smooth traction formulation and/or using the finer FEM mesh as truth.
The implemented runner for that next step is:
`python3 scripts/run_three_layer_physics_aligned_retrain.py --epochs 1200 --device <cpu|mps|cuda>`.

## Commands Used

- Syntax check:
  - `PYTHONPYCACHEPREFIX=.pycache python3 -m py_compile ...`
- One-layer refined generalization:
  - `PYTHONPYCACHEPREFIX=.pycache PINN_FORCE_CPU=1 python3 scripts/run_one_layer_generalization.py --n-cases 2 --seed 20260428 --ne-x 16 --ne-y 16 --ne-z 8 --out-csv graphs/data/one_layer_random_generalization_refined.csv --out-summary graphs/data/one_layer_random_generalization_refined_summary.json`
- Three-layer refined generalization:
  - `PYTHONPYCACHEPREFIX=.pycache PINN_FORCE_CPU=1 python3 scripts/run_random_interior_generalization.py --n-cases 2 --seed 20260428 --ne-x 16 --ne-y 16 --ne-z 8 --out-csv graphs/data/three_layer_random_generalization_refined.csv --out-summary graphs/data/three_layer_random_generalization_refined_summary.json`
- Timing with reduced repeats:
  - `PYTHONPYCACHEPREFIX=.pycache PINN_FORCE_CPU=1 python3 scripts/report_efficiency_timing.py --repeats 2 --ne-x 16 --ne-y 16 --ne-z 8 --out-csv graphs/data/efficiency_timing_refined.csv --out-summary graphs/data/efficiency_timing_refined_summary.json`
  - `PYTHONPYCACHEPREFIX=.pycache PINN_FORCE_CPU=1 python3 scripts/report_one_layer_efficiency_timing.py --repeats 2 --ne-x 16 --ne-y 16 --ne-z 8 --out-csv graphs/data/one_layer_efficiency_timing_refined.csv --out-summary graphs/data/one_layer_efficiency_timing_refined_summary.json`
- Timing with 100 repeats:
  - `PYTHONPYCACHEPREFIX=.pycache PINN_FORCE_CPU=1 PINN_CALIBRATION_JSON=graphs/data/three_layer_compliance_calibration_refined_optimized.json python3 scripts/report_efficiency_timing.py --repeats 100 --ne-x 16 --ne-y 16 --ne-z 8 --out-csv graphs/data/efficiency_timing_refined_100.csv --out-summary graphs/data/efficiency_timing_refined_100_summary.json`
  - `PYTHONPYCACHEPREFIX=.pycache PINN_FORCE_CPU=1 PINN_DISPLACEMENT_COMPLIANCE_SCALE=0.83753602 PINN_E_COMPLIANCE_POWER=0.90929019 PINN_THICKNESS_COMPLIANCE_ALPHA=0.89139522 python3 scripts/report_one_layer_efficiency_timing.py --repeats 100 --ne-x 16 --ne-y 16 --ne-z 8 --out-csv graphs/data/one_layer_efficiency_timing_refined_100.csv --out-summary graphs/data/one_layer_efficiency_timing_refined_100_summary.json`
- Gradient optimization smoke run:
  - `PYTHONPYCACHEPREFIX=.pycache PINN_FORCE_CPU=1 python3 scripts/run_surrogate_optimization.py --n-candidates 3 --optimization-steps 10 --surrogate-ne-x 8 --surrogate-ne-y 8 --fem-ne-x 8 --fem-ne-y 8 --fem-ne-z 4 --top-k 2 --out-candidates-csv graphs/data/surrogate_optimization_gradient_candidates.csv --out-topk-csv graphs/data/surrogate_optimization_gradient_topk.csv --out-confirmation-csv graphs/data/surrogate_optimization_gradient_confirmation.csv --out-summary graphs/data/surrogate_optimization_gradient_summary.json`
- FEM convergence:
  - `PYTHONPYCACHEPREFIX=.pycache MPLCONFIGDIR=.mplconfig python3 graphs/scripts/run_fem_convergence_study.py`
