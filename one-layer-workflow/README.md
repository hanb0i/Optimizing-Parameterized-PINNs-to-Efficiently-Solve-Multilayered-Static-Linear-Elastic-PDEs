## One-layer PINN baseline (recovered)

Recovered from commit `18793ca` (remote branch `origin/Analyzing-Surrogate`) and placed here to avoid interfering with the current multi-layer pipeline.

### What it is

- Single-layer, homogeneous, static linear elasticity PINN.
- Physics-driven training: equilibrium PDE (`-div(sigma)=0`), side clamps, traction patch on the top surface, and an energy/work term.
- Parametric inputs include `E`, total `thickness`, and optional `restitution/friction/impact_velocity` channels (can be treated as neutral for FEA parity).

### Files

- `one-layer/pinn_config.py`: baseline config/hyperparameters
- `one-layer/model.py`: MLP predicting displacement (or compliance-scaled output)
- `one-layer/physics.py`: residuals + boundary/energy losses
- `one-layer/data.py`: collocation + optional FEM supervision sampler
- `one-layer/soap.py`: SOAP optimizer implementation used by `train.py`
- `one-layer/train.py`: training entrypoint (writes outputs inside `one-layer/`)
- `one-layer/verify_parametric_pinn.py`: runs FEA comparisons and writes plots to `one-layer/visualization/`
- `one-layer/pinn_model.pth`: checkpoint recovered from `18793ca`

### How to run

From the repo root:

- Train: `python3 one-layer/train.py`
- Verify (FEA vs PINN): `python3 one-layer/verify_parametric_pinn.py`

Outputs stay in `one-layer/` (e.g. `one-layer/visualization/`).

