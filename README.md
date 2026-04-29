# Three-Layer PINN for Impact-Attenuation Design

Physics-Informed Neural Network (PINN) for predicting displacement fields in three-layered structures under impact loading.

## Overview

This repository implements a complete pipeline for:
1. **Physics Simulation**: PINN predicts 3D displacement fields for multi-layered materials
2. **Validation**: Comparison tools verify PINN accuracy against FEA

## Governing Problem and FEM Formulation

The PDE solved by both FEM and PINN/P2INN is static small-strain linear elasticity. For displacement
\(\mathbf{u}(\mathbf{x})\), strain \(\boldsymbol{\epsilon}(\mathbf{u})=\frac{1}{2}(\nabla \mathbf{u}+\nabla \mathbf{u}^{T})\), and stress
\(\boldsymbol{\sigma}(\mathbf{u})=\lambda\,\mathrm{tr}(\boldsymbol{\epsilon})\mathbf{I}+2\mu\boldsymbol{\epsilon}\), the strong form is

\[
-\nabla\cdot\boldsymbol{\sigma}(\mathbf{u})=\mathbf{b}\quad\mathrm{in}\ \Omega,
\]

with clamped side boundaries \(\mathbf{u}=\mathbf{0}\) on \(\Gamma_D\), applied top traction
\(\boldsymbol{\sigma}\mathbf{n}=\bar{\mathbf{t}}\) on the load patch \(\Gamma_t\), and zero traction on free surfaces. The Lamé
parameters are computed from Young's modulus \(E\) and Poisson ratio \(\nu\) as
\(\lambda=E\nu/((1+\nu)(1-2\nu))\) and \(\mu=E/(2(1+\nu))\).

The FEM weak form used after the strong form is: find \(\mathbf{u}\in V\) with prescribed Dirichlet data such that

\[
\int_{\Omega} \boldsymbol{\epsilon}(\mathbf{v}):\boldsymbol{\sigma}(\mathbf{u})\,d\Omega
=\int_{\Omega}\mathbf{v}\cdot\mathbf{b}\,d\Omega+\int_{\Gamma_t}\mathbf{v}\cdot\bar{\mathbf{t}}\,d\Gamma
\quad \forall \mathbf{v}\in V_0.
\]

The current scripts use \(\mathbf{b}=0\) and a smooth downward pressure distribution on the top load patch,
with the same patch profile in FEM load integration and PINN traction residuals.

## Multilayer Coupling

Layers are ordered bottom-to-top. Each layer has its own Young's modulus and thickness: one-layer cases use `(E, thickness)`, while three-layer cases use `(E1,t1,E2,t2,E3,t3)`. FEM assigns each Hex8 element to a material by its centroid `z` coordinate and cumulative layer thicknesses. Interface nodes are shared, so displacement continuity is enforced by the global displacement DOFs; traction continuity follows from assembled equilibrium across adjacent elements.

The P2INN/PINN uses one continuous network output for the full stack and selects local material parameters by comparing the point coordinate `z` with `t1` and `t1+t2`. The layer ordering matters because the bottom, middle, and top stiffnesses are sampled at different `z` locations, so swapping `E1,E2,E3` changes the stress path and final deflection.

## PINN/P2INN Losses

Training logs save individual components in `loss_history_components.csv`. The total loss is

\[
\mathcal{L}=w_{pde}\mathcal{L}_{pde}+w_{bc}\mathcal{L}_{bc}+w_{load}\mathcal{L}_{load}
+w_{int}\mathcal{L}_{int}+w_{data}\mathcal{L}_{data}+w_E\mathcal{L}_{energy}.
\]

The primary terms are \(\mathcal{L}_{pde}=\|-\nabla\cdot\sigma(u)-b\|_2^2\), side Dirichlet and free-surface traction boundary losses in \(\mathcal{L}_{bc}\), top-patch traction loss \(\mathcal{L}_{load}=\|\sigma(u)n-\bar{t}\|_2^2\), interface traction-continuity loss \(\mathcal{L}_{int}\), and sparse FEM supervision \(\mathcal{L}_{data}=\|u_\theta-u_{FEM}\|_2^2\) when enabled. The default three-layer model uses sparse FEM supervision from 36,000 sampled FEM nodal values across the configured training parameter grid; one-layer defaults are physics-only unless `PINN_USE_SUPERVISION_DATA=1`.

## Repository Structure

```
├── compare_three_layer_pinn_fem.py  # 3-layer validation
├── compare_two_layer_pinn_fem.py    # 2-layer validation
├── pinn-workflow/         # PINN training code
│   ├── train.py           # Main training script
│   ├── model.py           # Network architecture
│   ├── physics.py         # Physics loss functions
│   ├── data.py            # Data sampling
│   └── pinn_config.py     # Training configuration
├── fea-workflow/          # FEA solver
│   └── solver/
│       └── fem_solver.py  # Hex8 FEM implementation
└── pinn-workflow-2layer/  # 2-layer PINN (legacy)
```

## Quick Start

### Installation

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Train the PINN

```bash
cd pinn-workflow
python train.py
```

The trained model is saved as `pinn_model.pth`.

### Validate Against FEA

```bash
# 3-layer validation
python compare_three_layer_pinn_fem.py

# 2-layer validation (requires 2-layer model)
python compare_two_layer_pinn_fem.py
```

### Generate Paper Figures

```bash
python3 graphs/make_all_graphs.py
```

## PINN Architecture

The PINN (`MultiLayerPINN`) predicts 3D displacement fields:

**Inputs** (12 features):
- Spatial: `x, y, z`
- Material: `E1, t1, E2, t2, E3, t3` (Young's modulus and thickness per layer)
- Impact: `restitution, friction, impact_velocity`

**Outputs** (3 components):
- Displacement: `ux, uy, uz`

**Network**: Fully-connected MLP with:
- Input normalization using configured parameter ranges
- Derived features (interface indicators, bending terms)
- Optional hard boundary condition enforcement

### Compliance Scaling

The network outputs `v` which is converted to displacement `u` via:
```
u = scale * v / E^p * (H/t)^alpha
```

where `p` (`E_COMPLIANCE_POWER`) and `alpha` (`THICKNESS_COMPLIANCE_ALPHA`) are configurable.

## Training

### Configuration

Key parameters in `pinn_config.py`:

```python
# Loss weights
WEIGHTS = {
    'pde': 10.0,         # Equilibrium equation
    'bc': 0.7,           # Boundary conditions
    'load': 5.0,         # Load patch traction
    'energy': 0.63,      # Energy consistency
    'interface_u': 300.0,# Interface continuity
    'data': 400.0,       # FEA supervision
}

# Sampling
N_INTERIOR = 15000     # Interior collocation points
N_SIDES = 2000         # Side boundary points
N_TOP_LOAD = 6000      # Load patch points
N_TOP_FREE = 2000      # Free surface points
N_INTERFACE = 16000    # Interface points

# Training
EPOCHS_SOAP = 400      # SOAP optimizer steps
EPOCHS_LBFGS = 0       # L-BFGS fine-tuning steps
LEARNING_RATE = 1e-3
```

### Environment Variables

Override config without editing files:

```bash
# Device selection
export PINN_DEVICE=cuda

# Loss weights
export PINN_W_PDE=20.0
export PINN_W_INTERFACE_U=500.0
export PINN_W_LOAD=10.0

# Supervision data
export PINN_DATA_E_VALUES="1.0,5.0,10.0"
export PINN_DATA_T1_VALUES="0.02,0.06,0.10"

# Warm start
export PINN_WARM_START=1

# Output directory
export PINN_OUT_DIR=/path/to/outputs
```

### Training Process

1. **SOAP Optimization**: Main training using second-order preconditioning
2. **Adaptive Resampling**: Every 500 epochs, resample based on PDE residuals
3. **L-BFGS Fine-tuning**: Optional second-stage optimization
4. **Checkpointing**: Model saved after each L-BFGS step

## FEA Solver

The FEA solver (`fem_solver.py`) implements:
- Hex8 elements with 2x2x2 Gauss quadrature
- Layer-wise material properties
- Penalty method for clamped boundaries
- Load patch with smooth distribution

Functions:
- `solve_fem()`: Single-layer baseline
- `solve_two_layer_fem()`: Two-layer structure
- `solve_three_layer_fem()`: Three-layer structure

## Validation

### Comparison Scripts

Generate side-by-side contour plots:

```bash
python compare_three_layer_pinn_fem.py
```

Creates visualizations in `pinn-workflow/visualization_three_layer/`:
- `{case}_top.png`: Top surface displacement and error
- `{case}_cross_section.png`: Cross-section displacement and error
- `three_layer_sweep_tmin_E2*.png`: MAE heatmaps across E1-E3 space

### Current Accuracy Reporting

Benchmark scripts now report relative L2/integral error, mean absolute error,
average displacement error, and max error as a secondary diagnostic. Current
refined-mesh checks show that the saved one-layer and three-layer checkpoints do
not both reach the <5% target on held-out top-surface metrics. With transparent
compliance recalibration, the one-layer held-out top-surface MAE is below 5% on
the tested set, while the three-layer checkpoint remains above target on some
cases. Retrain or recalibrate the three-layer model against the corrected smooth
load-patch formulation before claiming universal <5% top-surface agreement.

## Configuration Reference

### Geometry (`pinn_config.py`)

```python
Lx = 1.0          # Plate length (x)
Ly = 1.0          # Plate width (y)
H = 0.1           # Total thickness (z)
NUM_LAYERS = 3    # Number of material layers
```

### Material Parameters

```python
E_RANGE = [1.0, 10.0]           # Young's modulus range
T1_RANGE = [0.02, 0.10]         # Layer 1 thickness
T2_RANGE = [0.02, 0.10]         # Layer 2 thickness
T3_RANGE = [0.02, 0.10]         # Layer 3 thickness
nu_vals = [0.3]                 # Poisson's ratio
```

### Loading

```python
p0 = 1.0                        # Load magnitude
LOAD_PATCH_X = [Lx/3, 2*Lx/3]  # Load patch x-range
LOAD_PATCH_Y = [Ly/3, 2*Ly/3]  # Load patch y-range
```

## Troubleshooting

### Low Displacement Magnitude
- Increase `WEIGHTS['load']` or `WEIGHTS['energy']`
- Check compliance scaling parameters
- Verify FEA supervision data is loading (if enabled)

### High MAE on Validation
- Increase training epochs
- Adjust loss weights (increase `WEIGHTS['data']` if using supervision)
- Enable adaptive resampling
- Check interface continuity loss weight

### Convergence Issues
- Reduce learning rate
- Adjust SOAP precondition frequency
- Check for NaN in losses (may indicate sampling issues)

## Citation

If using this code, please cite relevant PINN literature and acknowledge the physics-informed neural network approach for multi-layered structures.

## License

This project is provided as-is for research and educational purposes.
