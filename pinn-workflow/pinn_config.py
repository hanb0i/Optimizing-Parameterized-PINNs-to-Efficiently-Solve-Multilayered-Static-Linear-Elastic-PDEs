import torch
import numpy as np

# --- Geometry Dimensions ---
Lx = 1.0
Ly = 1.0
H = 0.1  # Total height (baseline thickness)

# --- Geometry Configuration ---
GEOMETRY_TYPE = "FLAT" # Options: "FLAT", "DENT"
DENT_DEPTH = 0.0      # Amplitude of the dent (z_min at center = H - DENT_DEPTH)
DENT_WIDTH = 0.2       # Sigma (spread) of the Gaussian
DENT_CENTER_X = Lx / 2.0
DENT_CENTER_Y = Ly / 2.0

def get_domain_height(x, y):
    """
    Returns the top surface z-coordinate at (x, y).
    z_top = H - A * exp(-r^2 / (2*sigma^2))
    """
    if GEOMETRY_TYPE == "FLAT":
        return torch.full_like(x, H)
    elif GEOMETRY_TYPE == "DENT":
        r2 = (x - DENT_CENTER_X)**2 + (y - DENT_CENTER_Y)**2
        dent = DENT_DEPTH * torch.exp(-r2 / (2 * DENT_WIDTH**2))
        return H - dent
    return torch.full_like(x, H)

# 3-Layer Geometry (Sandwich Plate)
# Top/Bot = 0.02, Core = 0.06
LAYER_THICKNESSES = [0.02, 0.06, 0.02]
# Ratios relative to total local thickness: [0.2, 0.8]
LAYER_Z_RATIOS = [0.2, 0.8] 
LAYER_Z_RANGES = [
    [0.0, 0.02],          # Bottom Face Sheets (Wait, Z=0 is usually bottom)
    [0.02, 0.08],         # Core
    [0.08, 0.1]           # Top Face Sheets
]
# Material Properties per Layer
LAYER_E_VALS = [10.0, 1.0, 10.0] # Stiff-Soft-Stiff
LAYER_NU_VALS = [0.3, 0.3, 0.3]

# Parameterized PINN settings unused in Phase 5
E_vals = [1.0] 
nu_vals = [0.3]
# Parameterized PINN settings for 12D Multi-Layer
# Params: [E1, E2, E3, t1, t2, t3, r, mu, v0] = 9 params
E_RANGE = [1.0, 20.0]
THICKNESS_RANGE = [0.01, 0.08] # Per-layer range
RESTITUTION_RANGE = [0.1, 0.9]
FRICTION_RANGE = [0.0, 0.6]
IMPACT_VELOCITY_RANGE = [0.2, 2.0]

# Baseline values for evaluation
E1_vals = [10.0]
E2_vals = [1.0]
E3_vals = [10.0]
t1_vals = [0.02]
t2_vals = [0.06]
t3_vals = [0.02]
PARAM_DIM = 9 

# ... (Reference values) ...
RESTITUTION_REF = 0.5
FRICTION_REF = 0.3
IMPACT_VELOCITY_REF = 1.0

# --- Parametric compliance scaling ---
THICKNESS_COMPLIANCE_ALPHA = 1.6 
E_COMPLIANCE_POWER = 1.0        

def get_lame_params(E, nu):
    lm = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lm, mu

Lame_Params = [get_lame_params(e, n) for e, n in zip(E_vals, nu_vals)]

# --- Loading ---
p0 = 1.0 # Load magnitude

# --- Unit-consistent loss scaling ---
PDE_LENGTH_SCALE = H
OUTPUT_SCALE_Z = 10.0 # Crucial factor for Tanh stability

# --- Boundary condition handling ---
USE_HARD_SIDE_BC = False 
HARD_BC_EPOCHS = 0
FORCE_SOFT_SIDE_BC_FROM_START = True

# Load patch boundaries (normalized coordinates)
LOAD_PATCH_X = [Lx/3, 2*Lx/3]
LOAD_PATCH_Y = [Ly/3, 2*Ly/3]

# --- Network Architecture ---
LAYERS = 6
NEURONS = 64

# --- Training Hyperparameters ---
LEARNING_RATE = 1e-3
EPOCHS_ADAM = 2500 
EPOCHS_LBFGS = 500 
SOAP_PRECONDITION_FREQUENCY = 10

WEIGHTS = {
    'pde': 10.0,
    'bc': 1.0,
    'load': 100.0,   # Increase to force magnitude
    'energy': 0.1,
    'impact_invariance': 0.0,
    'impact_contact': 0.0002,
    'friction_coulomb': 0.001,
    'friction_stick': 0.0005,
    'interface_u': 50.0,      # High interface continuity
    'interface_traction': 50.0, # Stress continuity
    'data': 1.0
}

# Loss weight ramp: load-first to raise displacement while preserving shape.
WEIGHT_RAMP_EPOCHS = 0
LOAD_WEIGHT_START = WEIGHTS['load']
PDE_WEIGHT_START = WEIGHTS['pde']
ENERGY_WEIGHT_START = WEIGHTS['energy']
# Force soft side boundary conditions from the beginning.
FORCE_SOFT_SIDE_BC_FROM_START = True
SOFT_MODE_PDE_WEIGHT_SCALE = 1.0 # Simplified: No auto-scaling, set base weights directly
SOFT_MODE_LOAD_WEIGHT_SCALE = 1.0
# Sampling
N_INTERIOR = 4000 # Per layer
N_SIDES = 1000  # Clamped side faces
N_TOP_LOAD = 2000  # Load patch (more points to boost displacement)
N_TOP_FREE = 1000  # Top free surface
N_BOTTOM = 1000  # Bottom free surface
UNDER_PATCH_FRACTION = 0.95 # More interior points focus under the load patch

#Resampling/perturbation control
SAMPLING_NOISE_SCALE = 0.08  # Larger perturbations widen coverage while still sampling residual-rich zones.

# Auxiliary load-patch average displacement penalty
LOAD_PATCH_UZ_TARGET = -0.05  # Encourage the mean vertical deflection on the load patch
LOAD_PATCH_UZ_WEIGHT = 0.02   # Keep the auxiliary penalty small so shape stays intact

# Fourier Features
FOURIER_DIM = 0 # Reverted to 0 to analyze Experiment B checkpoint
FOURIER_SCALE = 1.0 # Standard deviation for frequency sampling

# Hybrid / Parametric Training Data
N_DATA_POINTS = 9000
DATA_E_VALUES = [1.0, 5.0, 10.0]
DATA_THICKNESS_VALUES = [0.05, 0.1, 0.15]
USE_SUPERVISION_DATA = False

# --- Explicit impact/friction physics controls ---
# When enabled, restitution/friction influence boundary losses directly.
USE_EXPLICIT_IMPACT_PHYSICS = True
# If True, keeps restitution/friction neutral (used before explicit physics).
ENFORCE_IMPACT_INVARIANCE = False
# Restitution-coupled load amplification gain.
IMPACT_RESTITUTION_GAIN = 0.03
# Impact-velocity gain for dynamic traction amplification.
IMPACT_VELOCITY_GAIN = 0.03
