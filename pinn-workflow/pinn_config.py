
import torch
import numpy as np

# --- Geometry Dimensions ---
Lx = 1.0
Ly = 1.0
H = 0.1  # Total height
Layer_Interfaces = [0.0, H]

# --- Material Properties ---
<<<<<<< HEAD
# Young's Modulus (E) and Poisson's Ratio (nu)
<<<<<<< HEAD
# Can be different per layer
E_vals = [360.0, 360.0, 360.0] # Match FEA material
nu_vals = [0.3, 0.3, 0.3]
=======
# Single layer to match FEM
E_vals = [1.0] # Normalized
=======
E_vals = [1.0]  # Normalized
>>>>>>> 3176abcf323e43483e790d268adaa1838f1907f2
nu_vals = [0.3]
>>>>>>> a204439ef0cee6b426c4e683743f2eee33c9b01a

def get_lame_params(E, nu):
    lm = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lm, mu

Lame_Params = [get_lame_params(e, n) for e, n in zip(E_vals, nu_vals)]

# --- Loading ---
<<<<<<< HEAD
p0 = 0.1 # Load magnitude

# Load patch boundaries (normalized coordinates)
LOAD_PATCH_X = [Lx/3, 2*Lx/3]  # [0.333, 0.667]
LOAD_PATCH_Y = [Ly/3, 2*Ly/3]  # [0.333, 0.667]
USE_LOAD_MASK = False  # Match FEM's uniform patch pressure by default
LOAD_MASK_SCALE = 1.0  # Use to rescale masked pressure if enabled

# --- Training Hyperparameters ---
LEARNING_RATE = 1e-3
<<<<<<< HEAD
EPOCHS_ADAM = 1000 # Optimal balance (Knee point at ~200-500)
EPOCHS_LBFGS = 1500 # Testing user's hypothesis (1000-2000 range)
WEIGHTS = {
    'pde': 1.0,    # Balanced with Load
    'bc': 100.0,    # Strong constraint (Trial 12)
    'load': 100.0,   # Natural Physics (Trial 12)
    'interface_u': 100.0, # Balanced (Trial 12)
    'interface_t': 1.0   # Matches Traction (Trial 12)
}
# Sampling
N_INTERIOR = 2000 # Standard resolution (Trial 12)
N_BOUNDARY = 2000  # Standard resolution

# Fourier Features
FOURIER_DIM = 64 # Number of Fourier frequencies
FOURIER_SCALE = 5.0 # Increased to capture sharp load edges
OUTPUT_SCALE = 1.0 # Removed scaling to allow natural physics-driven magnitude
=======
EPOCHS_ADAM = 2000 # Increased to enforce load and reduce underfit
EPOCHS_LBFGS = 30 # Increased from 500. Resampling here. Should help convergence. 
#Plot Physical Residuals Every N Epochs every 100 epochs. 
WEIGHTS = {
    'pde': 1.0,    # Increased from 1.0
    'bc': 1.0,      # Reduced, as hard constraint handles side BCs now
    'load': 1.0, # Heavily increased to match traction target
    'interface_u': 1.0 
}
# Sampling
N_INTERIOR = 10000 # Per layer
N_BOUNDARY = 2000  # Per face type
=======
p0 = 1.0  # Load magnitude
LOAD_PATCH_X = [Lx/3, 2*Lx/3]
LOAD_PATCH_Y = [Ly/3, 2*Ly/3]

# --- Training Hyperparameters ---
LEARNING_RATE = 1e-3
EPOCHS_SOAP = 2000
EPOCHS_SSBFGS = 30
>>>>>>> 3176abcf323e43483e790d268adaa1838f1907f2

# SOAP optimizer
SOAP_PRECONDITION_FREQUENCY = 10

# SciPy self-scaled BFGS optimizer
SS_BFGS_METHOD = "BFGS"
SS_BFGS_VARIANT = "SSBFGS_AB"
SS_BFGS_MAXITER = 1
SS_BFGS_GTOL = 0.0
SS_BFGS_INITIAL_SCALE = False

# Loss Weights
WEIGHTS = {
    'pde': 1.0,
    'bc': 1.0,
    'load': 1.0,
}

# Sampling
N_INTERIOR = 10000
N_BOUNDARY = 2000

# Output Scaling
<<<<<<< HEAD
OUTPUT_SCALE = 3.55 # Scaling factor for network output
>>>>>>> a204439ef0cee6b426c4e683743f2eee33c9b01a
=======
OUTPUT_SCALE = 3.55
>>>>>>> 3176abcf323e43483e790d268adaa1838f1907f2
