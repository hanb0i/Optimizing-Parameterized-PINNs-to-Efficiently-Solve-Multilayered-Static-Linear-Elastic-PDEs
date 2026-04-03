import os

import numpy as np

import pinn_config as pc

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Number of layers for the surrogate design vector.
# Default matches this repo's current 3-layer PINN, but can be overridden to build
# a 2-layer surrogate (embedded into the 3-layer PINN input layout).
NUM_LAYERS = int(os.getenv("SURROGATE_NUM_LAYERS", "3"))

# Output paths
OUTPUT_DIR = os.path.join(ROOT_DIR, "surrogate_workflow", "outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
DATASET_PATH = os.path.join(OUTPUT_DIR, "phase1_dataset.npz")
MODEL_PATH = os.path.join(OUTPUT_DIR, "surrogate_model.pt")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "phase1_summary.txt")

def _t_range(i: int):
    name = f"T{i}_RANGE"
    if hasattr(pc, name):
        lo, hi = getattr(pc, name)
        return float(lo), float(hi)
    h = float(getattr(pc, "H", 0.1))
    guess = h / max(int(NUM_LAYERS), 1)
    return float(guess), float(guess)


# Design parameters for a multi-layer plate PINN surrogate (E_i, t_i for each layer).
DESIGN_PARAMS = []
DESIGN_RANGES = {}
for i in range(1, int(NUM_LAYERS) + 1):
    DESIGN_PARAMS.extend([f"E{i}", f"t{i}"])
    DESIGN_RANGES[f"E{i}"] = (float(pc.E_RANGE[0]), float(pc.E_RANGE[1]))
    DESIGN_RANGES[f"t{i}"] = _t_range(i)

# Dataset generation
N_SAMPLES = int(os.getenv("SURROGATE_N_SAMPLES", "2000"))
SEED = int(os.getenv("SURROGATE_SEED", "7"))
TRAIN_FRACTION = float(os.getenv("SURROGATE_TRAIN_FRACTION", "0.8"))
VAL_FRACTION = float(os.getenv("SURROGATE_VAL_FRACTION", "0.1"))

# Model hyperparameters
HIDDEN_LAYERS = int(os.getenv("SURROGATE_HIDDEN_LAYERS", "4"))
HIDDEN_UNITS = int(os.getenv("SURROGATE_HIDDEN_UNITS", "256"))
ACTIVATION = os.getenv("SURROGATE_ACTIVATION", "tanh")
FOURIER_DIM = int(os.getenv("SURROGATE_FOURIER_DIM", "0"))
FOURIER_SCALE = float(os.getenv("SURROGATE_FOURIER_SCALE", "1.0"))

# Training
LEARNING_RATE = float(os.getenv("SURROGATE_LEARNING_RATE", "1e-3"))
MAX_EPOCHS = int(os.getenv("SURROGATE_MAX_EPOCHS", "6000"))
BATCH_SIZE = int(os.getenv("SURROGATE_BATCH_SIZE", "64"))
PATIENCE = int(os.getenv("SURROGATE_PATIENCE", "400"))
MIN_DELTA = float(os.getenv("SURROGATE_MIN_DELTA", "1e-6"))

# Oversample corner anchors in the training loader to reduce worst-case errors at
# thickness/E extremes (used by PINN sweep comparisons).
ANCHOR_REPEAT = int(os.getenv("SURROGATE_ANCHOR_REPEAT", "1"))

# Validation and sweeps
TREND_SWEEP_PARAM = "E1"
TREND_SWEEP_POINTS = 60
OPT_CANDIDATES = 2000
TREND_ANCHOR_POINTS = 25

# Add a small set of (low/high) corners to reduce worst-case errors near extremes.
CORNER_ANCHORS = True

# Output transform.
Y_TRANSFORM = "log"  # "identity" | "log"
Y_EPS = 1e-6

# Loss mode: with `Y_TRANSFORM="log"`, plain MSE effectively targets relative accuracy.
LOSS_MODE = "mse"  # "mse" | "relative_mse"
RELATIVE_LOSS_EPS = 1e-3

def mid_design() -> np.ndarray:
    return np.array([(DESIGN_RANGES[name][0] + DESIGN_RANGES[name][1]) * 0.5 for name in DESIGN_PARAMS], dtype=float)
