"""Utility to inspect PINN checkpoint structure.

Prints the shape of all tensors in the checkpoint for debugging
and compatibility checking.

Usage:
    python inspect_checkpoint.py
"""

import torch
import os

path = "pinn-workflow/pinn_model.pth"
if os.path.exists(path):
    state = torch.load(path, map_location='cpu')
    print("Checkpoint Keys:")
    for k, v in state.items():
        print(f"{k}: {v.shape}")
else:
    print(f"File not found: {path}")

import sys
sys.path.append('pinn-workflow')
import model
pinn = model.MultiLayerPINN()
print("\nInitialized Model Keys:")
for k, v in pinn.state_dict().items():
    print(f"{k}: {v.shape}")
