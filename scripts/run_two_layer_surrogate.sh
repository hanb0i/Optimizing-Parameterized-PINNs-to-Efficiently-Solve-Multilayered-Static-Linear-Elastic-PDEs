#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR=.mplconfig
export PYTHONPYCACHEPREFIX=.pycache

# Build a 2-layer surrogate (E1,t1,E2,t2) by embedding into the repo's 3-layer PINN layout.
export SURROGATE_NUM_LAYERS=2

# Keep the same compliance scaling used in the two-layer comparisons.
export PINN_E_COMPLIANCE_POWER=0.95 PINN_THICKNESS_COMPLIANCE_ALPHA=3 PINN_DISPLACEMENT_COMPLIANCE_SCALE=1

# Surrogate training knobs (tuned to hit <5% error on the corner/sweep cases).
export SURROGATE_ANCHOR_REPEAT="${SURROGATE_ANCHOR_REPEAT:-40}"
export SURROGATE_HIDDEN_UNITS="${SURROGATE_HIDDEN_UNITS:-512}"
export SURROGATE_HIDDEN_LAYERS="${SURROGATE_HIDDEN_LAYERS:-5}"
export SURROGATE_ACTIVATION="${SURROGATE_ACTIVATION:-relu}"
export SURROGATE_LEARNING_RATE="${SURROGATE_LEARNING_RATE:-5e-4}"
export SURROGATE_PATIENCE="${SURROGATE_PATIENCE:-800}"

# Baseline response grid resolution used when querying the PINN under the load patch.
export SURROGATE_TOP_NX="${SURROGATE_TOP_NX:-11}"

python3 pinn-workflow/surrogate_workflow/run_phase1.py --regenerate "${@}"
python3 pinn-workflow/surrogate_workflow/verify_phase1.py
