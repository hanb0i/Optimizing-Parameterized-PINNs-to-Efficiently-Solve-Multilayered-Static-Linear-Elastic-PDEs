#!/usr/bin/env python3
"""FEM mesh convergence study for one-layer and three-layer models."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "serif"]
plt.rcParams["mathtext.fontset"] = "dejavuserif"
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "graphs" / "generalized_study" / "fem_convergence"
DATA_DIR = REPO_ROOT / "graphs" / "generalized_study" / "fem_convergence"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

FEA_DIR = REPO_ROOT / "fea-workflow" / "solver"
sys.path.insert(0, str(FEA_DIR))

import fem_solver


def _run_one_layer_fem(E_val: float, thickness: float, ne_x: int, ne_y: int, ne_z: int):
    cfg = {
        "geometry": {"Lx": 1.0, "Ly": 1.0, "H": thickness, "ne_x": ne_x, "ne_y": ne_y, "ne_z": ne_z},
        "material": {"E": E_val, "nu": 0.3},
        "load_patch": {
            "pressure": 1.0,
            "x_start": 1.0 / 3.0,
            "x_end": 2.0 / 3.0,
            "y_start": 1.0 / 3.0,
            "y_end": 2.0 / 3.0,
        },
    }
    x_nodes, y_nodes, z_nodes, u_grid = fem_solver.solve_fem(cfg)
    u_grid = np.array(u_grid, dtype=float)
    u_z_top = np.array(u_grid, dtype=float)[:, :, -1, 2]
    return {
        "peak_uz": float(np.min(u_z_top)),
        "avg_uz": float(np.mean(u_grid[..., 2])),
        "top_avg_uz": float(np.mean(u_z_top)),
        "top_mean_abs_uz": float(np.mean(np.abs(u_z_top))),
        "rms_uz": float(np.sqrt(np.mean(u_grid[..., 2] ** 2))),
    }


def _run_three_layer_fem(e1, e2, e3, t1, t2, t3, ne_x: int, ne_y: int, ne_z: int):
    thickness = t1 + t2 + t3
    cfg = {
        "geometry": {"Lx": 1.0, "Ly": 1.0, "H": thickness, "ne_x": ne_x, "ne_y": ne_y, "ne_z": ne_z},
        "material": {"E_layers": [float(e1), float(e2), float(e3)], "t_layers": [float(t1), float(t2), float(t3)], "nu": 0.3},
        "load_patch": {
            "pressure": 1.0,
            "x_start": 1.0 / 3.0,
            "x_end": 2.0 / 3.0,
            "y_start": 1.0 / 3.0,
            "y_end": 2.0 / 3.0,
        },
    }
    x_nodes, y_nodes, z_nodes, u_grid = fem_solver.solve_three_layer_fem(cfg)
    u_grid = np.array(u_grid, dtype=float)
    u_z_top = np.array(u_grid, dtype=float)[:, :, -1, 2]
    return {
        "peak_uz": float(np.min(u_z_top)),
        "avg_uz": float(np.mean(u_grid[..., 2])),
        "top_avg_uz": float(np.mean(u_z_top)),
        "top_mean_abs_uz": float(np.mean(np.abs(u_z_top))),
        "rms_uz": float(np.sqrt(np.mean(u_grid[..., 2] ** 2))),
    }


def _convergence_study(name: str, run_fn, mesh_sequence: list[tuple[int, int, int]]):
    results = []
    print(f"\n=== {name} ===")
    for ne_x, ne_y, ne_z in mesh_sequence:
        print(f"  Running mesh {ne_x}x{ne_y}x{ne_z} ...", end=" ", flush=True)
        metrics = run_fn(ne_x, ne_y, ne_z)
        h = 1.0 / ne_x
        n_elements = ne_x * ne_y * ne_z
        row = {"ne_x": ne_x, "ne_y": ne_y, "ne_z": ne_z, "h": h, "n_elements": n_elements}
        row.update(metrics)
        results.append(row)
        print(f"peak_uz={metrics['peak_uz']:.6f}, top_avg_uz={metrics['top_avg_uz']:.6f}")

    # Compute relative error vs finest mesh
    finest = results[-1]
    for r in results:
        r["peak_rel_err"] = abs((r["peak_uz"] - finest["peak_uz"]) / finest["peak_uz"]) if finest["peak_uz"] != 0 else 0.0
        r["avg_rel_err"] = abs((r["avg_uz"] - finest["avg_uz"]) / finest["avg_uz"]) if finest["avg_uz"] != 0 else 0.0
        r["top_avg_rel_err"] = abs((r["top_avg_uz"] - finest["top_avg_uz"]) / finest["top_avg_uz"]) if finest["top_avg_uz"] != 0 else 0.0
        r["rms_rel_err"] = abs((r["rms_uz"] - finest["rms_uz"]) / finest["rms_uz"]) if finest["rms_uz"] != 0 else 0.0

    # Save CSV
    csv_path = DATA_DIR / f"{name}_convergence.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ne_x", "ne_y", "ne_z", "h", "n_elements",
                "peak_uz", "avg_uz", "top_avg_uz", "top_mean_abs_uz", "rms_uz",
                "peak_rel_err", "avg_rel_err", "top_avg_rel_err", "rms_rel_err",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"  Saved: {csv_path}")

    return results


def _make_combined_4panel():
    """Create a single 4-panel figure matching the target style with OUR data."""
    
    # Load data
    one_data = []
    with open(DATA_DIR / "one_layer_convergence.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            one_data.append({k: float(v) for k, v in row.items()})
    
    three_data = []
    with open(DATA_DIR / "three_layer_convergence.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            three_data.append({k: float(v) for k, v in row.items()})
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "FEM Solver Validation: Mesh Convergence Study\n"
        "Benchmark mesh (16$\\times$16$\\times$8) indicated by red dashed line",
        fontsize=14, fontweight="bold", y=0.98
    )
    
    # Our benchmark mesh: 16x16x8 -> h = 1/16 = 0.0625
    h_benchmark = 0.0625
    bench_idx = 2  # 16x16x8 is index 2 in our sequence
    
    # --- Panel (a): One-layer peak displacement ---
    h1 = np.array([r["h"] for r in one_data])
    peak1 = np.array([abs(r["peak_uz"]) for r in one_data])
    
    axes[0,0].plot(h1, peak1, "o-", color="#1f77b4", linewidth=2, markersize=10, label="FEM solution", zorder=3)
    axes[0,0].axvline(x=h_benchmark, color="#d62728", linestyle="--", linewidth=1.5, 
                      label="Benchmark mesh (16$\\times$16$\\times$8)", zorder=2)
    
    # Benchmark annotation (inside plot, upper left)
    bench_val = peak1[bench_idx]
    axes[0,0].annotate(
        f"Benchmark: $|u_z|={bench_val:.3f}$",
        xy=(h_benchmark, bench_val),
        xytext=(0.20, 0.72),
        fontsize=10,
        color="#d62728",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#d62728", alpha=0.9),
        arrowprops=dict(arrowstyle="-", color="#d62728", lw=1.2),
        zorder=5,
    )
    
    axes[0,0].set_xlabel("Mesh size $h = 1/n_{e,x}$", fontsize=11)
    axes[0,0].set_ylabel("$|\\max u_z|$", fontsize=11)
    axes[0,0].set_title("(a) One-Layer: Peak Displacement vs Mesh Size\n$E=1$, $t=0.15$", fontsize=11)
    axes[0,0].invert_xaxis()
    axes[0,0].legend(loc="lower right", fontsize=9)
    axes[0,0].grid(True, alpha=0.3)
    
    # --- Panel (b): One-layer convergence rate ---
    err1 = np.array([r["peak_rel_err"] for r in one_data])
    
    axes[0,1].loglog(h1, err1, "o-", color="#1f77b4", linewidth=2, markersize=10, label="FEM convergence", zorder=3)
    h_ref = np.array([h1[0], h1[-1]])
    axes[0,1].loglog(h_ref, h_ref**2 * err1[0] / h1[0]**2, "k--", alpha=0.5, label="O(h$^2$) reference", zorder=2)
    axes[0,1].axvline(x=h_benchmark, color="#d62728", linestyle="--", linewidth=1.5, zorder=2)
    
    axes[0,1].set_xlabel("Mesh size $h$", fontsize=11)
    axes[0,1].set_ylabel("Relative error vs finest mesh", fontsize=11)
    axes[0,1].set_title("(b) One-Layer: Convergence Rate", fontsize=11)
    axes[0,1].invert_xaxis()
    axes[0,1].legend(loc="upper left", fontsize=9)
    axes[0,1].grid(True, alpha=0.3, which="both")
    
    # --- Panel (c): Three-layer peak displacement ---
    h3 = np.array([r["h"] for r in three_data])
    peak3 = np.array([abs(r["peak_uz"]) for r in three_data])
    
    axes[1,0].plot(h3, peak3, "o-", color="#2ca02c", linewidth=2, markersize=10, label="FEM solution", zorder=3)
    axes[1,0].axvline(x=h_benchmark, color="#d62728", linestyle="--", linewidth=1.5, 
                      label="Benchmark mesh (16$\\times$16$\\times$8)", zorder=2)
    
    # Benchmark annotation (inside plot, upper left)
    bench_val3 = peak3[bench_idx]
    axes[1,0].annotate(
        f"Benchmark: $|u_z|={bench_val3:.4f}$",
        xy=(h_benchmark, bench_val3),
        xytext=(0.20, 0.122),
        fontsize=10,
        color="#d62728",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#d62728", alpha=0.9),
        arrowprops=dict(arrowstyle="-", color="#d62728", lw=1.2),
        zorder=5,
    )
    
    axes[1,0].set_xlabel("Mesh size $h = 1/n_{e,x}$", fontsize=11)
    axes[1,0].set_ylabel("$|\\max u_z|$", fontsize=11)
    axes[1,0].set_title("(c) Three-Layer: Peak Displacement vs Mesh Size\n$E=[10,10,10]$, $t=[0.02,0.10,0.02]$", fontsize=11)
    axes[1,0].invert_xaxis()
    axes[1,0].legend(loc="lower right", fontsize=9)
    axes[1,0].grid(True, alpha=0.3)
    
    # --- Panel (d): Three-layer convergence rate ---
    err3 = np.array([r["peak_rel_err"] for r in three_data])
    
    axes[1,1].loglog(h3, err3, "o-", color="#2ca02c", linewidth=2, markersize=10, label="FEM convergence", zorder=3)
    h_ref3 = np.array([h3[0], h3[-1]])
    axes[1,1].loglog(h_ref3, h_ref3**2 * err3[0] / h3[0]**2, "k--", alpha=0.5, label="O(h$^2$) reference", zorder=2)
    axes[1,1].axvline(x=h_benchmark, color="#d62728", linestyle="--", linewidth=1.5, zorder=2)
    
    axes[1,1].set_xlabel("Mesh size $h$", fontsize=11)
    axes[1,1].set_ylabel("Relative error vs finest mesh", fontsize=11)
    axes[1,1].set_title("(d) Three-Layer: Convergence Rate", fontsize=11)
    axes[1,1].invert_xaxis()
    axes[1,1].legend(loc="upper left", fontsize=9)
    axes[1,1].grid(True, alpha=0.3, which="both")
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    out_pdf = REPO_ROOT / "fem_convergence_4panel.pdf"
    out_png = REPO_ROOT / "fem_convergence_4panel.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved combined figure:\n  {out_pdf}\n  {out_png}")


def main():
    # Mesh sequence: coarse → fine
    mesh_sequence = [
        (4, 4, 2),
        (8, 8, 4),
        (16, 16, 8),
        (32, 32, 16),
    ]

    # One-layer: E=1.0, t=0.15 (within parameter range, <5% error at 16x16x8)
    _convergence_study(
        "one_layer",
        lambda nx, ny, nz: _run_one_layer_fem(E_val=1.0, thickness=0.15, ne_x=nx, ne_y=ny, ne_z=nz),
        mesh_sequence,
    )

    # Three-layer: representative case
    _convergence_study(
        "three_layer",
        lambda nx, ny, nz: _run_three_layer_fem(
            e1=10.0, e2=10.0, e3=10.0,
            t1=0.02, t2=0.10, t3=0.02,
            ne_x=nx, ne_y=ny, ne_z=nz,
        ),
        mesh_sequence,
    )

    # Generate combined 4-panel figure
    _make_combined_4panel()

    print("\nConvergence study complete.")


if __name__ == "__main__":
    main()
