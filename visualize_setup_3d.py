import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

def visualize_3d_setup(E_vals, thicknesses, out_path="pinn-workflow/visualization/setup_3d_layered.png"):
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Rectangular footprint
    Lx, Ly = 2.0, 1.0
    
    # Material palette
    colors = ['#4e79a7', '#f28e2b', '#e15759'] # Blue, Orange, Red
    
    # Plot layers
    z_bottom = 0.0
    for i, (E, t) in enumerate(zip(E_vals, thicknesses)):
        z_top = z_bottom + t
        color = colors[i]
        
        # Surfaces
        xx, yy = np.meshgrid([0, Lx], [0, Ly])
        ax.plot_surface(xx, yy, np.full_like(xx, z_top), color=color, alpha=0.5, edgecolor='black', lw=0.5)
        
        # Vertical walls
        yy_wall, zz_wall = np.meshgrid([0, Ly], [z_bottom, z_top])
        ax.plot_surface(np.full_like(yy_wall, Lx), yy_wall, zz_wall, color=color, alpha=0.3, edgecolor='black', lw=0.3)
        ax.plot_surface(np.zeros_like(yy_wall), yy_wall, zz_wall, color=color, alpha=0.3, edgecolor='black', lw=0.3)
        
        xx_wall, zz_wall = np.meshgrid([0, Lx], [z_bottom, z_top])
        ax.plot_surface(xx_wall, np.full_like(xx_wall, Ly), zz_wall, color=color, alpha=0.3, edgecolor='black', lw=0.3)
        ax.plot_surface(xx_wall, np.zeros_like(xx_wall), zz_wall, color=color, alpha=0.3, edgecolor='black', lw=0.3)
        
        # Sidebar labels - moved further to the right (1.35)
        y_pos = 0.2 + i * 0.25 
        ax.text2D(1.35, y_pos, 
                  f"Layer {i+1}\nE = {E}\nh = {t:.3f}", 
                  transform=ax.transAxes,
                  color='black', fontsize=12, fontweight='bold',
                  verticalalignment='center',
                  bbox=dict(facecolor=color, alpha=0.3, edgecolor='black', lw=1, pad=5))
        
        z_bottom = z_top

    ax.set_xlabel('Length X (m)')
    ax.set_ylabel('Width Y (m)')
    ax.set_zlabel('Thickness Z (m)')
    
    ax.set_box_aspect((Lx, Ly, 0.4)) 
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_zlim(0, 0.15)
    
    ax.view_init(elev=30, azim=-60)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    visualize_3d_setup([1.0, 5.0, 10.0], [0.0333, 0.0333, 0.0333], "pinn-workflow/visualization/setup_3d_layered.png")
    visualize_3d_setup([10.0, 1.0, 5.0], [0.02, 0.05, 0.03], "pinn-workflow/visualization/setup_3d_unequal.png")
