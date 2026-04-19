import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- ISOMETRIC MATH ENGINE ---
def iso(x, y, z, scale=4.0, z_exag=20.0):
    """
    Mathematically projects 3D coordinates (x, y, z) into a 2D isometric plane.
    z_exag artificially inflates the Z-axis so thin plates are visible.
    """
    ang = np.radians(30)
    u = (x - y) * np.cos(ang) * scale
    v = -(x + y) * np.sin(ang) * scale + z * z_exag
    return u, v

def draw_iso_block(ax, z_bottom, z_top, fc, ec):
    """Draws a perfect isometric 3D block using ordered 2D polygons."""
    # Define faces
    pts_left = [iso(0,1,z_bottom), iso(1,1,z_bottom), iso(1,1,z_top), iso(0,1,z_top)]
    pts_right = [iso(1,0,z_bottom), iso(1,1,z_bottom), iso(1,1,z_top), iso(1,0,z_top)]
    pts_top = [iso(0,0,z_top), iso(1,0,z_top), iso(1,1,z_top), iso(0,1,z_top)]

    # Draw in back-to-front order (Painter's Algorithm)
    ax.add_patch(plt.Polygon(pts_left, facecolor=fc, edgecolor=ec, alpha=0.9, lw=1.2))
    ax.add_patch(plt.Polygon(pts_right, facecolor=fc, edgecolor=ec, alpha=0.8, lw=1.2))
    ax.add_patch(plt.Polygon(pts_top, facecolor=fc, edgecolor=ec, alpha=1.0, lw=1.2))

def draw_load_patch(ax, z_top):
    """Draws the yellow load patch directly on the top face."""
    pts = [iso(1/3, 1/3, z_top), iso(2/3, 1/3, z_top), iso(2/3, 2/3, z_top), iso(1/3, 2/3, z_top)]
    ax.add_patch(plt.Polygon(pts, facecolor='#F5B041', edgecolor='#D4A03E', lw=1.5, zorder=5))

def draw_axes(ax):
    """Draws coordinate axes perfectly aligned to the isometric grid with visible z-axis."""
    # Origin shifted further left to avoid crowding
    ox, oy, oz = -0.4, 1.3, 0 
    o_2d = iso(ox, oy, oz)
    x_2d = iso(ox+0.35, oy, oz)
    y_2d = iso(ox, oy-0.35, oz)
    z_2d = iso(ox, oy, oz+0.08) # Increased z magnitude so it renders clearly
    
    arrow_props = dict(arrowstyle="->", color="black", lw=1.5)
    ax.annotate("", xy=x_2d, xytext=o_2d, arrowprops=arrow_props)
    ax.annotate("", xy=y_2d, xytext=o_2d, arrowprops=arrow_props)
    ax.annotate("", xy=z_2d, xytext=o_2d, arrowprops=arrow_props)
    
    ax.text(x_2d[0]+0.3, x_2d[1]-0.1, "x", fontweight="bold", fontsize=12)
    ax.text(y_2d[0]-0.4, y_2d[1], "y", fontweight="bold", fontsize=12)
    ax.text(z_2d[0], z_2d[1]+0.3, "z", fontweight="bold", fontsize=12)

def format_canvas(ax, title):
    """Sets a wide canvas to allow long arrow pull-outs."""
    ax.set_axis_off()
    ax.set_xlim(-6.5, 8.5) # Widened bounds for longer arrows
    ax.set_ylim(-6.0, 7.5)
    ax.text(1.0, 6.5, title, fontsize=15, fontweight='bold', ha='center')

# --- PLOTTING ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
c1, c2, c3 = '#A8C8DC', '#6B9DBF', '#3D7299'
ec1, ec2, ec3 = '#5A8CAF', '#3D6F8F', '#2A4F6E'

# ============================================================
# PANEL A: One-Layer
# ============================================================
format_canvas(ax1, r"(a) One-layer configuration ($N_L = 1$)")
draw_axes(ax1)

H = 0.1
draw_iso_block(ax1, 0, H, c1, ec1)
draw_load_patch(ax1, H)

# Dimensions
mid_x = iso(0.5, 1, 0)
ax1.text(mid_x[0]-1.0, mid_x[1]-0.5, r'$L_x = 1.0$', fontsize=12)
mid_y = iso(1, 0.5, 0)
ax1.text(mid_y[0]+0.4, mid_y[1]-0.5, r'$L_y = 1.0$', fontsize=12)

# Load Callout (Pulled higher up)
lp_center = iso(0.5, 0.5, H)
ax1.annotate(r'Load: $p_0 = 1.0$' + '\n' + r'$x, y \in [L_x/3, 2L_x/3]$',
             xy=lp_center, xytext=(lp_center[0], lp_center[1] + 3.5),
             arrowprops=dict(arrowstyle="->", color='#B9770E', lw=1.5),
             color='#B9770E', fontweight='bold', ha='center', fontsize=11)

# Layer Callout (Pulled further right)
pt_layer = iso(1, 0.5, H/2)
ax1.annotate(r'$E_1$, $t_1 = H = 0.1$',
             xy=pt_layer, xytext=(pt_layer[0] + 2.5, pt_layer[1]),
             arrowprops=dict(arrowstyle="-", color='#1A3C5A', lw=1.2),
             color='#1A3C5A', fontweight='bold', va='center', fontsize=12)

# Clamped Callout (Pulled further left)
pt_clamp = iso(0, 0.5, H/2)
ax1.annotate("Clamped\n" + r"$\mathbf{u} = \mathbf{0}$",
             xy=pt_clamp, xytext=(pt_clamp[0] - 2.5, pt_clamp[1] + 1.5),
             arrowprops=dict(arrowstyle="->", color='#1B4F72', lw=1.2),
             color='#1B4F72', fontweight='bold', ha='right', va='center', fontsize=12)

# ============================================================
# PANEL B: Three-Layer
# ============================================================
format_canvas(ax2, r"(b) Three-layer configuration ($N_L = 3$)")
draw_axes(ax2)

t1, t2, t3 = 0.05, 0.03, 0.02
draw_iso_block(ax2, 0, t1, c1, ec1)
draw_iso_block(ax2, t1, t1+t2, c2, ec2)
draw_iso_block(ax2, t1+t2, H, c3, ec3)
draw_load_patch(ax2, H)

# Dimensions
ax2.text(mid_x[0]-1.0, mid_x[1]-0.5, r'$L_x = 1.0$', fontsize=12)

# Load Callout (Pulled higher up)
ax2.annotate(r'Load: $p_0 = 1.0$' + '\n' + r'$x, y \in [L_x/3, 2L_x/3]$',
             xy=lp_center, xytext=(lp_center[0], lp_center[1] + 3.5),
             arrowprops=dict(arrowstyle="->", color='#B9770E', lw=1.5),
             color='#B9770E', fontweight='bold', ha='center', fontsize=11)

# Layer Callouts (Staggered and pulled far right to prevent block overlap)
pt1 = iso(1, 0.5, t1/2)
ax2.annotate(r'Layer 1: $E_1 = 10.0$' + '\n' + r'$t_1 = 0.05$ (steel)',
             xy=pt1, xytext=(pt1[0] + 2.5, pt1[1] - 1.8),
             arrowprops=dict(arrowstyle="->", color='#1A3C5A', lw=1.2),
             color='#1A3C5A', fontweight='bold', va='center', fontsize=11)

pt2 = iso(1, 0.5, t1 + t2/2)
ax2.annotate(r'Layer 2: $E_2 = 1.0$' + '\n' + r'$t_2 = 0.03$ (polymer)',
             xy=pt2, xytext=(pt2[0] + 3.2, pt2[1]),
             arrowprops=dict(arrowstyle="->", color='#1A3C5A', lw=1.2),
             color='#1A3C5A', fontweight='bold', va='center', fontsize=11)

pt3 = iso(1, 0.5, t1 + t2 + t3/2)
ax2.annotate(r'Layer 3: $E_3 = 5.0$' + '\n' + r'$t_3 = 0.02$ (aluminum)',
             xy=pt3, xytext=(pt3[0] + 2.5, pt3[1] + 1.8),
             arrowprops=dict(arrowstyle="->", color='#1A3C5A', lw=1.2),
             color='#1A3C5A', fontweight='bold', va='center', fontsize=11)

# Clamped Callout (Pulled further left)
ax2.annotate("Clamped\n" + r"$\mathbf{u} = \mathbf{0}$",
             xy=pt_clamp, xytext=(pt_clamp[0] - 2.5, pt_clamp[1] + 1.5),
             arrowprops=dict(arrowstyle="->", color='#1B4F72', lw=1.2),
             color='#1B4F72', fontweight='bold', ha='right', va='center', fontsize=12)

# Shared Parameter Bar
fig.text(0.5, 0.05,
         r'$\mathbf{Material\ Parameters:}$ $E_i$: Young modulus $\bullet$ '
         r'$\nu = 0.3$ (Poisson ratio)    |    '
         r'$\mathbf{Parameter\ Ranges:}$ $E_i \in [1.0, 10.0]$ $\bullet$ '
         r'$t_i \in [0.02, 0.10]$',
         fontsize=13, ha='center', va='bottom',
         bbox=dict(boxstyle='round,pad=0.6', facecolor='#F4F6F7', edgecolor='#BDC3C7', lw=1.5))

plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.15, wspace=0.1)
plt.savefig('fig_problem_setup_pro_v2.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()