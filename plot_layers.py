import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def iso(x, y, z, scale=4.0, z_exag=20.0):
    ang = np.radians(30)
    u = (x - y) * np.cos(ang) * scale
    v = -(x + y) * np.sin(ang) * scale + z * z_exag
    return u, v

def draw_iso_block(ax, z_bottom, z_top, fc, ec):
    pts_left = [iso(0,1,z_bottom), iso(1,1,z_bottom), iso(1,1,z_top), iso(0,1,z_top)]
    pts_right = [iso(1,0,z_bottom), iso(1,1,z_bottom), iso(1,1,z_top), iso(1,0,z_top)]
    pts_top = [iso(0,0,z_top), iso(1,0,z_top), iso(1,1,z_top), iso(0,1,z_top)]
    ax.add_patch(plt.Polygon(pts_left, facecolor=fc, edgecolor=ec, alpha=0.9, lw=0.8))
    ax.add_patch(plt.Polygon(pts_right, facecolor=fc, edgecolor=ec, alpha=0.8, lw=0.8))
    ax.add_patch(plt.Polygon(pts_top, facecolor=fc, edgecolor=ec, alpha=1.0, lw=0.8))

def draw_load_patch(ax, z_top):
    pts = [iso(1/3, 1/3, z_top), iso(2/3, 1/3, z_top), iso(2/3, 2/3, z_top), iso(1/3, 2/3, z_top)]
    ax.add_patch(plt.Polygon(pts, facecolor='#F5B041', edgecolor='#D4A03E', lw=1.0, zorder=5))

def draw_axes(ax):
    ox, oy, oz = -0.4, 1.3, 0 
    o_2d, x_2d, y_2d, z_2d = iso(ox, oy, oz), iso(ox+0.35, oy, oz), iso(ox, oy-0.35, oz), iso(ox, oy, oz+0.08)
    props = dict(arrowstyle="->", color="black", lw=0.8)
    ax.annotate("", xy=x_2d, xytext=o_2d, arrowprops=props)
    ax.annotate("", xy=y_2d, xytext=o_2d, arrowprops=props)
    ax.annotate("", xy=z_2d, xytext=o_2d, arrowprops=props)
    ax.text(x_2d[0], x_2d[1]-0.1, "x", fontweight="bold", fontsize=8)
    ax.text(y_2d[0]-0.5, y_2d[1]+0.3, "y", fontweight="bold", fontsize=8)
    ax.text(z_2d[0]-0.1, z_2d[1], "z", fontweight="bold", fontsize=8)

def format_canvas(ax, title):
    ax.set_axis_off()
    ax.set_xlim(-6.5, 8.5)
    ax.set_ylim(-6.0, 7.5)
    ax.text(1.0, 6.5, title, fontsize=10, fontweight='bold', ha='center')

# Apply Times New Roman to regular text and a LaTeX-style font (STIX) to math
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.5))
c1, c2, c3 = '#A8C8DC', '#6B9DBF', '#3D7299'
ec1, ec2, ec3 = '#5A8CAF', '#3D6F8F', '#2A4F6E'

# PANEL A
format_canvas(ax1, r"(a) One-layer configuration ($N_L = 1$)")
draw_axes(ax1)
H = 0.1
draw_iso_block(ax1, 0, H, c1, ec1)
draw_load_patch(ax1, H)

mid_x, mid_y = iso(0.5, 1, 0), iso(1, 0.5, 0)
ax1.text(mid_x[0]-1.5, mid_x[1]-0.5, r"$L_x = 1.0$", fontsize=8)
ax1.text(mid_y[0]+0.4, mid_y[1]-0.5, r"$L_y = 1.0$", fontsize=8)

lp_center = iso(0.5, 0.5, H)
ax1.annotate(r"Load: $p_0 = 1.0$" + "\n" + r"$x, y \in [L_x/3, 2L_x/3]$", xy=lp_center, xytext=(lp_center[0], lp_center[1] + 3.5),
             arrowprops=dict(arrowstyle="->", color='#B9770E', lw=1.0), color='#B9770E', fontweight='bold', ha='center', fontsize=7.5)

pt_layer, pt_clamp = iso(1, 0.5, H/2), iso(0, 0.5, H/2)
ax1.annotate(r"$E_1$, $t_1 = H = 0.1$", xy=pt_layer, xytext=(pt_layer[0] + 2.5, pt_layer[1]),
             arrowprops=dict(arrowstyle="-", color='#1A3C5A', lw=0.8), color='#1A3C5A', fontweight='bold', va='center', fontsize=8)
ax1.annotate("Clamped\n" + r"$\mathbf{u} = \mathbf{0}$", xy=pt_clamp, xytext=(pt_clamp[0] - 2.5, pt_clamp[1] + 1.5),
             arrowprops=dict(arrowstyle="->", color='#1B4F72', lw=0.8), color='#1B4F72', fontweight='bold', ha='right', va='center', fontsize=8)

# PANEL B
format_canvas(ax2, r"(b) Three-layer configuration ($N_L = 3$)")
draw_axes(ax2)
t1, t2, t3 = 0.05, 0.03, 0.02
draw_iso_block(ax2, 0, t1, c1, ec1)
draw_iso_block(ax2, t1, t1+t2, c2, ec2)
draw_iso_block(ax2, t1+t2, H, c3, ec3)
draw_load_patch(ax2, H)

ax2.text(mid_x[0]-1.5, mid_x[1]-0.5, r"$L_x = 1.0$", fontsize=8)
ax2.annotate(r"Load: $p_0 = 1.0$" + "\n" + r"$x, y \in [L_x/3, 2L_x/3]$", xy=lp_center, xytext=(lp_center[0], lp_center[1] + 3.5),
             arrowprops=dict(arrowstyle="->", color='#B9770E', lw=1.0), color='#B9770E', fontweight='bold', ha='center', fontsize=7.5)

pt1, pt2, pt3 = iso(1, 0.5, t1/2), iso(1, 0.5, t1 + t2/2), iso(1, 0.5, t1 + t2 + t3/2)
ax2.annotate(r"Layer 1: $E_1 = 10.0$" + "\n" + r"$t_1 = 0.05$ (steel-like)", xy=pt1, xytext=(pt1[0] + 2.5, pt1[1] - 1.8),
             arrowprops=dict(arrowstyle="->", color='#1A3C5A', lw=0.8), color='#1A3C5A', fontweight='bold', va='center', fontsize=7.5)
ax2.annotate(r"Layer 2: $E_2 = 1.0$" + "\n" + r"$t_2 = 0.03$ (polymer-like)", xy=pt2, xytext=(pt2[0] + 3.2, pt2[1]),
             arrowprops=dict(arrowstyle="->", color='#1A3C5A', lw=0.8), color='#1A3C5A', fontweight='bold', va='center', fontsize=7.5)
ax2.annotate(r"Layer 3: $E_3 = 5.0$" + "\n" + r"$t_3 = 0.02$ (aluminum-like)", xy=pt3, xytext=(pt3[0] + 2.5, pt3[1] + 1.8),
             arrowprops=dict(arrowstyle="->", color='#1A3C5A', lw=0.8), color='#1A3C5A', fontweight='bold', va='center', fontsize=7.5)
ax2.annotate("Clamped\n" + r"$\mathbf{u} = \mathbf{0}$", xy=pt_clamp, xytext=(pt_clamp[0] - 2.5, pt_clamp[1] + 1.5),
             arrowprops=dict(arrowstyle="->", color='#1B4F72', lw=0.8), color='#1B4F72', fontweight='bold', ha='right', va='center', fontsize=8)

# Constructing the LaTeX-integrated legend string
legend_text = (
    r"$\mathbf{Material\ Parameters:}$ $E_i$: Young modulus $\bullet$ $\nu = 0.3$ (Poisson ratio)    $|$    "
    r"$\mathbf{Parameter\ Ranges:}$ $E_i \in [1.0, 10.0]$ $\bullet$ $t_i \in [0.02, 0.10]$"
)

fig.text(0.5, 0.05, legend_text,
         fontsize=8, ha='center', va='bottom', bbox=dict(boxstyle='round,pad=0.6', facecolor='#F4F6F7', edgecolor='#BDC3C7', lw=0.8))

plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.18, wspace=0.1)
plt.savefig('fig_problem_setup.pdf', format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()