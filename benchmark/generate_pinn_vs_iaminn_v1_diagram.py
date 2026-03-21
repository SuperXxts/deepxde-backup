import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = "/public/home/xinxi/wxtian/WXTIAN/PINN/deepxde/benchmark/images/model_architecture"
OUT = os.path.join(OUT_DIR, "pinn_vs_iaminn_v1_architecture.png")
os.makedirs(OUT_DIR, exist_ok=True)

BLUE = "#355F94"
GOLD = "#D2A071"
INK = "#13233A"
PALE = "#EEF3F8"
PALE2 = "#F7F3EC"
GREEN = "#3C7A5A"
RED = "#A94F4F"

def box(ax, x, y, w, h, text, fc=PALE, ec=BLUE, fontsize=11, weight="regular"):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                           linewidth=2.0, edgecolor=ec, facecolor=fc)
    ax.add_patch(patch)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize, color=INK, weight=weight)
    return patch

def arrow(ax, x1, y1, x2, y2, color=BLUE, lw=2.0, style='-|>'):
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                          linewidth=lw, color=color, shrinkA=2, shrinkB=2)
    ax.add_patch(arr)
    return arr

fig = plt.figure(figsize=(16, 9), dpi=220)
ax = plt.axes([0,0,1,1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

ax.text(0.25, 0.94, 'PINN Baseline', ha='center', va='center', fontsize=22, color=INK, weight='bold')
ax.text(0.75, 0.94, 'IAMINN-v1', ha='center', va='center', fontsize=22, color=INK, weight='bold')

# Left: PINN
box(ax, 0.07, 0.78, 0.14, 0.08, '(x, y)', fc=PALE2, fontsize=14, weight='bold')
box(ax, 0.07, 0.63, 0.18, 0.09, 'Fourier feature map\n(shared input encoding)', fc=PALE, fontsize=12)
box(ax, 0.06, 0.43, 0.22, 0.12, 'Single MLP backbone\n128-128-128-128', fc='#E8F0FA', fontsize=13, weight='bold')
box(ax, 0.05, 0.22, 0.24, 0.12, 'Direct outputs\nux, uy, sxx, syy, sxy, lambda(x,y), mu(x,y)', fc='#DDEAF8', fontsize=12)
box(ax, 0.04, 0.05, 0.26, 0.10, 'Material field assumption\nfree continuous field regression', fc='#E8F3EC', ec=GREEN, fontsize=12, weight='bold')
arrow(ax, 0.14, 0.78, 0.16, 0.72)
arrow(ax, 0.16, 0.63, 0.17, 0.55)
arrow(ax, 0.17, 0.43, 0.17, 0.34)
arrow(ax, 0.17, 0.22, 0.17, 0.15, color=GREEN)

# Right: IAMINN
box(ax, 0.59, 0.78, 0.14, 0.08, '(x, y)', fc=PALE2, fontsize=14, weight='bold')
box(ax, 0.56, 0.63, 0.20, 0.09, 'Fourier feature map\n(shared input encoding)', fc=PALE, fontsize=12)
box(ax, 0.47, 0.43, 0.18, 0.12, 'State net\n128-128-128-128', fc='#E8F0FA', fontsize=13, weight='bold')
box(ax, 0.72, 0.43, 0.18, 0.12, 'Interface net\n64-64-64', fc='#F4E8DA', ec=GOLD, fontsize=13, weight='bold')
box(ax, 0.45, 0.22, 0.22, 0.12, 'State outputs\nux, uy, sxx, syy, sxy', fc='#DDEAF8', fontsize=12)
box(ax, 0.71, 0.24, 0.20, 0.08, 'Region logits', fc='#FAEFD9', ec=GOLD, fontsize=12)
box(ax, 0.71, 0.13, 0.20, 0.08, 'Softmax over\nbackground + regions', fc='#FAEFD9', ec=GOLD, fontsize=12)
box(ax, 0.73, 0.03, 0.17, 0.07, 'class probabilities', fc='#FAEFD9', ec=GOLD, fontsize=11)
box(ax, 0.49, 0.03, 0.19, 0.10, 'Global region params\nraw_lambda_params\nraw_mu_params', fc='#FBE7E7', ec=RED, fontsize=11, weight='bold')
box(ax, 0.58, 0.18, 0.17, 0.09, 'Weighted mixture\n=> lambda(x,y), mu(x,y)', fc='#E8F3EC', ec=GREEN, fontsize=11, weight='bold')
box(ax, 0.46, 0.84, 0.48, 0.04, 'Structured assumption: material field = region mixture, not free field regression', fc='#F4F7FB', fontsize=11)
arrow(ax, 0.66, 0.78, 0.66, 0.72)
arrow(ax, 0.66, 0.63, 0.56, 0.55)
arrow(ax, 0.66, 0.63, 0.81, 0.55)
arrow(ax, 0.56, 0.43, 0.56, 0.34)
arrow(ax, 0.81, 0.43, 0.81, 0.32, color=GOLD)
arrow(ax, 0.81, 0.24, 0.81, 0.21, color=GOLD)
arrow(ax, 0.81, 0.13, 0.81, 0.10, color=GOLD)
arrow(ax, 0.68, 0.08, 0.72, 0.18, color=RED)
arrow(ax, 0.73, 0.07, 0.67, 0.18, color=GOLD)
arrow(ax, 0.66, 0.18, 0.66, 0.15, color=GREEN)

# Shared loss section
box(ax, 0.23, 0.005, 0.54, 0.06, 'Shared training/evaluation protocol: same PDE residuals + same soft boundary points + same train/val/eval observations + same optimizer budget', fc='#F4F7FB', fontsize=11, weight='bold')

# Side notes
box(ax, 0.02, 0.87, 0.26, 0.05, 'PINN advantage: high flexibility for lambda(x,y), mu(x,y)', fc='#E8F3EC', ec=GREEN, fontsize=10)
box(ax, 0.70, 0.87, 0.26, 0.05, 'IAMINN-v1 prior: encourages region-like material structure', fc='#FBE7E7', ec=RED, fontsize=10)

fig.savefig(OUT, dpi=220, bbox_inches='tight')
print(OUT)
