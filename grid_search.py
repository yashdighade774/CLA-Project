"""
Definitive 2D Grid Search for

Evaluates 2D Pareto Frontier:
  - Subspace Ranks: r in {2, 4, 8, 16, 32}
  - Thresholds:     k in {25, 50, 100, 250, 500, inf}

Generates:
  - grid_search_3panel.png (Drift vs. Repairs vs. Latency)
  - grid_search_data.npz   (Raw numerical matrices archived to disk)
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from vid import cols
from stab import Trk, exact

VIDEO_PATH = "robbery_input.mp4"
RANKS = [2, 4, 8, 16, 32]
# Includes inf to show the unmonitored Brand baseline
K_VALUES = [25, 50, 100, 250, 500, np.inf]
K_LABELS = ['25', '50', '100', '250', '500', r'$\infty$ (Off)']

if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"Video file '{VIDEO_PATH}' not found in current directory. Please check the path.")
print(f"RUNNING DEFINITIVE 2D GRID SEARCH: {len(RANKS)} Ranks x {len(K_VALUES)} Thresholds = {len(RANKS)*len(K_VALUES)} configurations")

drift_grid = np.zeros((len(RANKS), len(K_VALUES)))
repairs_grid = np.zeros((len(RANKS), len(K_VALUES)), dtype=int)
latency_grid = np.zeros((len(RANKS), len(K_VALUES)))

t_start = time.time()

for i, r in enumerate(RANKS):
    for j, k_val in enumerate(K_VALUES):
        trk = None
        times = []
        is_auto = not np.isinf(k_val)
        actual_k = 100 if np.isinf(k_val) else k_val
        
        for frame_idx, col in enumerate(cols(VIDEO_PATH, size=(228, 228), dt=np.float32)):
            if trk is None:
                trk = Trk(col.size, dt=np.float32, rmax=r, k=actual_k, auto=is_auto)
                
            t0 = time.perf_counter()
            trk.step(col)
            times.append((time.perf_counter() - t0) * 1000.0)
            
        e_gt = exact(trk.st)
        drift_grid[i, j] = e_gt['U']
        repairs_grid[i, j] = trk.nfix
        latency_grid[i, j] = np.mean(times[5:])
        
        lbl = K_LABELS[j]
        print(f"Config (r={r:2d}, k={lbl:>10}) -> Drift: {e_gt['U']:.2e} | Repairs: {trk.nfix:2d} | Latency: {latency_grid[i, j]:.2f} ms")

# Save raw matrices to disk
np.savez("grid_search_data.npz", 
         ranks=RANKS, k_values=K_VALUES, k_labels=K_LABELS,
         drift=drift_grid, repairs=repairs_grid, latency=latency_grid)
print(f"\n Grid Search Complete in {time.time() - t_start:.1f}s. Data saved to 'grid_search_data.npz'.")
# GENERATE 3-PANEL HEATMAP
fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

# Log10 Basis Drift
log_drift = np.log10(np.clip(drift_grid, 1e-8, 1.0))
im1 = axes[0].imshow(log_drift, cmap='magma', aspect='auto')
axes[0].set_title(r'Log$_{10}$ Drift $\|(U U\prime)^T (U U\prime) - I\|_F$', fontsize=11, fontweight='bold', pad=10)
axes[0].set_xticks(range(len(K_VALUES)))
axes[0].set_xticklabels(K_LABELS, fontsize=9.5)
axes[0].set_yticks(range(len(RANKS)))
axes[0].set_yticklabels(RANKS, fontsize=10)
axes[0].set_xlabel('Repair Multiplier ($k$)', fontsize=10, fontweight='bold')
axes[0].set_ylabel('Subspace Rank ($r_{\max}$)', fontsize=10, fontweight='bold')

min_val, max_val = log_drift.min(), log_drift.max()
midpoint = (min_val + max_val) / 2.0
for i in range(len(RANKS)):
    for j in range(len(K_VALUES)):
        val = drift_grid[i, j]
        txt_color = 'white' if log_drift[i, j] < midpoint else 'black'
        axes[0].text(j, i, f"{val:.1e}", ha='center', va='center', color=txt_color, fontsize=8, fontweight='bold')

cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
cbar1.set_label(r'Log$_{10}$ Drift', fontsize=10)

# Number of MGS Repairs
im2 = axes[1].imshow(repairs_grid, cmap='viridis', aspect='auto')
axes[1].set_title('Repair Invocations ($N_{\mathrm{fix}}$)', fontsize=11, fontweight='bold', pad=10)
axes[1].set_xticks(range(len(K_VALUES)))
axes[1].set_xticklabels(K_LABELS, fontsize=9.5)
# Remove redundant y tick labels
axes[1].set_yticks(range(len(RANKS)))
axes[1].set_yticklabels([])
axes[1].set_xlabel('Repair Multiplier ($k$)', fontsize=10, fontweight='bold')

mid_rep = (repairs_grid.max() + repairs_grid.min()) / 2.0
for i in range(len(RANKS)):
    for j in range(len(K_VALUES)):
        rep = repairs_grid[i, j]
        txt_color = 'white' if rep < mid_rep else 'black'
        axes[1].text(j, i, f"{rep}", ha='center', va='center', color=txt_color, fontsize=9.5, fontweight='bold')

cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
cbar2.set_label('Total MGS Repairs', fontsize=10)

# Per Frame Update Latency
im3 = axes[2].imshow(latency_grid, cmap='plasma', aspect='auto')
axes[2].set_title('Per-Frame Update Latency (ms)', fontsize=11, fontweight='bold', pad=10)
axes[2].set_xticks(range(len(K_VALUES)))
axes[2].set_xticklabels(K_LABELS, fontsize=9.5)
# Remove redundant y tick labels
axes[2].set_yticks(range(len(RANKS)))
axes[2].set_yticklabels([])
axes[2].set_xlabel('Repair Multiplier ($k$)', fontsize=10, fontweight='bold')

mid_lat = (latency_grid.max() + latency_grid.min()) / 2.0
for i in range(len(RANKS)):
    for j in range(len(K_VALUES)):
        lat = latency_grid[i, j]
        txt_color = 'white' if lat < mid_lat else 'black'
        axes[2].text(j, i, f"{lat:.2f}", ha='center', va='center', color=txt_color, fontsize=9, fontweight='bold')

cbar3 = fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
cbar3.set_label('Compute Latency (ms)', fontsize=10)

plt.suptitle(r'2D Hyperparameter Grid Search: Subspace Rank ($r_{\max}$) vs. Trigger Threshold ($k$)', 
             fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('grid_search_3panel.png', dpi=160, bbox_inches='tight')
print("Saved 'grid_search_3panel.png'")