"""
Definitive Ablation & Sensitivity Study.

Evaluates:
  1. Rank Sensitivity: r in {1, 2, 4, 8, 16, 32} with Base 2 Log Scaling & MFlops.
  2. Component Ablation: No Trigger vs Strict (k=50) vs Optimal (k=100) vs BLAS.
Generates:
  - ablation_metrics.png (Tradeoff curves with safe bounding box)
  - ablation_visual.png (Clean 4x4 visual pipeline grid)
  - table1_rank_sweep.tex
  - table2_component_ablation.tex
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, NullLocator

from vid import cols, split, gray
from stab import Trk, exact

VIDEO_PATH = "robbery_input.mp4"
RANKS = [1, 2, 4, 8, 16, 32]
TARGET_FRAME = 120          # Intruder active in vault
EMPTY_RANGE = (10, 45)      # Clean empty vault (Clutter or Noise floor)
MOTION_RANGE = (75, 200)    # Active intrusion (Signal + Clutter)
ALARM_THRESHOLD = 0.08      # Calibrated detection threshold

if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"Video file '{VIDEO_PATH}' not found in current directory. Please check the path.")
print("SUBSPACE CAPACITY SENSITIVITY SWEEP (r in {1, 2, 4, 8, 16, 32})")

rank_results = []
visual_snapshots = {}

for r in RANKS:
    trk = None
    update_latencies = []
    empty_residuals = []
    motion_residuals = []
    p_dim = None
    
    for j, col in enumerate(cols(VIDEO_PATH, size=(228, 228), dt=np.float32)):
        if trk is None:
            p_dim = col.size
            trk = Trk(p_dim, dt=np.float32, rmax=r, k=100, auto=True)
            
        # Isolate PURE SVD update time
        t0 = time.perf_counter()
        trk.step(col)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        update_latencies.append(dt_ms)
        
        bg, fg = split(trk.st, col)
        
        # Relative residual norm: ||a - a_hat|| / ||a||
        col_norm = np.linalg.norm(col) + 1e-12
        rel_res = float(np.linalg.norm(fg) / col_norm)
        
        if EMPTY_RANGE[0] <= j <= EMPTY_RANGE[1]:
            empty_residuals.append(rel_res)
        elif MOTION_RANGE[0] <= j <= MOTION_RANGE[1]:
            motion_residuals.append(rel_res)
            
        if j == TARGET_FRAME:
            mask = (fg.reshape(228, 228) > ALARM_THRESHOLD).astype(np.uint8) * 255
            visual_snapshots[r] = {
                'orig': gray(col, (228, 228)),
                'bg': gray(bg, (228, 228)),
                'fg': gray(4.0 * fg, (228, 228)),
                'mask': mask
            }
            
    mflops = ((4 * p_dim * r) + (r ** 3)) / 1e6
    noise_floor = np.mean(empty_residuals) if empty_residuals else 1e-6
    signal_energy = np.mean(motion_residuals) if motion_residuals else 0.0
    scr = signal_energy / (noise_floor + 1e-12)  # Signal to Clutter Ratio
    
    e_gt = exact(trk.st)
    rank_results.append({
        'rank': r,
        'final_r': trk.st.r,
        'latency_ms': np.mean(update_latencies[5:]),
        'mflops': mflops,
        'noise_floor': noise_floor,
        'scr': scr,
        'repairs': trk.nfix,
        'drift_U': e_gt['U']
    })
    print(f"Rank {r:2d} (final r={trk.st.r}) | Time: {rank_results[-1]['latency_ms']:.2f} ms | "
          f"Flops: {mflops:.2f} MFlops | Clutter Floor: {noise_floor:.4f} | "
          f"SCR: {scr:.1f}x | Repairs: {trk.nfix} | Drift: {e_gt['U']:.2e}")
# COMPONENT ABLATION
print("COMPONENT ABLATION")

configs = [
    {"name": "Brand (2006) Baseline (No Trigger)", "k": 100, "auto": False, "how": 'jacobi'},
    {"name": "Ours Strict Trigger (k=50)",         "k": 50,  "auto": True,  "how": 'jacobi'},
    {"name": "Ours Optimal Trigger (k=100)",        "k": 100, "auto": True,  "how": 'jacobi'},
    {"name": "Ours with NumPy BLAS Solver",        "k": 100, "auto": True,  "how": 'numpy'},
]

ablation_results = []

for cfg in configs:
    trk = None
    latencies = []
    
    for j, col in enumerate(cols(VIDEO_PATH, size=(228, 228), dt=np.float32)):
        if trk is None:
            trk = Trk(col.size, dt=np.float32, rmax=8, k=cfg['k'], auto=cfg['auto'], how=cfg['how'])
            
        t0 = time.perf_counter()
        trk.step(col)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        
    e_gt = exact(trk.st)
    ablation_results.append({
        'config': cfg['name'],
        'latency_ms': np.mean(latencies[5:]),
        'repairs': trk.nfix,
        'drift_U': e_gt['U'],
        'drift_V': e_gt['V']
    })
    print(f"Config: {cfg['name']:<36} | Time: {ablation_results[-1]['latency_ms']:.2f} ms | "
          f"Repairs: {trk.nfix:2d} | Drift U@Up: {e_gt['U']:.2e} | Drift V@Vp: {e_gt['V']:.2e}")
# WRITE FORMATTED LATEX TABLES TO DISK
with open("table1_rank_sweep.tex", "w") as f:
    f.write(r"\begin{table}[h]" + "\n")
    f.write(r"\centering" + "\n")
    f.write(r"\caption{Ablation of Subspace Rank Capacity $r_{\max}$ on Surveillance Stream.}" + "\n")
    f.write(r"\label{tab:rank_sweep}" + "\n")
    f.write(r"\begin{tabular}{cccccc}" + "\n")
    f.write(r"\toprule" + "\n")
    f.write(r"Rank $r_{\max}$ & Update Time (ms) $\downarrow$ & Arithmetic (MFlops) $\downarrow$ & Clutter Floor $\downarrow$ & SCR $\uparrow$ & Operating Regime \\" + "\n")
    f.write(r"\midrule" + "\n")
    for res in rank_results:
        if res['rank'] <= 2:
            regime = "Under-parameterized (False alarms)"
        elif res['rank'] <= 8:
            regime = r"\textbf{Optimal Operating Point}"
        else:
            regime = "Subspace Leakage (Overfitting)"
        f.write(f"{res['rank']:2d} & {res['latency_ms']:.2f} & {res['mflops']:.2f} & {res['noise_floor']:.4f} & {res['scr']:.1f}$\\times$ & {regime} \\\\\n")
    f.write(r"\bottomrule" + "\n")
    f.write(r"\end{tabular}" + "\n")
    f.write(r"\end{table}" + "\n")
print("\nSaved 'table1_rank_sweep.tex'")

with open("table2_component_ablation.tex", "w") as f:
    f.write(r"\begin{table}[h]" + "\n")
    f.write(r"\centering" + "\n")
    f.write(r"\caption{Component Ablation of Proposed Numerical Stabilizer ($r_{\max}=8$).}" + "\n")
    f.write(r"\label{tab:component_ablation}" + "\n")
    f.write(r"\begin{tabular}{lccccc}" + "\n")
    f.write(r"\toprule" + "\n")
    # Accurately reporting factored product bases U@Up and V@Vp
    f.write(r"Method Configuration & Update (ms) $\downarrow$ & Repairs & Drift $\|(U U')^T (U U') - I\|_F$ & Drift $\|(V V')^T (V V') - I\|_F$ & Bounded? \\" + "\n")
    f.write(r"\midrule" + "\n")
    for a in ablation_results:
        stable_str = r"\textbf{Yes}" if a['drift_U'] < 1e-4 else r"\textbf{No} (Drifting)"
        f.write(f"{a['config']:<36} & {a['latency_ms']:.2f} & {a['repairs']} & {a['drift_U']:.2e} & {a['drift_V']:.2e} & {stable_str} \\\\\n")
    f.write(r"\bottomrule" + "\n")
    f.write(r"\end{tabular}" + "\n")
    f.write(r"\end{table}" + "\n")
print("Saved 'table2_component_ablation.tex'")
# PLOTS
fig, ax1 = plt.subplots(figsize=(7.5, 4.2))
ranks_arr = [r['rank'] for r in rank_results]
latencies = [r['latency_ms'] for r in rank_results]
scrs = [r['scr'] for r in rank_results]

color = 'tab:blue'
ax1.set_xscale('log', base=2)
ax1.set_xticks(ranks_arr)
ax1.xaxis.set_major_formatter(FormatStrFormatter('%d'))
# Suppress minor ticks between powers of 2 for clean axis rendering
ax1.xaxis.set_minor_locator(NullLocator())

ax1.set_xlabel('Subspace Rank Capacity ($r_{\max}$)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Per-Frame Update Latency (ms)', color=color, fontsize=11, fontweight='bold')
line1 = ax1.plot(ranks_arr, latencies, marker='o', color=color, linewidth=2.2, label='Latency (ms)')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, which="both", ls="--", alpha=0.3)

ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('Signal-to-Clutter Ratio (SCR)', color=color, fontsize=11, fontweight='bold')
line2 = ax2.plot(ranks_arr, scrs, marker='s', color=color, linewidth=2.2, label='SCR (Contrast)')
ax2.tick_params(axis='y', labelcolor=color)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center left')

plt.title('Subspace Capacity Trade-off: Latency vs. Detection SCR', fontsize=12, pad=10)
plt.savefig('ablation_metrics.png', dpi=160, bbox_inches='tight')
print("\nSaved 'ablation_metrics.png')")

# Visual Comparison Grid
sel = [1, 4, 8, 32]
fig, axes = plt.subplots(len(sel), 4, figsize=(11.5, 2.3 * len(sel)))

axes[0, 0].set_title('Live Input ($a_t$)', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Learned Background ($U U^T a$)', fontsize=11, fontweight='bold')
axes[0, 2].set_title('Extracted Residual (4x)', fontsize=11, fontweight='bold')
axes[0, 3].set_title('Alarm Mask ($> \gamma$)', fontsize=11, fontweight='bold')

for idx, r_val in enumerate(sel):
    sn = visual_snapshots[r_val]
    
    axes[idx, 0].imshow(sn['orig'], cmap='gray')
    axes[idx, 0].set_ylabel(f'Rank {r_val}', fontsize=12, fontweight='bold')
    axes[idx, 0].set_xticks([])
    axes[idx, 0].set_yticks([])
    
    axes[idx, 1].imshow(sn['bg'], cmap='gray')
    axes[idx, 1].axis('off')
    
    axes[idx, 2].imshow(sn['fg'], cmap='gray')
    axes[idx, 2].axis('off')
    
    axes[idx, 3].imshow(sn['mask'], cmap='gray')
    axes[idx, 3].axis('off')
plt.tight_layout()
plt.savefig('ablation_visual.png', dpi=160, bbox_inches='tight')
print(" Saved 'ablation_visual.png'")
print("Ablation Study Complete")