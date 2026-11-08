"""Experiments for the stability section: drift vs frame, with and without the trigger.

Synthetic stream: rank-r signal with slowly wandering coefficients plus small noise,
p = 1024 pixels, run in float32 so that roundoff is visible.
"""
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stab import Trk, exact, ur
from isvd import full

p, n, r = 1024, 3000, 8
rng = np.random.default_rng(3)
B = np.linalg.qr(rng.standard_normal((p, r)))[0] * np.arange(r, 0, -1)
c = np.cumsum(0.05 * rng.standard_normal((r, n)), axis=1) + rng.standard_normal((r, 1))
X = (B @ c + 0.01 * rng.standard_normal((p, n))).astype(np.float32)
import sys
if 'ordered' not in sys.argv:       # a random order = stationary stream (see paper, sec. 6)
    X = X[:, rng.permutation(n)]
print('stream order:', 'as generated (drifting)' if 'ordered' in sys.argv else 'shuffled (stationary)')


def run(label, **kw):
    t = Trk(p, np.float32, rmax=r + 2, **kw)
    ts, eu, ev = [], [], []
    t0 = time.time()
    for j in range(n):
        t.step(X[:, j])
        if j % 100 == 99:
            e = exact(t.st)
            ts.append(j + 1)
            eu.append(e['U'])
            ev.append(e['V'])
    el = time.time() - t0
    print('%-22s exact drift U %.2e  V %.2e | repairs %4d | %.1fs'
          % (label, eu[-1], ev[-1], t.nfix, el))
    return t, ts, eu, ev


runs = {'no trigger': run('no trigger', auto=False),
        'trigger k=100': run('trigger k=100', k=100),
        'trigger k=1000': run('trigger k=1000', k=1000)}

# accuracy against the batch SVD of the same data
U, s, Vt = np.linalg.svd(X.astype(np.float64), full_matrices=False)
k_ = r + 2
best = np.linalg.norm(X - (U[:, :k_] * s[:k_]) @ Vt[:k_])
print('\nbatch rank-%d error %.4f' % (k_, best))
for name, (t, *_ ) in runs.items():
    err = np.linalg.norm(X - full(t.st).astype(np.float32))
    rel = np.abs(t.st.sv[:3] - s[:3]) / s[:3]
    print('%-16s rank-%d reconstruction error %.4f   top-3 sv rel. err %s'
          % (name, k_, err, ' '.join('%.1e' % x for x in rel)))

fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for name, (t, ts, eu, ev) in runs.items():
    ax[0].semilogy(ts, eu, label=name)
    ax[1].semilogy(ts, ev, label=name)
u = ur(np.float32)
for a_, tt in zip(ax, ('left basis  ||(U Up)\'(U Up) - I||_F', 'right basis  ||(V Vp)\'(V Vp) - I||_F')):
    a_.axhline(u, color='gray', ls=':', label='unit roundoff u')
    a_.set_title(tt, fontsize=10)
    a_.set_xlabel('frame')
    a_.grid(alpha=0.3)
ax[0].legend()
plt.tight_layout()
plt.savefig('drift.png', dpi=130)
print('saved drift.png')
