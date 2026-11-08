"""Incremental thin SVD after Brand (2006), "Fast low-rank modifications of the thin SVD".

State is kept in the five-factor form of eq. 11:

    X  =  U @ Up @ diag(sv) @ Vp.T @ V.T

U (p x r) and V (q x r) are tall and only ever grow (a column / a row at a time).
Up, sv, Vp are small (r x r) and absorb every rotation.
Vi is the inverse of Vp (the paper's V'^+), kept so V can get a new row cheaply.
"""
import numpy as np


class St:
    def __init__(self, p, dt=np.float64):
        self.p, self.dt = p, dt
        self.U = np.zeros((p, 0), dt)
        self.Up = np.zeros((0, 0), dt)
        self.sv = np.zeros(0, dt)
        self.Vp = np.zeros((0, 0), dt)
        self.Vi = np.zeros((0, 0), dt)
        self.V = np.zeros((0, 0), dt)
        self.dU = 0.0          # running ||U'U - I||_F, filled in by grow() (role 2)

    @property
    def r(self):
        return len(self.sv)

    @property
    def q(self):
        return self.V.shape[0]


def lsv(st):
    return st.U @ st.Up


def rsv(st):
    return st.V @ st.Vp


def full(st):
    return (lsv(st) * st.sv) @ rsv(st).T


# ---- small helpers ---------------------------------------------------------

def pad1(M):
    """[[M, 0], [0, 1]] for any shape (n x k -> n+1 x k+1)."""
    n, k = M.shape
    out = np.zeros((n + 1, k + 1), M.dtype)
    out[:n, :k] = M
    out[n, k] = 1
    return out


def rot(M, i, j, c, s):
    """Givens rotation applied to columns i, j of M, in place."""
    x = M[:, i].copy()
    M[:, i] = c * x - s * M[:, j]
    M[:, j] = s * x + c * M[:, j]


def jsvd(A, sweeps=40):
    """Thin SVD of A (m x n, m >= n) with plane rotations only (one-sided Jacobi).

    Each rotation makes two columns of A orthogonal. Sweep until nothing moves.
    Returns Uj, sig, Vj with A = Uj @ diag(sig) @ Vj.T, sig descending.
    """
    A0 = A
    A = A.copy()
    n = A.shape[1]
    V = np.eye(n, dtype=A.dtype)
    ep = np.finfo(A.dtype).eps
    for _ in range(sweeps):
        moved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                x, y = A[:, i], A[:, j]
                al, be, ga = x @ x, y @ y, x @ y
                if abs(ga) <= 2 * ep * np.sqrt(al * be):
                    continue
                moved = True
                z = (be - al) / (2 * ga)
                t = (1.0 if z >= 0 else -1.0) / (abs(z) + np.sqrt(1 + z * z))
                c = 1 / np.sqrt(1 + t * t)
                rot(A, i, j, c, c * t)
                rot(V, i, j, c, c * t)
        if not moved:
            break
    sig = np.linalg.norm(A, axis=0)
    o = np.argsort(-sig)
    sig = sig[o]
    if sig.size and sig[-1] <= 1e-6 * ep * sig[0]:       # a zero column: U is not determined
        u, sg, vt = np.linalg.svd(A0, full_matrices=False)
        return u, sg, vt.T
    return A[:, o] / sig, sig, V[:, o]


def small_svd(A, how='jacobi'):
    if how == 'numpy':
        u, sg, vt = np.linalg.svd(A, full_matrices=False)
        return u, sg, vt.T
    return jsvd(A)


def wpinv(W, w):
    """Inverse of the square block W of [W; w] with orthonormal columns (eq. 17)."""
    d = 1 - w @ w
    if d < 1e-3:                       # W nearly singular: the formula would blow up
        return np.linalg.pinv(W)
    return W.T + np.outer(w, w @ W.T) / d


# ---- the update ------------------------------------------------------------

def proj(st, a, twice=True):
    """Split a into the part inside span(U) (coords m, w.r.t. U@Up) and the rest p."""
    g = st.U.T @ a
    p = a - st.U @ g
    if twice and st.r and np.linalg.norm(p) < 0.7071 * np.linalg.norm(a):
        c = st.U.T @ p                  # cancellation: one more pass ("twice is enough")
        p = p - st.U @ c
        g = g + c
    return st.Up.T @ g, p


def grow(st, m, p, rho, how):
    """Rank-increasing update: K is (r+1) x (r+1), the broken arrow."""
    r = st.r
    K = np.zeros((r + 1, r + 1), st.dt)
    K[:r, :r] = np.diag(st.sv)
    K[:r, r] = m
    K[r, r] = rho
    C, sg, D = small_svd(K, how)
    P = p / rho
    e = st.U.astype(np.float64).T @ P.astype(np.float64)       # role 2: running drift of U
    P64 = P.astype(np.float64)
    st.dU = float(np.sqrt(st.dU ** 2 + 2 * e @ e + (P64 @ P64 - 1) ** 2))
    st.Up = pad1(st.Up) @ C
    st.U = np.hstack([st.U, P[:, None]])
    st.Vp = pad1(st.Vp) @ D
    st.Vi = D.T @ pad1(st.Vi)
    st.V = pad1(st.V)
    st.sv = sg.astype(st.dt)


def stay(st, m, how):
    """Same-rank update: the new column is projected onto span(U); K is r x (r+1)."""
    r = st.r
    if r == 0:                          # nothing seen yet except zero frames
        st.V = np.vstack([st.V, np.zeros((1, 0), st.dt)])
        return
    K = np.hstack([np.diag(st.sv), m[:, None]])
    D, sg, C = small_svd(K.T, how)      # K = C diag(sg) D.T
    W, w = D[:r], D[r]
    st.Up = st.Up @ C
    st.Vp = st.Vp @ W
    st.Vi = wpinv(W, w) @ st.Vi
    st.V = np.vstack([st.V, w @ st.Vi])
    st.sv = sg.astype(st.dt)


def update(st, a, rmax=20, how='jacobi', decay=1.0, twice=True):
    """Append column a. The rank grows until rmax, after that the part of a outside
    span(U) is dropped (section 6 of the paper: set ||p|| -> 0)."""
    a = np.asarray(a, st.dt).reshape(-1)
    st.sv = st.sv * st.dt(decay)
    m, p = proj(st, a, twice)
    rho = np.linalg.norm(p)
    if rho > 50 * np.finfo(st.dt).eps * np.linalg.norm(a) and st.r < rmax:
        grow(st, m, p, rho, how)
    else:
        stay(st, m, how)
    return st
