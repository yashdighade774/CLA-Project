"""Role 2: orthogonality drift meters, custom MGS, repair, and the adaptive trigger.

Which factors must stay orthonormal (Brand, below eq. 11):
    U, Up, U@Up, V@Vp      yes
    Vp, V alone            no  (only Vp must stay the inverse of Vi)
So the meters are
    dU   running ||U'U - I||_F      exact, costs O(pr) per new column (probe in grow())
    dUp  ||Up'Up - I||_F            O(r^3)
    dVi  ||Vi@Vp - I||_F            O(r^3)
and `exact()` gives the real ||(U Up)'(U Up) - I|| and ||(V Vp)'(V Vp) - I|| for experiments.
"""
import numpy as np
from isvd import St, update, small_svd, lsv, rsv

F64 = np.float64


def ur(dt):
    """unit roundoff u = eps / 2 of the working precision"""
    return np.finfo(dt).eps / 2


def dev(M):
    """||M'M - I||_F measured in float64"""
    M = M.astype(F64)
    return float(np.linalg.norm(M.T @ M - np.eye(M.shape[1])))


# ---- meters ----------------------------------------------------------------

def drift(st):
    """cheap meters, no O(p) or O(q) work"""
    dvi = np.linalg.norm(st.Vi.astype(F64) @ st.Vp.astype(F64) - np.eye(st.r))
    return {'U': st.dU, 'Up': dev(st.Up), 'Vi': float(dvi)}


def exact(st):
    """ground truth for the experiments; costs O(p r^2 + q r^2)"""
    return {'U': dev(st.U.astype(F64) @ st.Up.astype(F64)),
            'V': dev(st.V.astype(F64) @ st.Vp.astype(F64))}


# ---- repair ----------------------------------------------------------------

def mgs(A):
    """Modified Gram-Schmidt, A = Q R. Rows of Q are worked on for speed."""
    Q = np.array(A.T, copy=True)
    r = Q.shape[0]
    R = np.zeros((r, r), A.dtype)
    for j in range(r):
        for i in range(j):
            R[i, j] = Q[i] @ Q[j]
            Q[j] -= R[i, j] * Q[i]
        R[j, j] = np.linalg.norm(Q[j])
        Q[j] /= R[j, j]
    return Q.T.copy(), R


def refac(st, M, how='jacobi'):
    """Replace Up by M (nearly orthogonal) without changing the product.

    U M S Vp' = U (Qs Sg Rs') Vp' = U Qs Sg (Vp Rs)'  with  M diag(sv) = Qs Sg Rs'.
    """
    Qs, sg, Rs = small_svd(M * st.sv, how)
    st.Up = Qs.astype(st.dt)
    st.sv = sg.astype(st.dt)
    st.Vp = (st.Vp @ Rs).astype(st.dt)
    st.Vi = np.linalg.inv(st.Vp.astype(F64)).astype(st.dt)     # re-sync Vi with Vp


def fix(st, tall, how='jacobi'):
    """tall=False: the paper's O(r^3) refactor of the small factors.
    tall=True: also re-orthogonalize U with MGS (O(p r^2)), folding R into Up."""
    M = st.Up
    if tall:
        st.U, R = mgs(st.U)
        M = R @ st.Up
        st.dU = dev(st.U)
    refac(st, M, how)


def fixV(st, how='jacobi'):
    """Re-orthogonalize V@Vp (q x r) with MGS, cost O(q r^2).

    V Vp = Q R,  so  X = U Up S R' Q'.  Refactor the small matrix Up S R' = Q2 Sg R2'
    and set  V <- Q,  Vp <- R2,  Vi <- R2'  (R2 is orthogonal).
    """
    Q, R = mgs((st.V @ st.Vp).astype(st.dt))
    Q2, sg, R2 = small_svd((st.Up * st.sv) @ R.T, how)
    st.V = Q
    st.Up = Q2.astype(st.dt)
    st.sv = sg.astype(st.dt)
    st.Vp = R2.astype(st.dt)
    st.Vi = R2.T.astype(st.dt)


# ---- the adaptive trigger --------------------------------------------------

class Trk:
    """Wraps the engine. After every frame the cheap meters are read; when the
    worst one passes  tau = k * r * u  the factors are repaired.

    Why that threshold: one update does r-term inner products, and the standard
    model fl(a o b) = (a o b)(1 + d), |d| <= u, gives an error of about r*u per
    update (gamma_r = r u / (1 - r u)). tau is k updates' worth of that worst case.
    """

    def __init__(self, p, dt=np.float32, rmax=20, k=100, auto=True, how='jacobi', vgap=None):
        self.st = St(p, dt)
        self.rmax, self.k, self.auto, self.how = rmax, k, auto, how
        self.vgap = vgap or rmax      # V@Vp is checked every vgap frames: O(q r^2)/r = O(q r) per frame
        self.log = []          # per frame: dU, dUp, dVi, dV (nan between checks), repair flags
        self.nfix = 0          # flags: 1 small refactor, 2 + MGS on U, 4 MGS on V

    def tau(self):
        return self.k * max(self.st.r, 1) * ur(self.st.dt)

    def step(self, a):
        update(self.st, a, rmax=self.rmax, how=self.how)
        d = drift(self.st)
        code, dv = 0, np.nan
        if self.auto and self.st.r and max(d.values()) > self.tau():
            tall = d['U'] > self.tau()
            fix(self.st, tall, self.how)
            code = 2 if tall else 1
            d = drift(self.st)
        if self.auto and self.st.r and len(self.log) % self.vgap == self.vgap - 1:
            dv = dev(self.st.V.astype(F64) @ self.st.Vp.astype(F64))
            if dv > self.tau():
                fixV(self.st, self.how)
                code |= 4
                d = drift(self.st)
        self.nfix += code > 0
        self.log.append((d['U'], d['Up'], d['Vi'], dv, code))
        return d
