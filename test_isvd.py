import numpy as np
from isvd import St, update, jsvd, full, lsv, rsv
from stab import Trk, fix, dev, exact, drift

rng = np.random.default_rng(0)


def lowrank(p, q, r, noise=0.0):
    X = rng.standard_normal((p, r)) @ rng.standard_normal((r, q))
    return X + noise * rng.standard_normal((p, q))


def run(X, dt=np.float64, **kw):
    st = St(X.shape[0], dt)
    for j in range(X.shape[1]):
        update(st, X[:, j], **kw)
    return st


def test_jacobi():
    for shape in [(12, 12), (13, 12), (1, 1), (5, 3)]:
        A = rng.standard_normal(shape)
        u, s, v = jsvd(A)
        assert np.allclose(u @ np.diag(s) @ v.T, A, atol=1e-12)
        assert np.allclose(s, np.linalg.svd(A, compute_uv=False), atol=1e-12)
        assert np.allclose(u.T @ u, np.eye(shape[1]), atol=1e-12)
    print('jacobi ok')


def test_exact_lowrank():
    X = lowrank(200, 40, 5)
    for how in ('jacobi', 'numpy'):
        st = run(X, how=how, rmax=20)
        sv = np.linalg.svd(X, compute_uv=False)[:5]
        assert st.r == 5, st.r
        assert np.allclose(st.sv, sv, rtol=1e-9), (st.sv, sv)
        assert np.allclose(full(st), X, atol=1e-8 * np.linalg.norm(X))
        assert dev(lsv(st)) < 1e-12 and dev(rsv(st)) < 1e-12
        assert st.q == 40
    print('exact low rank ok (rank 5 found, sv match, both solvers)')


def test_truncation():
    # Greedy truncation can miss a signal direction if the first few (noisy) frames
    # are badly conditioned (paper, sec. 6). The paper's remedy: keep headroom.
    X = lowrank(300, 60, 4, noise=0.05)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    best = np.linalg.norm(X - (U[:, :4] * s[:4]) @ Vt[:4])
    tight = np.linalg.norm(X - full(run(X, rmax=4)))
    roomy = np.linalg.norm(X - full(run(X, rmax=12)))
    print('rank-4 optimum %.2f | rmax=4: %.2f | rmax=12: %.2f' % (best, tight, roomy))
    st = run(X, rmax=12)
    u = st.U.astype(np.float64)
    assert abs(roomy - np.linalg.norm(X - u @ (u.T @ X))) < 1e-8      # = projection onto span(U)
    assert roomy < 1.3 * best and roomy < tight / 3
    assert np.allclose(st.sv[:4], s[:4], rtol=0.03)


def test_zero_frames():
    X = lowrank(50, 10, 3)
    X[:, 0] = 0
    X[:, 4] = 0
    st = run(X)
    assert np.allclose(full(st), X, atol=1e-9)
    print('zero frames ok')


def test_dU_running_matches_exact():
    X = lowrank(400, 120, 8, noise=0.01)
    st = run(X, np.float32, rmax=8)
    ex = dev(st.U)
    print('running dU %.3e   exact %.3e' % (st.dU, ex))
    assert abs(st.dU - ex) <= 1e-6 * max(ex, 1e-12) + 1e-12


def test_fix_keeps_matrix():
    X = lowrank(300, 80, 6, noise=0.01)
    st = run(X, np.float32, rmax=6)
    before = full(st).astype(np.float64)
    for tall in (False, True):
        fix(st, tall)
        after = full(st).astype(np.float64)
        rel = np.linalg.norm(after - before) / np.linalg.norm(before)
        print('fix(tall=%s): change %.2e, exact drift %s' % (tall, rel, exact(st)))
        assert rel < 1e-4
    assert exact(st)['U'] < 5e-6


def test_fixV():
    from stab import fixV
    X = lowrank(300, 90, 6, noise=0.01)
    st = run(X, np.float32, rmax=6)
    before = full(st).astype(np.float64)
    e0 = exact(st)
    fixV(st)
    e1 = exact(st)
    rel = np.linalg.norm(full(st) - before) / np.linalg.norm(before)
    print('fixV: V drift %.2e -> %.2e, matrix change %.1e, ||Vi Vp - I|| %.1e'
          % (e0['V'], e1['V'], rel, drift(st)['Vi']))
    assert e1['V'] < 5e-6 and rel < 1e-4 and e1['U'] < 5e-6


def test_trigger():
    X = lowrank(500, 600, 6, noise=0.01)
    X32 = X.astype(np.float32)
    a, b = Trk(500, np.float32, rmax=6, k=100, auto=False), Trk(500, np.float32, rmax=6, k=1, auto=True)
    for j in range(600):
        a.step(X32[:, j])
        b.step(X32[:, j])
    ea, eb = exact(a.st), exact(b.st)
    print('no trigger  :', ea, ' | trigger k=1:', eb, 'repairs', b.nfix)
    assert b.nfix > 0
    assert eb['U'] <= ea['U'] * 1.01 + 1e-7
    assert eb['V'] <= ea['V'] * 1.01 + 1e-7
    c = Trk(500, np.float32, rmax=6, k=100, auto=True)      # the default threshold
    for j in range(600):
        c.step(X32[:, j])
    ec = exact(c.st)
    print('default k=100: ', ec, 'repairs', c.nfix, 'tau %.1e' % c.tau())
    assert ec['U'] < 2 * c.tau() and ec['V'] < 2 * c.tau()


if __name__ == '__main__':
    test_jacobi()
    test_exact_lowrank()
    test_truncation()
    test_zero_frames()
    test_dU_running_matches_exact()
    test_fix_keeps_matrix()
    test_fixV()
    test_trigger()
    print('all passed')
