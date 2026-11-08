"""OpenCV pipeline: video -> one flattened column per frame -> tracked thin SVD.

Each frame is gray, resized to `size` and flattened (p = h*w entries). The tracker
sees one column at a time. The low-rank part is the background; the residual is
the moving foreground.

    python vid.py demo.avi --make-demo          # write a synthetic clip first
    python vid.py demo.avi --out result.avi     # frame | background | foreground
"""
import argparse
import numpy as np
import cv2
from stab import Trk


def cols(src, size=(228, 228), dt=np.float32, every=1, maxn=None):
    """Yield each frame as a flat column in [0, 1]."""
    cap = cv2.VideoCapture(src)
    n = k = 0
    while maxn is None or n < maxn:
        ok, img = cap.read()
        if not ok:
            break
        k += 1
        if (k - 1) % every:
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, size, interpolation=cv2.INTER_AREA)
        yield g.astype(dt).reshape(-1) / dt(255)
        n += 1
    cap.release()


def split(st, a):
    """background = projection of the frame on span(U); U@Up spans the same space,
    so U (U'a) is enough and costs O(pr)."""
    bg = st.U @ (st.U.T @ a)
    return bg, np.abs(a - bg)


def gray(v, shape):
    return np.clip(v.reshape(shape) * 255, 0, 255).astype(np.uint8)


def make_demo(path, n=240, size=(228, 228), seed=1):
    """Static textured scene, slow brightness drift, a bright square moving around."""
    rng = np.random.default_rng(seed)
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    base = 90 + 60 * np.sin(xx / 23.0) * np.cos(yy / 31.0) + 0.15 * xx
    base += cv2.GaussianBlur(rng.normal(0, 25, (h, w)).astype(np.float32), (0, 0), 6)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 25, size, False)
    for t in range(n):
        f = base * (1 + 0.08 * np.sin(t / 40.0)) + rng.normal(0, 2, (h, w))
        cx = int(w / 2 + 70 * np.cos(t / 18.0))
        cy = int(h / 2 + 60 * np.sin(t / 25.0))
        cv2.rectangle(f, (cx - 14, cy - 14), (cx + 14, cy + 14), 240, -1)
        vw.write(np.clip(f, 0, 255).astype(np.uint8))
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('--out', default='result.avi')
    ap.add_argument('--make-demo', action='store_true')
    ap.add_argument('--rmax', type=int, default=8)
    ap.add_argument('--k', type=float, default=100, help='trigger threshold = k*r*u')
    ap.add_argument('--f64', action='store_true')
    ap.add_argument('--no-trigger', action='store_true')
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--maxn', type=int, default=None)
    a = ap.parse_args()
    if a.make_demo:
        make_demo(a.src)
    size = (228, 228)
    dt = np.float64 if a.f64 else np.float32
    shape = (size[1], size[0])
    trk = vw = None
    for j, col in enumerate(cols(a.src, size, dt, a.every, a.maxn)):
        if trk is None:
            trk = Trk(col.size, dt, a.rmax, a.k, not a.no_trigger)
            vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*'MJPG'), 25,
                                 (3 * size[0], size[1]), False)
        d = trk.step(col)
        bg, fg = split(trk.st, col)
        vw.write(np.hstack([gray(col, shape), gray(bg, shape), gray(4 * fg, shape)]))
        if j % 40 == 0:
            print('frame %4d  rank %2d  dU %.1e  dUp %.1e  dVi %.1e  repairs %d'
                  % (j, trk.st.r, d['U'], d['Up'], d['Vi'], trk.nfix))
    vw.release()
    print('frames', j + 1, 'rank', trk.st.r, 'repairs', trk.nfix, '->', a.out)


if __name__ == '__main__':
    main()
