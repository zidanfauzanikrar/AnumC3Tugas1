"""Solver manual untuk least squares; tanpa solver/faktorisasi library."""
import numpy as np


def validate(A, b):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or b.shape != (A.shape[0],) or A.shape[0] < A.shape[1]:
        raise ValueError('Perlu A berukuran m x n, m >= n, dan b berukuran m.')
    if not np.isfinite(A).all() or not np.isfinite(b).all():
        raise ValueError('Data mengandung NaN atau infinity.')
    return A, b


def reflector(a):
    scale = float(np.max(np.abs(a)))
    if scale == 0:
        return np.zeros_like(a)
    v = a / scale
    alpha = -np.copysign(np.sqrt(v @ v), v[0])
    v[0] -= alpha
    return v / np.sqrt(v @ v)


def back_substitution(R, c, tol):
    n = len(c)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if abs(R[i, i]) <= tol:
            raise ValueError('Matriks rank deficient atau hampir singular pada toleransi kerja.')
        x[i] = (c[i] - R[i, i + 1:] @ x[i + 1:]) / R[i, i]
    return x


def householder_qr(A, b, explicit_q=False):
    A, b = validate(A, b)
    m, n = A.shape
    R, c = A.copy(), b.copy()
    Q = np.eye(m) if explicit_q else None
    tol = np.finfo(float).eps * max(m, n) * np.max(np.abs(A))
    for k in range(n):
        v = reflector(R[k:, k])
        R[k:, k:] -= 2 * np.outer(v, v @ R[k:, k:])
        c[k:] -= 2 * v * (v @ c[k:])
        if Q is not None:
            Q[:, k:] -= 2 * np.outer(Q[:, k:] @ v, v)
        R[k + 1:, k] = 0
    x = back_substitution(R[:n, :n], c[:n], tol)
    return x, R, c, Q


def first_reflection(A):
    A = np.asarray(A, dtype=float)
    v = reflector(A[:, 0])
    H = np.eye(len(A)) - 2 * np.outer(v, v)
    HA = H @ A
    return v, H, HA


def normal_equations(A, b):
    A, b = validate(A, b)
    G, c = A.T @ A, A.T @ b
    n = len(c)
    L = np.zeros((n, n))
    tol = np.finfo(float).eps * n * np.max(np.abs(G))
    for i in range(n):
        for j in range(i + 1):
            s = G[i, j] - L[i, :j] @ L[j, :j]
            if i == j:
                if s <= tol:
                    raise ValueError('Cholesky gagal: Gram tidak positif definit pada toleransi kerja.')
                L[i, j] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    y = np.zeros(n)
    for i in range(n):
        y[i] = (c[i] - L[i, :i] @ y[:i]) / L[i, i]
    return back_substitution(L.T, y, 0.0)


def condition_2(M, max_sweeps=200):
    """Estimasi singular values lewat one-sided Jacobi manual."""
    C = np.array(M, dtype=float, copy=True)
    scale = np.max(np.abs(C))
    if scale == 0:
        return float('inf')
    C /= scale
    eps = np.finfo(float).eps
    for _ in range(max_sweeps):
        changed = False
        for p in range(C.shape[1] - 1):
            for q in range(p + 1, C.shape[1]):
                a, b = C[:, p].copy(), C[:, q].copy()
                aa, bb, ab = a @ a, b @ b, a @ b
                if aa == 0 or bb == 0 or abs(ab) <= 10 * eps * np.sqrt(aa * bb):
                    continue
                tau = (bb - aa) / (2 * ab)
                t = np.copysign(1.0, tau) / (abs(tau) + np.hypot(1.0, tau))
                cs = 1 / np.sqrt(1 + t * t)
                sn = cs * t
                C[:, p], C[:, q] = cs * a - sn * b, sn * a + cs * b
                changed = True
        if not changed:
            s = np.sqrt(np.sum(C * C, axis=0))
            return float(s.max() / s.min()) if s.min() > 0 else float('inf')
    raise ValueError('Jacobi belum konvergen; naikkan max_sweeps.')
