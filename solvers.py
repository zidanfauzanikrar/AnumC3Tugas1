import numpy as np

# 0. bangun A = I - T^T,  B (baris pertama diganti), dan b
def build_system(T):
    N = T.shape[0]
    A = np.eye(N) - T.T
    B = A.copy()
    B[0, :] = 0.0
    B[0, 0] = 1.0
    b = np.zeros(N)
    b[0] = 1.0
    return A, B, b

def compute_bandwidth(B, tol=1e-12):
    rows, cols = np.nonzero(np.abs(B) > tol)
    diff = rows - cols
    p = int(diff.max()) if len(diff) > 0 and diff.max() > 0 else 0
    q = int((-diff).max()) if len(diff) > 0 and (-diff).max() > 0 else 0
    return p, q

# 1. solver dense: faktorisasi LU dense dengan partial pivoting (PB = LU)
def lu_dense_partial_pivot(B, dtype=float):
    n = B.shape[0]
    U = np.array(B, dtype=dtype, copy=True)  # satu kali salin aja
    L = np.eye(n, dtype=dtype)
    P = np.arange(n)

    for k in range(n - 1):
        pivot_row = k + int(np.argmax(np.abs(U[k:, k])))
        if pivot_row != k:
            U[[k, pivot_row], :] = U[[pivot_row, k], :]
            P[[k, pivot_row]] = P[[pivot_row, k]]
            if k > 0:
                L[[k, pivot_row], :k] = L[[pivot_row, k], :k]

        pivot_val = U[k, k]
        if pivot_val == 0.0:
            raise np.linalg.LinAlgError(f"B singular: pivot nol di kolom {k}")

        factors = U[k + 1:, k] / pivot_val
        U[k + 1:, k:] -= np.outer(factors, U[k, k:])
        L[k + 1:, k] = factors

    return L, U, P

def solve_dense_lu(L, U, P, b):
    n = len(b)
    bp = b[P]

    y = np.zeros(n, dtype=U.dtype)
    for i in range(n):
        y[i] = bp[i] - L[i, :i] @ y[:i]

    z = np.zeros(n, dtype=U.dtype)
    for i in range(n - 1, -1, -1):
        z[i] = (y[i] - U[i, i + 1:] @ z[i + 1:]) / U[i, i]

    return z

# 2. solver banded (thomas digeneralisasi) dengan partial pivoting
def build_band_storage(B, p, q):
    n = B.shape[0]
    W = 2 * p + q + 1
    AB = np.zeros((n, W))
    for d in range(W):
        j_minus_i = d - p  # j = i + (d-p)
        i_lo = max(0, -j_minus_i)
        i_hi = min(n, n - j_minus_i)
        if i_hi > i_lo:
            idx_i = np.arange(i_lo, i_hi)
            idx_j = idx_i + j_minus_i
            AB[idx_i, d] = B[idx_i, idx_j]
    return AB

def solve_banded_thomas(AB_in, p, q, b_in):
    AB = AB_in.copy()
    b = np.array(b_in, dtype=float, copy=True)
    n = AB.shape[0]

    for k in range(n - 1):
        last_row = min(k + p, n - 1)
        last_col = min(k + p + q, n - 1)
        m = last_col - k + 1 # panjang segmen kolom k..last_col

        # cari pivot di kolom k, baris k..last_row  (elemen (i,k) ada di d = p-(i-k))
        best_row, best_val = k, abs(AB[k, p])
        for i in range(k + 1, last_row + 1):
            val = abs(AB[i, p - (i - k)])
            if val > best_val:
                best_row, best_val = i, val

        # tukar segmen kolom k..last_col antara baris k & best_row
        if best_row != k:
            s = best_row - k
            seg_k = AB[k, p:p + m].copy()
            AB[k, p:p + m] = AB[best_row, p - s:p - s + m]
            AB[best_row, p - s:p - s + m] = seg_k
            b[k], b[best_row] = b[best_row], b[k]

        pivot_val = AB[k, p]
        if pivot_val == 0.0:
            raise np.linalg.LinAlgError(f"B singular: pivot nol di kolom {k}")

        # eliminasi baris k+1..last_row pakai baris k (forward elimination)
        row_k = AB[k, p:p + m]
        for i in range(k + 1, last_row + 1):
            d_ik = p - (i - k)
            aik = AB[i, d_ik]
            if aik == 0.0:
                continue
            factor = aik / pivot_val
            AB[i, d_ik:d_ik + m] -= factor * row_k
            AB[i, d_ik] = factor # simpan multiplier (L)
            b[i] -= factor * b[k]

    # backward substitution (U punya lebar pita atas p+q setelah pivoting)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        j_hi = min(i + p + q, n - 1)
        L = j_hi - i
        s = b[i] - AB[i, p + 1:p + 1 + L] @ x[i + 1:j_hi + 1]
        x[i] = s / AB[i, p]

    return x

def solve_banded_thomas_loop(AB_in, p, q, b_in):
    AB = AB_in.copy()
    b = b_in.astype(float).copy()
    n = AB.shape[0]

    for k in range(n - 1):
        last_row = min(k + p, n - 1)

        # cari pivot
        best_row, best_val = k, abs(AB[k, p])
        for i in range(k + 1, last_row + 1):
            d_ik = k - i + p
            val = abs(AB[i, d_ik])
            if val > best_val:
                best_row, best_val = i, val

        # tuker (kolom k..k+p+q) antara baris k & best_row
        if best_row != k:
            last_col = min(k + p + q, n - 1)
            for j in range(k, last_col + 1):
                dk = j - k + p
                db = j - best_row + p
                AB[k, dk], AB[best_row, db] = AB[best_row, db], AB[k, dk]
            b[k], b[best_row] = b[best_row], b[k]

        pivot_val = AB[k, p]
        if pivot_val == 0.0:
            raise np.linalg.LinAlgError(f"B singular: pivot nol di kolom {k}")

        # eliminasi baris k+1..last_row pake baris k (f.e)
        last_col = min(k + p + q, n - 1)
        for i in range(k + 1, last_row + 1):
            d_ik = k - i + p
            aik = AB[i, d_ik]
            if aik == 0.0:
                continue
            factor = aik / pivot_val
            for j in range(k, last_col + 1):
                d_ij = j - i + p
                d_kj = j - k + p
                AB[i, d_ij] -= factor * AB[k, d_kj]
            AB[i, d_ik] = factor
            b[i] -= factor * b[k]

    # b.s
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = b[i]
        j_hi = min(i + p + q, n - 1)
        for j in range(i + 1, j_hi + 1):
            d_ij = j - i + p
            s -= AB[i, d_ij] * x[j]
        x[i] = s / AB[i, p]

    return x

# 2b. bangun penyimpanan pita LANGSUNG dari T (tanpa membentuk B dense N x N)
def compute_bandwidth_from_T(T, tol=1e-12):
    r, c = np.nonzero(np.abs(T) > tol) # T[r,c] -> B[c,r]
    keep = c >= 1 # baris 0 dari B diganti e1
    diff = c[keep] - r[keep] # (i - j) untuk entri B[c,r]
    p = int(max(diff.max(), 0)) if diff.size else 0
    q = int(max((-diff).max(), 0)) if diff.size else 0
    return p, q

def build_band_from_T(T, p, q):
    n = T.shape[0]
    AB = np.zeros((n, 2 * p + q + 1))
    for off in range(-p, q + 1): # off = j - i
        i = np.arange(max(0, -off), min(n, n - off))
        j = i + off
        AB[i, off + p] = (off == 0) - T[j, i] # B[i,j] = delta_ij - T[j,i]
    AB[0, :] = 0.0
    AB[0, p] = 1.0
    return AB

# 3. utilitas
def recover_T_from_B(B):
    N = B.shape[0]
    TT = np.eye(N) - B
    TT[0, :] = 1.0 - TT[1:, :].sum(axis=0)
    return TT.T

def theory_flops(N, p, q):
    dense = (2 / 3) * N**3 + 2 * N**2 # LU + substitusi maju/mundur
    banded = 2 * N * p * (p + q + 1) + 2 * N * (p + q) + N # eliminasi + b.s + pembagian
    return dense, banded

def inverse_via_lu(B):
    n = B.shape[0]
    L, U, P = lu_dense_partial_pivot(B)
    Y = np.eye(n)[P]
    for i in range(n):
        Y[i] -= L[i, :i] @ Y[:i]
    X = np.zeros((n, n))
    for i in range(n - 1, -1, -1):
        X[i] = (Y[i] - U[i, i + 1:] @ X[i + 1:]) / U[i, i]
    return X
