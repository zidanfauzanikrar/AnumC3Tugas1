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

    if n > 0 and U[-1, -1] == 0.0:
        raise np.linalg.LinAlgError(f"B singular: pivot nol di kolom {n - 1}")

    return L, U, P

def solve_dense_lu(L, U, P, b):
    n = len(b)
    bp = b[P]

    y = np.zeros(n, dtype=U.dtype)
    for i in range(n):
        y[i] = bp[i] - L[i, :i] @ y[:i]

    z = np.zeros(n, dtype=U.dtype)
    for i in range(n - 1, -1, -1):
        if U[i, i] == 0.0:
            raise np.linalg.LinAlgError(f"B singular: pivot nol di kolom {i}")
        z[i] = (y[i] - U[i, i + 1:] @ z[i + 1:]) / U[i, i]

    return z

# 2. solver banded (thomas digeneralisasi) dengan partial pivoting


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
        if AB[i, p] == 0.0:
            raise np.linalg.LinAlgError(f"B singular: pivot nol di kolom {i}")
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

def solve_banded_thomas(AB_in, p, q, b_in):
    """Solver pita dengan pivoting dan ruang fill-in."""
    if not isinstance(p, (int, np.integer)) or not isinstance(q, (int, np.integer)):
        raise ValueError('Bandwidth harus bilangan bulat.')
    if p < 0 or q < 0:
        raise ValueError('Bandwidth harus nonnegatif.')
    AB = np.array(AB_in, dtype=float, copy=True)
    b = np.array(b_in, dtype=float, copy=True)
    if AB.ndim != 2 or AB.shape[1] != 2*p+q+1:
        raise ValueError('Ukuran penyimpanan band salah.')
    n = len(AB)
    if n == 0 or b.shape != (n,) or not np.all(np.isfinite(AB)) or not np.all(np.isfinite(b)):
        raise ValueError('Masukan harus finite dan dimensi sesuai.')
    for k in range(n-1):
        last_row = min(k+p, n-1)
        last_col = min(k+p+q, n-1)
        m = last_col-k+1
        best_row, best_val = k, abs(AB[k,p])
        for i in range(k+1,last_row+1):
            val = abs(AB[i,p-(i-k)])
            if val > best_val:
                best_row,best_val = i,val
        if best_row != k:
            s = best_row-k
            seg_k = AB[k,p:p+m].copy()
            AB[k,p:p+m] = AB[best_row,p-s:p-s+m]
            AB[best_row,p-s:p-s+m] = seg_k
            b[k],b[best_row] = b[best_row],b[k]
        pivot = AB[k,p]
        if pivot == 0:
            raise ValueError(f'Matriks singular: pivot nol di kolom {k}.')
        row_k = AB[k,p:p+m]
        for i in range(k+1,last_row+1):
            d_ik = p-(i-k)
            aik = AB[i,d_ik]
            if aik == 0:
                continue
            factor = aik/pivot
            AB[i,d_ik:d_ik+m] -= factor*row_k
            AB[i,d_ik] = factor
            b[i] -= factor*b[k]
    x = np.zeros(n)
    for i in range(n-1,-1,-1):
        if AB[i,p] == 0:
            if i == n-1:
                raise ValueError('Matriks singular: pivot terakhir nol.')
            raise ValueError(f'Matriks singular: pivot nol di kolom {i}.')
        j_hi = min(i+p+q,n-1)
        length = j_hi-i
        x[i] = (b[i]-AB[i,p+1:p+1+length] @ x[i+1:j_hi+1])/AB[i,p]
    return x


def build_band_storage(B, p, q):
    """Konversi dense ke band; menolak bandwidth yang memotong koefisien."""
    B = np.asarray(B, dtype=float)
    n = len(B)
    rows,cols = np.nonzero(B)
    if np.any(rows-cols > p) or np.any(cols-rows > q):
        raise ValueError('Bandwidth terlalu kecil; koefisien akan terpotong.')
    AB = np.zeros((n,2*p+q+1))
    for off in range(-p,q+1):
        i = np.arange(max(0,-off),min(n,n-off))
        AB[i,off+p] = B[i,i+off]
    return AB


def condition_inf(B):
    """Kondisi norma infinity melalui invers dari LU buatan kelompok."""
    Binv = inverse_via_lu(B)
    return float(np.max(np.sum(np.abs(B), axis=1)) *
                 np.max(np.sum(np.abs(Binv), axis=1)))


def lu_factor(B):
    """LU dense partial pivoting. B[perm, :] = L @ U."""
    lu = np.array(B, dtype=float, copy=True)
    if lu.ndim != 2 or lu.shape[0] != lu.shape[1]:
        raise ValueError('B harus persegi.')
    n = len(lu)
    perm = np.arange(n)
    for k in range(n):
        pivot = k + int(np.argmax(np.abs(lu[k:, k])))
        if lu[pivot, k] == 0:
            raise ValueError('Matriks singular: pivot nol.')
        if pivot != k:
            lu[[k, pivot], :] = lu[[pivot, k], :]
            perm[[k, pivot]] = perm[[pivot, k]]
        if k + 1 < n:
            lu[k+1:, k] /= lu[k, k]
            # Eliminasi Gauss; outer product hanya operasi perkalian.
            lu[k+1:, k+1:] -= np.outer(lu[k+1:, k], lu[k, k+1:])
    return lu, perm


def lu_solve(lu, perm, b):
    """Forward/back substitution; mendukung satu atau beberapa ruas kanan."""
    rhs = np.asarray(b, dtype=float)
    vector = rhs.ndim == 1
    if vector:
        rhs = rhs[:, None]
    if rhs.ndim != 2 or rhs.shape[0] != len(lu):
        raise ValueError('Dimensi ruas kanan tidak cocok.')
    x = rhs[perm, :].copy()  # P b, bukan b asli.
    n = len(lu)
    for i in range(n):
        x[i] -= lu[i, :i] @ x[:i]
    for i in range(n-1, -1, -1):
        if lu[i, i] == 0:
            raise ValueError('Matriks singular: pivot nol.')
        x[i] = (x[i] - lu[i, i+1:] @ x[i+1:]) / lu[i, i]
    return x[:, 0] if vector else x


def dense_solve(B, b):
    lu, perm = lu_factor(B)
    return lu_solve(lu, perm, b)


def thomas_pivot(dl, d, du, b):
    """Tridiagonal + partial pivoting; tidak membentuk matriks N x N."""
    dl, d, du, rhs = [np.array(a, dtype=float, copy=True)
                       for a in (dl, d, du, b)]
    n = len(d)
    if n < 1 or len(dl) != n-1 or len(du) != n-1 or rhs.shape != (n,):
        raise ValueError('Ukuran vektor tridiagonal tidak cocok.')
    du2 = np.zeros(max(n-2, 0))
    for i in range(n-1):
        if abs(d[i]) >= abs(dl[i]):
            if d[i] == 0:
                raise ValueError('Matriks singular: pivot nol.')
            multiplier = dl[i] / d[i]
            d[i+1] -= multiplier * du[i]
            rhs[i+1] -= multiplier * rhs[i]
        else:
            multiplier = d[i] / dl[i]
            d[i] = dl[i]
            next_diagonal = d[i+1]
            d[i+1] = du[i] - multiplier * next_diagonal
            du[i] = next_diagonal
            if i < n-2:
                du2[i] = du[i+1]
                du[i+1] = -multiplier * du[i+1]
            old_rhs = rhs[i]
            rhs[i] = rhs[i+1]
            rhs[i+1] = old_rhs - multiplier * rhs[i+1]
    if d[-1] == 0:
        raise ValueError('Matriks singular: pivot terakhir nol.')
    # Tulis hasil langsung ke rhs untuk menghemat satu vektor.
    rhs[-1] /= d[-1]
    if n >= 2:
        rhs[-2] = (rhs[-2] - du[-1]*rhs[-1]) / d[-2]
    for i in range(n-3, -1, -1):
        rhs[i] = (rhs[i] - du[i]*rhs[i+1] - du2[i]*rhs[i+2]) / d[i]
    return rhs
