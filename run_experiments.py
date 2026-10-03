import os
import time
import tracemalloc
import numpy as np
import pandas as pd

from solvers import (
    compute_bandwidth, build_system, recover_T_from_B, theory_flops,
    lu_dense_partial_pivot, solve_dense_lu, inverse_via_lu,
    compute_bandwidth_from_T, build_band_from_T, build_band_storage,
    solve_banded_thomas, condition_inf,
)

NPZ_PATH = "output_B_b/B_b_all.npz"
N_LIST = [16, 32, 64, 128, 256, 512]
REPEAT = 7
NORM_ORD = 1

os.makedirs("output", exist_ok=True)
os.makedirs("plots", exist_ok=True)

def time_it(fn, repeat=REPEAT):
    fn()# warm-up (cache, alokasi awal)
    times = []
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), result

def peak_memory_bytes(fn):
    tracemalloc.start()
    tracemalloc.clear_traces()
    result = fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak, result

def condition_number(B, ord_=NORM_ORD):
    Binv = inverse_via_lu(B)
    kappa = np.linalg.norm(B, ord=ord_) * np.linalg.norm(Binv, ord=ord_)
    return float(kappa)

def normalized_backward_error(B, x, b):
    residual = B @ x - b
    denom = np.linalg.norm(B, ord=np.inf) * np.linalg.norm(x, ord=np.inf) + np.linalg.norm(b, ord=np.inf)
    if denom == 0.0:
        return float(np.linalg.norm(residual, ord=np.inf))
    return float(np.linalg.norm(residual, ord=np.inf) / denom)

def load_T(data, N, B):
    if f"T_{N}" in data.files:
        return data[f"T_{N}"], "npz"
    csv = f"data/T_{N}.csv"
    if os.path.exists(csv):
        return np.loadtxt(csv, delimiter=","), "csv"
    return recover_T_from_B(B), "rekonstruksi"

def validate_input(N, B, b, T, T_src):
    assert B.shape == (N, N) and b.shape == (N,), f"shape salah untuk N={N}"
    assert T.shape == (N, N) and np.all(np.isfinite(T)) and np.all(T >= 0), f"T tidak valid untuk N={N}"
    e1 = np.zeros(N); e1[0] = 1.0
    assert np.allclose(B[0], e1), (
        f"N={N}: baris 0 B bukan [1,0,...,0]. Jika Role 1 memakai baris "
        "normalisasi [1,1,...,1], pita rusak (q = N-1) -> solver banded tidak bermakna.")
    assert np.allclose(b, e1), f"N={N}: b bukan e1"
    assert np.allclose(T.sum(axis=1), 1.0, atol=1e-10, rtol=0.0), f"N={N}: T bukan stokastik baris"
    if T_src != "rekonstruksi":
        _, B_chk, _ = build_system(T)
        assert np.allclose(B, B_chk, atol=1e-12), f"N={N}: B tidak konsisten dengan T"

# uji pada ukuran kecil (dijalankan otomatis sebelum eksperimen)
def run_small_tests():
    print("#" * 70)
    print("UJI PADA UKURAN KECIL")
    print("#" * 70)
    results = []
    np.set_printoptions(precision=6, suppress=True)

    def check(name, cond):
        results.append(bool(cond))
        print(f"  [{'LULUS' if cond else 'GAGAL'}] {name}")

    # Uji 1: rantai 4 halte yang bisa dihitung tangan
    print("Uji 1: rantai 4 halte (solusi analitik)")
    T = np.array([[0.6, 0.4, 0.0, 0.0],
                  [0.2, 0.5, 0.3, 0.0],
                  [0.0, 0.3, 0.5, 0.2],
                  [0.0, 0.0, 0.4, 0.6]])
    ratio = [T[i, i + 1] / T[i + 1, i] for i in range(3)]
    pi_exact = np.cumprod([1.0] + ratio); pi_exact /= pi_exact.sum()
    A, B, b = build_system(T)
    p, q = compute_bandwidth(B)
    L, U, P = lu_dense_partial_pivot(B)
    z_d = solve_dense_lu(L, U, P, b)
    z_b = solve_banded_thomas(build_band_from_T(T, p, q), p, q, b)
    print("  pi analitik :", pi_exact)
    print("  pi dense    :", z_d / z_d.sum())
    print("  pi banded   :", z_b / z_b.sum())
    check("dense = analitik", np.allclose(z_d / z_d.sum(), pi_exact, atol=1e-14))
    check("banded = analitik", np.allclose(z_b / z_b.sum(), pi_exact, atol=1e-14))
    check("PB = LU", np.allclose(B[P], L @ U, atol=1e-14))

    # Uji 2: data asli N = 16
    print("\nUji 2: data T_16.csv")
    T = np.loadtxt("data/T_16.csv", delimiter=",")
    A, B, b = build_system(T)
    p, q = compute_bandwidth(B)
    check(f"bandwidth dari B = dari T  (p={p}, q={q})", (p, q) == compute_bandwidth_from_T(T))
    check("pita dari T = pita diekstrak dari B",
          np.array_equal(build_band_from_T(T, p, q), build_band_storage(B, p, q)))
    L, U, P = lu_dense_partial_pivot(B)
    check("PB = LU", np.allclose(B[P], L @ U, atol=1e-14))
    z_known = np.arange(1.0, len(b) + 1.0)
    rhs_known = B @ z_known
    z_known_dense = solve_dense_lu(L, U, P, rhs_known)
    z_known_band = solve_banded_thomas(build_band_from_T(T, p, q), p, q, rhs_known)
    z_d = solve_dense_lu(L, U, P, b)
    z_b = solve_banded_thomas(build_band_from_T(T, p, q), p, q, b)
    check("dense: solusi buatan diketahui", np.allclose(z_known_dense, z_known, rtol=1e-12))
    check("banded: solusi buatan diketahui", np.allclose(z_known_band, z_known, rtol=1e-12))
    pi = z_b / z_b.sum()
    check("T^T pi = pi", np.linalg.norm(T.T @ pi - pi) < 1e-14)
    check("pi >= 0", pi.min() >= 0)

    # Uji 3: memaksa partial pivoting (diagonal kecil)
    print("\nUji 3: matriks pita acak dengan diagonal kecil (pivoting pasti terjadi)")
    rng = np.random.default_rng(0)
    worst, n_swap = 0.0, 0
    for _ in range(200):
        n = int(rng.integers(5, 40)); p, q = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(max(0, i - p), min(n, i + q + 1)):
                M[i, j] = rng.normal()
        M[np.arange(n), np.arange(n)] *= 0.05
        rhs = rng.normal(size=n)
        Lm, Um, Pm = lu_dense_partial_pivot(M)
        n_swap += int(np.any(Pm != np.arange(n)))
        pp, qq = compute_bandwidth(M)
        for x in (solve_dense_lu(Lm, Um, Pm, rhs),
                  solve_banded_thomas(build_band_storage(M, pp, qq), pp, qq, rhs)):
            # Galat mundur memakai norma infinity.
            worst = max(worst, normalized_backward_error(M, x, rhs))
    print(f"  {n_swap}/200 kasus melakukan pertukaran baris")
    check(f"galat mundur maks {worst:.1e} < 1e-13", worst < 1e-13)

    # Pivot terakhir nol harus ditolak, termasuk matriks 1 x 1.
    for singular in (np.array([[1.0, 1.0], [1.0, 1.0]]), np.zeros((1, 1))):
        p, q = compute_bandwidth(singular)
        for name, solver in (
            ("dense", lambda: lu_dense_partial_pivot(singular)),
            ("banded", lambda: solve_banded_thomas(
                build_band_storage(singular, p, q), p, q, np.ones(len(singular)))),
        ):
            try:
                solver()
            except (ValueError, np.linalg.LinAlgError):
                check(f"{name}: pivot terakhir nol, N={len(singular)}", True)
            else:
                check(f"{name}: pivot terakhir nol, N={len(singular)}", False)

    n_ok = sum(results)
    print(f"\n{n_ok}/{len(results)} uji lulus")
    if n_ok != len(results):
        raise SystemExit("ADA UJI YANG GAGAL - eksperimen dihentikan, periksa solvers.py")
    print("SEMUA UJI LULUS - lanjut ke eksperimen\n")

run_small_tests()

data = np.load(NPZ_PATH)
rows = []

for N in N_LIST:
    print(f"=== N = {N} ===")
    B = data[f"B_{N}"].astype(float)
    b = data[f"b_{N}"].astype(float)
    T, T_src = load_T(data, N, B)
    validate_input(N, B, b, T, T_src)

    p, q = compute_bandwidth(B)
    assert (p, q) == compute_bandwidth_from_T(T), "bandwidth dari B dan dari T berbeda"
    if p + q + 1 > N // 2:
        print(f"  PERINGATAN: pita lebar (p={p}, q={q}) relatif terhadap N={N}; "
              "keunggulan solver banded akan hilang.")

    # solver dense (faktorisasi + solve, keduanya dihitung)
    def dense_factorize_solve():
        L, U, P = lu_dense_partial_pivot(B)
        return solve_dense_lu(L, U, P, b)

    t_dense, z_dense = time_it(dense_factorize_solve)
    mem_dense, _ = peak_memory_bytes(dense_factorize_solve)

    # solver banded: pita dibangun LANGSUNG dari T (B dense tidak dibentuk)
    t_build, AB0 = time_it(lambda: build_band_from_T(T, p, q))

    def banded_solve():
        return solve_banded_thomas(AB0, p, q, b)

    t_band, z_band = time_it(banded_solve)
    mem_band, _ = peak_memory_bytes(banded_solve)

    # normalisasi pi = z / (1^T z)
    pi_dense = z_dense / z_dense.sum()
    pi_band = z_band / z_band.sum()

    # galat & kondisi (poin vii)
    r_dense = np.linalg.norm(T.T @ pi_dense - pi_dense)
    r_band = np.linalg.norm(T.T @ pi_band - pi_band)
    e_dense = abs(pi_dense.sum() - 1.0)
    e_band = abs(pi_band.sum() - 1.0)
    diff_methods = np.max(np.abs(pi_dense - pi_band))
    min_pi = min(pi_dense.min(), pi_band.min()) # distribusi stasioner harus >= 0
    cond_B = condition_number(B)
    cond_B_inf = condition_inf(B)
    Bz_res_dense = float(np.linalg.norm(B @ z_dense - b, ord=np.inf))
    Bz_res_band = float(np.linalg.norm(B @ z_band - b, ord=np.inf))
    backward_dense = normalized_backward_error(B, z_dense, b)
    backward_band = normalized_backward_error(B, z_band, b)

    # Estimasi galat terhadap solusi LU longdouble; bukan solusi eksak.
    # Presisi longdouble bergantung platform, bisa sama dengan float64.
    Lx, Ux, Px = lu_dense_partial_pivot(B, dtype=np.longdouble)
    z_ref = solve_dense_lu(Lx, Ux, Px, b.astype(np.longdouble))
    pi_ref = z_ref / z_ref.sum()
    fwd_dense = float(np.max(np.abs(pi_dense - pi_ref)) / np.max(np.abs(pi_ref)))
    fwd_band = float(np.max(np.abs(pi_band - pi_ref)) / np.max(np.abs(pi_ref)))
    bound = cond_B * np.finfo(float).eps / 2  # indikator sensitivitas, bukan batas galat universal

    # teori (poin vi)
    mem_dense_theory = 2 * N * N * 8 # L + U, masing-masing N x N
    W = 2 * p + q + 1
    mem_band_theory = N * W * 8 + 2 * N * 8 # AB + vektor b, x
    flop_dense, flop_band = theory_flops(N, p, q)

    rows.append({
        "N": N, "p": p, "q": q, "T_source": T_src,
        "t_dense_s": t_dense, "t_banded_s": t_band, "t_band_build_s": t_build,
        "speedup_x": t_dense / t_band if t_band > 0 else np.nan,
        "flop_dense_theory": flop_dense, "flop_banded_theory": flop_band,
        "mem_dense_measured_KB": mem_dense / 1024,
        "mem_banded_measured_KB": mem_band / 1024,
        "mem_dense_theory_KB": mem_dense_theory / 1024,
        "mem_banded_theory_KB": mem_band_theory / 1024,
        "cond_B_1norm": cond_B,
        "cond_B_infnorm": cond_B_inf,
        "r_dense": r_dense, "r_banded": r_band,
        "e_dense": e_dense, "e_banded": e_band,
        "Bz_residual_dense_inf": Bz_res_dense,
        "Bz_residual_banded_inf": Bz_res_band,
        "backward_err_dense_inf": backward_dense,
        "backward_err_banded_inf": backward_band,
        "max_diff_dense_vs_banded": diff_methods,
        "fwd_err_dense": fwd_dense, "fwd_err_banded": fwd_band, "kappa_x_u": bound,
        "longdouble_eps": float(np.finfo(np.longdouble).eps),
        "reference_higher_precision": bool(np.finfo(np.longdouble).eps < np.finfo(float).eps),
        "min_pi": min_pi,
        "halte_pi_max": int(np.argmax(pi_band)) + 1, "pi_max": float(pi_band.max()),
        "halte_pi_min": int(np.argmin(pi_band)) + 1, "pi_min": float(pi_band.min()),
        "pi_dense": pi_dense, "pi_banded": pi_band,
    })
    print(f"  T dari {T_src}  p={p} q={q}  t_dense={t_dense:.6f}s  t_banded={t_band:.6f}s  "
          f"cond(B)={cond_B:.4e}  r_dense={r_dense:.2e}  r_band={r_band:.2e}")
    if min_pi < -1e-12:
        print(f"  PERINGATAN: ada komponen pi negatif ({min_pi:.2e})")

df = pd.DataFrame(rows).drop(columns=["pi_dense", "pi_banded"])
pd.set_option("display.width", 200)
print()
print(df.to_string(index=False))

df.to_csv("output/tabel_eksperimen.csv", index=False)
np.savez("output/pi_solutions.npz",
         **{f"pi_dense_{r['N']}": r["pi_dense"] for r in rows},
         **{f"pi_banded_{r['N']}": r["pi_banded"] for r in rows})

# interpretasi (poin viii): 5 halte dengan proporsi terbesar per N
top = []
for r in rows:
    idx = np.argsort(r["pi_banded"])[::-1][:5]
    for rank, i in enumerate(idx, 1):
        top.append({"N": r["N"], "peringkat": rank, "halte": int(i) + 1,
                    "pi": r["pi_banded"][i], "N_x_pi": r["N"] * r["pi_banded"][i]})
pd.DataFrame(top).to_csv("output/top5_halte.csv", index=False)
print("\nTersimpan: output/tabel_eksperimen.csv, output/pi_solutions.npz, output/top5_halte.csv")
