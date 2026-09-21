import numpy as np
import pandas as pd
import os
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

N_list = [16, 32, 64, 128, 256, 512]
matrices = {}
for N in N_list:
    matrices[N] = np.loadtxt(f'data/T_{N}.csv', delimiter=',')

def is_irreducible(T, tol=1e-12):
    """
    Cek irreducibility via strongly connected components pada graf adjacency T,
    bukan lewat perpangkatan matriks (untuk hindari underflow floating-point).
    """
    adj = csr_matrix(T > tol)
    n_components, _ = connected_components(adj, directed=True, connection='strong')
    return n_components == 1

def build_B_b(T):
    N = T.shape[0]
    A = np.eye(N) - T.T
    B = A.copy()
    B[0, :] = 0
    B[0, 0] = 1
    b = np.zeros(N)
    b[0] = 1
    return B, b

B_matrices = {}
b_vectors = {}
for N in N_list:
    B_matrices[N], b_vectors[N] = build_B_b(matrices[N])

rows_validasi = []
for N in N_list:
    T = matrices[N]
    is_square = (T.shape[0] == T.shape[1] == N)
    all_nonneg = np.all(T >= 0)
    row_sums = np.sum(T, axis=1)
    rows_sum_to_one = np.allclose(row_sums, 1.0, atol=1e-6)
    irreducible = is_irreducible(T)
    rows_validasi.append({
        'N': N, 'Square': is_square, 'Nonneg': all_nonneg,
        'RowSum=1': rows_sum_to_one, 'Irreducible': irreducible
    })

df_validasi = pd.DataFrame(rows_validasi)
print(df_validasi.to_string(index=False))

os.makedirs('output_B_b', exist_ok=True)
df_validasi.to_csv('output_B_b/tabel_validasi.csv', index=False)
print("Tabel validasi diexport ke output_B_b/tabel_validasi.csv\n")

np.savez('output_B_b/B_b_all.npz',
          **{f'B_{N}': B_matrices[N] for N in N_list},
          **{f'b_{N}': b_vectors[N] for N in N_list})

print("Semua B dan b berhasil disimpan ke output_B_b/B_b_all.npz")