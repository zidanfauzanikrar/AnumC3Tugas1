import numpy as np

def load_matrix_numpy(filepath):
    T = np.loadtxt(filepath, delimiter=',')
    return T

N_list = [16, 32, 64, 128, 256, 512]
matrices = {}

for N in N_list:
    filepath = f"data/T_{N}.csv" 
    matrices[N] = load_matrix_numpy(filepath)
    print(f"N={N}: shape={matrices[N].shape}")

print(f"{'N':<6}{'Square':<10}{'Nonneg':<10}{'RowSum=1':<12}")
print("-" * 38)

for N, T in matrices.items():
    is_square = (T.shape[0] == T.shape[1] == N)
    all_nonneg = np.all(T >= 0)
    row_sums = np.sum(T, axis=1)
    rows_sum_to_one = np.allclose(row_sums, 1.0, atol=1e-6)
    
    print(f"{N:<6}{str(is_square):<10}{str(all_nonneg):<10}{str(rows_sum_to_one):<12}")
    
    if not rows_sum_to_one:
        bad_rows = np.where(~np.isclose(row_sums, 1.0, atol=1e-6))[0]
        print(f"Baris bermasalah: {bad_rows}")

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
for N, T in matrices.items():
    B_matrices[N], b_vectors[N] = build_B_b(T)
    print(f"N={N}: B shape={B_matrices[N].shape}, b shape={b_vectors[N].shape}")

np.savez('output_B_b/B_b_all.npz', **{f'B_{N}': B_matrices[N] for N in N_list},
                                     **{f'b_{N}': b_vectors[N] for N in N_list})