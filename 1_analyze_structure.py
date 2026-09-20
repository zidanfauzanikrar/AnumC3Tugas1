import numpy as np
import pandas as pd

data = np.load('output_B_b/B_b_all.npz')
N_list = [16, 32, 64, 128, 256, 512]

def compute_bandwidth(B, tol=1e-12):
    rows, cols = np.nonzero(np.abs(B) > tol)
    diff = rows - cols
    p = int(diff.max()) if len(diff) > 0 and diff.max() > 0 else 0
    q = int((-diff).max()) if len(diff) > 0 and (-diff).max() > 0 else 0
    return p, q

rows = []
for N in N_list:
    B = data[f'B_{N}']
    p, q = compute_bandwidth(B)
    dense_kb = N * N * 8 / 1024
    banded_kb = N * (p + q + 1) * 8 / 1024
    hemat = (1 - banded_kb / dense_kb) * 100
    rows.append({
        'N': N, 'p': p, 'q': q,
        'Dense (KB)': round(dense_kb, 2),
        'Banded (KB)': round(banded_kb, 2),
        'Hemat (%)': round(hemat, 2)
    })

df_bandwidth = pd.DataFrame(rows)

print(df_bandwidth.to_string(index=False))

df_bandwidth.to_csv('output_B_b/tabel_bandwidth.csv', index=False)
print("\nTabel bandwidth berhasil diexport ke output_B_b/tabel_bandwidth.csv")