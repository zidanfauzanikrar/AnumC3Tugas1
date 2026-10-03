import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

os.makedirs("plots", exist_ok=True)

df = pd.read_csv("output/tabel_eksperimen.csv")
N = df["N"].values

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

COLOR_DENSE = "#d1495b"
COLOR_BAND = "#2e7d32"


# 1. waktu eksekusi vs N (log-log) + garis referensi O(N^3) & O(N)
fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(N, df["t_dense_s"], "o-", color=COLOR_DENSE, label="Dense LU (partial pivoting)", linewidth=2, markersize=7)
ax.loglog(N, df["t_banded_s"], "s-", color=COLOR_BAND, label="Banded Thomas (partial pivoting)", linewidth=2, markersize=7)

# garis referensi teoritis, diskalakan supaya berimpit di titik N terakhir
ref_cubic = df["t_dense_s"].iloc[-1] * (N / N[-1]) ** 3
ref_linear = df["t_banded_s"].iloc[-1] * (N / N[-1]) ** 1
ax.loglog(N, ref_cubic, "--", color=COLOR_DENSE, alpha=0.5, label=r"referensi $O(N^3)$")
ax.loglog(N, ref_linear, "--", color=COLOR_BAND, alpha=0.5, label=r"referensi $O(N)$")

ax.set_xlabel("N (jumlah halte)")
ax.set_ylabel("Waktu eksekusi (detik)")
ax.set_title("Waktu Eksekusi Solver vs Ukuran Matriks N")
ax.set_xticks(N)
ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())
ax.legend()
fig.tight_layout()
fig.savefig("plots/01_waktu_vs_N.png")
plt.close(fig)

# 2. memori vs N (log-log): terukur (tracemalloc) vs teoritis
fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(N, df["mem_dense_measured_KB"], "o-", color=COLOR_DENSE, label="Dense - terukur (tracemalloc)", linewidth=2, markersize=7)
ax.loglog(N, df["mem_dense_theory_KB"], "o--", color=COLOR_DENSE, alpha=0.5, label="Dense - teoritis (2N²·8B)")
ax.loglog(N, df["mem_banded_measured_KB"], "s-", color=COLOR_BAND, label="Banded - terukur (tracemalloc)", linewidth=2, markersize=7)
ax.loglog(N, df["mem_banded_theory_KB"], "s--", color=COLOR_BAND, alpha=0.5, label="Banded - teoritis (N(2p+q+1)·8B)")

ax.set_xlabel("N (jumlah halte)")
ax.set_ylabel("Memori puncak (KB)")
ax.set_title("Kebutuhan Memori Solver vs Ukuran Matriks N")
ax.set_xticks(N)
ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("plots/02_memori_vs_N.png")
plt.close(fig)

# 3. speedup (dense time / banded time) vs N
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(N, df["speedup_x"], "D-", color="#1565c0", linewidth=2, markersize=7)
for x, y in zip(N, df["speedup_x"]):
    ax.annotate(f"{y:.1f}×", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
ax.set_xscale("log")
ax.set_xticks(N)
ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())
ax.set_xlabel("N (jumlah halte)")
ax.set_ylabel("Speedup = t_dense / t_banded")
ax.set_title("Percepatan Solver Banded relatif terhadap Solver Dense")
fig.tight_layout()
fig.savefig("plots/03_speedup_vs_N.png")
plt.close(fig)

# 4. condition number vs N
fig, ax = plt.subplots(figsize=(7, 5))
ax.semilogy(N, df["cond_B_1norm"], "^-", color="#6a1b9a", linewidth=2, markersize=8)
for x, y in zip(N, df["cond_B_1norm"]):
    ax.annotate(f"{y:.2e}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
ax.set_xscale("log")
ax.set_xticks(N)
ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())
ax.set_xlabel("N (jumlah halte)")
ax.set_ylabel(r"$\kappa_1(B) = \|B\|_1 \|B^{-1}\|_1$")
ax.set_title("Condition Number Matriks B (norma-1) vs N")
fig.tight_layout()
fig.savefig("plots/04_condition_number_vs_N.png")
plt.close(fig)

# 5. residual r dan e_norm vs N (kedua solver)
fig, ax = plt.subplots(figsize=(7, 5))
eps = 1e-19
ax.semilogy(N, df["r_dense"] + eps, "o-", color=COLOR_DENSE, label=r"$r$ = $\|T^T\pi-\pi\|$ (dense)", linewidth=2)
ax.semilogy(N, df["r_banded"] + eps, "s-", color=COLOR_BAND, label=r"$r$ = $\|T^T\pi-\pi\|$ (banded)", linewidth=2)
ax.semilogy(N, df["max_diff_dense_vs_banded"] + eps, "d--", color="#888888", label="beda maks. $\\pi$ dense vs banded")
ax.semilogy(N, df["e_dense"] + eps, "o:", color=COLOR_DENSE, alpha=0.6, label=r"$e_{norm}=|1^T\pi-1|$ (dense)")
ax.semilogy(N, df["e_banded"] + eps, "s:", color=COLOR_BAND, alpha=0.6, label=r"$e_{norm}=|1^T\pi-1|$ (banded)")
mach_eps = np.finfo(float).eps
ax.axhline(mach_eps, color="black", linestyle=":", linewidth=1, label=f"machine epsilon ({mach_eps:.1e})")
ax.set_xscale("log")
ax.set_xticks(N)
ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())
ax.set_xlabel("N (jumlah halte)")
ax.set_ylabel("Residual (skala log)")
ax.set_title("Residual dan Selisih Solusi Dense vs Banded")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("plots/05_residual_vs_N.png")
plt.close(fig)

# 6. flop teoritis vs N (menghubungkan analisis kompleksitas dengan eksperimen)
fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(N, df["flop_dense_theory"], "o-", color=COLOR_DENSE, label=r"Dense: $\frac{2}{3}N^3 + 2N^2$", linewidth=2)
ax.loglog(N, df["flop_banded_theory"], "s-", color=COLOR_BAND, label=r"Banded: $2Np(p+q+1) + 2N(p+q) + N$", linewidth=2)
ax.set_xticks(N)
ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())
ax.set_xlabel("N (jumlah halte)")
ax.set_ylabel("Jumlah flop (teoritis)")
ax.set_title("Kompleksitas Operasi Teoritis vs N")
ax.legend()
fig.tight_layout()
fig.savefig("plots/06_flop_teoritis_vs_N.png")
plt.close(fig)

# 7. (poin viii) pi_i terhadap nomor halte, satu panel per N
pis = np.load("output/pi_solutions.npz")
fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=False)
for ax, n in zip(axes.ravel(), N):
    pi = pis[f"pi_banded_{n}"]
    halte = np.arange(1, n + 1)
    ax.plot(halte, pi, "-", color="#1565c0", linewidth=1.5, marker="o" if n <= 32 else None, markersize=3)
    ax.axhline(1 / n, color="gray", linestyle="--", linewidth=1, label="seragam 1/N")
    i = int(np.argmax(pi))
    ax.plot(i + 1, pi[i], "*", color=COLOR_DENSE, markersize=12, label=f"maks: halte {i + 1}")
    j = int(np.argmin(pi))
    ax.plot(j + 1, pi[j], "v", color="#ef6c00", markersize=9, label=f"min: halte {j + 1}")
    ax.set_title(f"N = {n}")
    ax.set_xlabel("nomor halte i")
    ax.set_ylabel(r"$\pi_i$")
    ax.legend(fontsize=8)
fig.suptitle("Distribusi steady state $\\pi_i$ terhadap nomor halte")
fig.tight_layout()
fig.savefig("plots/07_pi_vs_halte.png")
plt.close(fig)

# 8. profil ternormalisasi N*pi_i terhadap posisi relatif i/N (semua N ditumpuk)
fig, ax = plt.subplots(figsize=(8, 5))
cmap = plt.get_cmap("viridis")
for k, n in enumerate(N):
    pi = pis[f"pi_banded_{n}"]
    ax.plot(np.arange(1, n + 1) / n, n * pi, color=cmap(k / (len(N) - 1)), linewidth=1.5, label=f"N = {n}")
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="seragam")
ax.set_xlabel("posisi relatif halte i/N")
ax.set_ylabel(r"$N \cdot \pi_i$  (1 = rata-rata)")
ax.set_title("Profil steady state ternormalisasi untuk semua N")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("plots/08_profil_pi_ternormalisasi.png")
plt.close(fig)

# 9. (poin viii, penjelasan) pi_i dan drift per halte untuk N = 512
n_focus = 512
T_focus = np.loadtxt(f"data/T_{n_focus}.csv", delimiter=",")
pi_f = pis[f"pi_banded_{n_focus}"]
idx = np.arange(1, n_focus + 1)
drift = T_focus @ idx - idx   # drift_i = sum_j T_ij * j - i

i_max = int(np.argmax(pi_f)) + 1
i_min = int(np.argmin(pi_f)) + 1

# drift di bagian dalam koridor (tanpa 10 halte di tiap ujung), skala diperbesar
inner = slice(10, n_focus - 10)
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(idx[inner], drift[inner], color=COLOR_BAND)
ax.axhline(0, color="gray", linestyle="--")
ax.axvline(i_max, color=COLOR_DENSE, linestyle=":", label=f"maks: halte {i_max}")
ax.axvline(i_min, color="#ef6c00", linestyle=":", label=f"min: halte {i_min}")
ax.set_xlabel("nomor halte i")
ax.set_ylabel("drift")
ax.set_title("Drift di bagian dalam koridor (N = 512)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("plots/10_drift_bagian_dalam.png")
plt.close(fig)

# rata-rata drift 30 halte di kiri dan kanan titik maks dan min
def win(i, w=30):
    return drift[i - 1 - w:i - 1].mean(), drift[i:i + w].mean()
print("sekitar maks (kiri, kanan):", win(i_max))
print("sekitar min  (kiri, kanan):", win(i_min))

# variansi lompatan lokal dan hasil kali pi * variansi
var = np.array([T_focus[i] @ (idx - (i + 1)) ** 2 for i in range(n_focus)])
prod = (pi_f * var)[10:-10]
print("pi*variansi di bagian dalam: rata-rata", prod.mean(), "simpangan relatif", prod.std() / prod.mean())

fig, ax1 = plt.subplots(figsize=(9, 4.5))
l1, = ax1.plot(idx, pi_f, color="#1565c0", linewidth=1.5, label=r"$\pi_i$")
ax1.axvline(i_max, color=COLOR_DENSE, linestyle=":", linewidth=1.2, label=f"maks: halte {i_max}")
ax1.axvline(i_min, color="#ef6c00", linestyle=":", linewidth=1.2, label=f"min: halte {i_min}")
ax1.set_xlabel("nomor halte i")
ax1.set_ylabel(r"$\pi_i$", color="#1565c0")

ax2 = ax1.twinx()
ax2.grid(False)
l2, = ax2.plot(idx, drift, color=COLOR_BAND, linewidth=1, alpha=0.85, label="drift")
ax2.axhline(0, color="gray", linestyle="--", linewidth=1)
ax2.set_ylabel(r"drift $=\sum_j T_{ij}\,j - i$", color=COLOR_BAND)

handles = [l1, l2] + ax1.lines[1:3]
ax1.legend(handles=handles, labels=[h.get_label() for h in handles], fontsize=8, loc="upper right")
ax1.set_title(f"Distribusi steady state dan drift per halte (N = {n_focus})")
fig.tight_layout()
fig.savefig("plots/09_pi_dan_drift_N512.png")
plt.close(fig)

# cek hipotesis: di mana drift berganti tanda
sb = np.signbit(drift)
cross = np.where(sb[:-1] != sb[1:])[0]
cross = cross[(cross > 10) & (cross < n_focus - 10)]   # buang lonjakan di ujung
plus_to_minus = [int(i) + 1 for i in cross if not sb[i]]
minus_to_plus = [int(i) + 1 for i in cross if sb[i]]
print("+ ke -:", plus_to_minus)
print("- ke +:", minus_to_plus)

print("Semua grafik tersimpan di folder plots/")
for f in sorted(__import__("os").listdir("plots")):
    print(" -", f)
