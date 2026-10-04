import pandas as pd

# ---------- Tabel 1: halte dan nilai π maks/min per N ----------
df = pd.read_csv('output/tabel_eksperimen.csv')

tabel1 = pd.DataFrame({
    'N': df['N'],
    'Halte π maks': df['halte_pi_max'],
    'π maks': df['pi_max'].round(6),
    'Posisi maks (%)': (df['halte_pi_max'] / df['N'] * 100).round(2),
    'Halte π min': df['halte_pi_min'],
    'π min': df['pi_min'].round(6),
    'Posisi min (%)': (df['halte_pi_min'] / df['N'] * 100).round(2),
    'Rasio maks/min': (df['pi_max'] / df['pi_min']).round(2),
})
tabel1.to_csv('output/tabel_pi_ekstrem.csv', index=False)

# ---------- Tabel 2: lima halte teratas, N = 512 ----------
top5 = pd.read_csv('output/top5_halte.csv')

t2 = top5[top5.N == 512].copy()
t2['posisi_relatif_%'] = (t2.halte / 512 * 100).round(1)
t2['pi'] = t2['pi'].round(8)
t2['N_x_pi'] = t2['N_x_pi'].round(6)
t2[['peringkat', 'halte', 'pi', 'N_x_pi', 'posisi_relatif_%']].to_csv(
    'output/tabel_top5_N512.csv', index=False)

# ---------- Tabel 3: dua puncak di setiap N ----------
rows = []
for N, g in top5.groupby('N'):
    g = g.sort_values('peringkat')
    p1, p2 = g.pi.iloc[0], g.pi.iloc[1]
    rows.append({
        'N': N,
        'Halte peringkat 1': g.halte.iloc[0],
        'Posisi #1 (%)': round(g.halte.iloc[0] / N * 100, 1),
        'Halte peringkat 2': g.halte.iloc[1],
        'Posisi #2 (%)': round(g.halte.iloc[1] / N * 100, 1),
        'Selisih relatif': f"{(p1 - p2) / p1:.2e}",
        'N·π maks': round(g.N_x_pi.iloc[0], 4),
    })
tabel3 = pd.DataFrame(rows)
tabel3.to_csv('output/tabel_dua_puncak.csv', index=False)

print(tabel1.to_string(index=False), '\n')
print(t2.to_string(index=False), '\n')
print(tabel3.to_string(index=False))
print("\nTersimpan di folder output/: tabel_pi_ekstrem.csv, tabel_top5_N512.csv, tabel_dua_puncak.csv")