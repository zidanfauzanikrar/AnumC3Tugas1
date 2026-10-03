import pandas as pd

df = pd.read_csv('output/tabel_eksperimen.csv')

tabel1 = pd.DataFrame({
    'N': df['N'],
    'Halte π maks': df['halte_pi_max'],
    'π maks': df['pi_max'],
    'Posisi relatif maks (%)': df['halte_pi_max'] / df['N'] * 100,
    'Halte π min': df['halte_pi_min'],
    'π min': df['pi_min'],
    'Posisi relatif min (%)': df['halte_pi_min'] / df['N'] * 100,
    'Rasio maks/min': df['pi_max'] / df['pi_min'],
})

tabel1 = tabel1.round({'π maks': 6, 'π min': 6,
                       'Posisi relatif maks (%)': 2,
                       'Posisi relatif min (%)': 2,
                       'Rasio maks/min': 2})

print(tabel1.to_string(index=False))
tabel1.to_csv('output/tabel_pi_ekstrem.csv', index=False)
print("\nTersimpan: output/tabel_pi_ekstrem.csv")