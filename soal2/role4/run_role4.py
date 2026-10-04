"""Eksperimen Role 4; jalankan python run_role4.py --help."""
import argparse
import csv
import json
import platform
import time
import tracemalloc
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_role4 import load_prices, build_datasets
from solvers_role4 import householder_qr, normal_equations, first_reflection, condition_2
from role3_adapter import load_role3, verify_role3, role3_normal


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def measure(solver, A, b, repeats):
    solver(A, b)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        solver(A, b)
        times.append(time.perf_counter() - start)
    tracemalloc.start()
    result = solver(A, b)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, float(np.median(times) * 1000), peak / 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=str(Path(__file__).resolve().parents[1]/'data/stock_train.csv'))
    parser.add_argument('--test', default=str(Path(__file__).resolve().parents[1]/'data/stock_test.csv'))
    parser.add_argument('--price-column')
    parser.add_argument('--date-column')
    parser.add_argument('--out', default=str(Path(__file__).resolve().parent/'output_role4'))
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--test-mode', choices=['continuous', 'independent'], default='continuous')
    parser.add_argument('--demo', action='store_true', help='Data sintetis untuk uji alur; bukan hasil tugas.')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats harus positif.')
    out = Path(args.out)
    if args.demo:
        out = out / 'DEMO_SYNTHETIC'
        rng = np.random.default_rng(2026)
        r = rng.normal(0.0004, 0.012, 382)
        prices = 100 * np.cumprod(np.r_[1.0, 1 + r])
        train, test = prices[:303], prices[303:]
        dates_train = dates_test = None
    else:
        for path in [args.train, args.test]:
            if not Path(path).is_file():
                parser.error(f'File belum ada: {path}. Masukkan CSV asli, atau gunakan --demo untuk uji sintetis.')
        train, dates_train = load_prices(args.train, args.price_column, args.date_column)
        test, dates_test = load_prices(args.test, args.price_column, args.date_column)
        if (dates_train is None) != (dates_test is None):
            raise ValueError('Kedua CSV harus sama-sama punya tanggal atau sama-sama tanpa tanggal.')
        if dates_train and dates_train[-1] >= dates_test[0]:
            raise ValueError('Tanggal train/test tumpang tindih atau urutannya salah.')
    out.mkdir(parents=True, exist_ok=True)
    continuous = args.test_mode == 'continuous'
    A, b, At, bt, train_idx, test_idx = build_datasets(train, test, continuous)
    m, n = A.shape
    notebook = Path(__file__).resolve().parents[1]/'Peran3_Soal2.ipynb'
    role3 = load_role3(notebook) if not args.demo else None
    crosscheck = verify_role3(role3, train, A, b) if role3 else {'status': 'DEMO_ONLY'}
    solvers = {'QR_implicit': lambda a, b: householder_qr(a, b),
               'QR_explicit_Q': lambda a, b: householder_qr(a, b, True),
               'Normal_Role3': (lambda a,b: role3_normal(role3,a,b)) if role3 else normal_equations}
    rows, solutions, failures = [], {}, {}
    for name, solver in solvers.items():
        try:
            result, milliseconds, peak = measure(solver, A, b, args.repeats)
        except ValueError as exc:
            failures[name] = str(exc)
            continue
        x = result[0] if isinstance(result, tuple) else result
        solutions[name] = x
        qr_flops = 2 * m * n * n - 2 * n**3 / 3
        flops = (qr_flops if name.startswith('QR') else 2*m*n*n + n**3/3)
        if name == 'QR_explicit_Q':
            flops += 4 * m * sum(m-k for k in range(n))
        rows.append(dict(method=name, residual_l2=float(np.sqrt(np.sum((A@x-b)**2))),
                         rmse_train=float(np.sqrt(np.mean((A@x-b)**2))),
                         rmse_test=float(np.sqrt(np.mean((At@x-bt)**2))),
                         median_ms=milliseconds, peak_tracemalloc_KiB=peak,
                         leading_flops_estimate=flops,
                         explicit_Q_bytes=m*m*8 if name == 'QR_explicit_Q' else 0))
    if 'QR_implicit' not in solutions:
        raise ValueError(f'QR tidak dapat dipakai: {failures}. Periksa dukungan data setiap rezim.')
    x = solutions['QR_implicit']
    v, H, HA = first_reflection(A)
    _, R, c, Q = householder_qr(A, b, True)
    metadata = dict(dataset='DEMO_SYNTHETIC' if args.demo else 'USER_CSV',
                    train_file=None if args.demo else str(Path(args.train).resolve()),
                    test_file=None if args.demo else str(Path(args.test).resolve()),
                    test_mode=args.test_mode, A_shape=list(A.shape), A_test_shape=list(At.shape),
                    regime1_train=int(A[:,0].sum()), regime2_train=int(A[:,3].sum()),
                    kappa2_A=condition_2(A), kappa2_AtA=condition_2(A.T@A),
                    H1_subdiagonal_max=float(np.max(np.abs(HA[1:,0]))),
                    H1_orthogonality_max=float(np.max(np.abs(H.T@H-np.eye(m)))),
                    QR_factorization_max=float(np.max(np.abs(Q@R-A))),
                    Q_orthogonality_max=float(np.max(np.abs(Q.T@Q-np.eye(m)))),
                    chosen_method='QR_implicit', failures=failures, role3_crosscheck=crosscheck,
                    selection_reason='Pilihan berdasarkan stabilitas algoritma, bukan pemilihan pada test set.',
                    python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
                    repeats=args.repeats, price_column=args.price_column or 'auto',
                    evaluation='One-step-ahead menggunakan lag aktual; parameter di-fit hanya pada train.')
    if 'Normal_Role3' in solutions:
        metadata['coefficient_difference_max'] = float(np.max(np.abs(x-solutions['Normal_Role3'])))
    np.savez_compressed(out/'matrices_and_H1.npz', A=A, b=b, A_test=At, b_test=bt,
                        v1=v, H1=H, H1A=HA, R=R, Q=Q, Qt_b=c, x_qr=x)
    write_csv(out/'comparison.csv', rows)
    labels = ['alpha1','phi11','phi12','alpha2','phi21','phi22']
    write_csv(out/'coefficients.csv', [dict(parameter=label, **{k: float(z[i]) for k,z in solutions.items()}) for i,label in enumerate(labels)])
    predictions = []
    all_dates = dates_train + dates_test if dates_train else None
    for split, idx, a, target in [('train',train_idx,A,b),('test',test_idx,At,bt)]:
        for j, i in enumerate(idx):
            predictions.append(dict(split=split, price_row_1based=int(i+1),
                                    date=all_dates[i].strftime('%Y-%m-%d') if all_dates else '', actual=float(target[j]),
                                    predicted=float(a[j]@x), residual=float(target[j]-a[j]@x)))
    write_csv(out/'predictions.csv', predictions)
    residual = b-A@x
    squared = residual**2
    total = float(squared.sum())
    order = np.argsort(-squared)[:min(10,m)]
    write_csv(out/'largest_residuals.csv', [dict(price_row_1based=int(train_idx[i]+1), actual=float(b[i]),
              predicted=float(A[i]@x), residual=float(residual[i]), squared_residual=float(squared[i]),
              share_train_sse=float(squared[i]/total) if total else 0.0) for i in order])
    test_residual = bt - At@x
    test_squared = test_residual**2
    test_total = float(test_squared.sum())
    test_order = np.argsort(-test_squared)[:min(10,len(bt))]
    write_csv(out/'largest_residuals_test.csv', [dict(price_row_1based=int(test_idx[i]+1),
        date=all_dates[test_idx[i]].strftime('%Y-%m-%d') if all_dates else '',
        actual=float(bt[i]), predicted=float(At[i]@x), residual=float(test_residual[i]),
        squared_residual=float(test_squared[i]),
        share_test_sse=float(test_squared[i]/test_total) if test_total else 0.0) for i in test_order])
    metadata['outlier_contribution'] = dict(train_largest_residual_sse_share=float(squared[order[0]]/total),
        test_largest_residual_sse_share=float(test_squared[test_order[0]]/test_total),
        test_largest_residual_date=all_dates[test_idx[test_order[0]]].strftime('%Y-%m-%d') if all_dates else None)
    np.savetxt(out/'H1A_preview.csv',HA[:10],delimiter=',')
    h1_notes = [f'H1 = I - 2 v1 v1^T; H1 berukuran {m} x {m}.',
        f'Norma kolom pertama = {np.sqrt(A[:,0]@A[:,0]):.16g}.',
        'Kolom pertama H1A adalah -norma(A[:,0])*e1 hingga pembulatan.',
        f'Max subdiagonal kolom pertama H1A: {metadata["H1_subdiagonal_max"]:.6e}.',
        'H1 dan H1A lengkap tersimpan di matrices_and_H1.npz.',
        'H1A_preview.csv berisi sepuluh baris pertama, bukan matriks lengkap.']
    (out/'H1_verification.txt').write_text('\n'.join(h1_notes),encoding='utf-8')
    k = int(np.argmin(b))
    changed_train = train.copy()
    changed_train[train_idx[k]] *= 0.8
    Ashock, bshock, _, _, _, _ = build_datasets(changed_train, test, continuous)
    xshock = householder_qr(Ashock, bshock)[0]
    metadata['synthetic_shock'] = dict(label='SENSITIVITY_ONLY_NOT_REAL_DATA', price_drop_fraction=0.2,
        price_row_1based=int(train_idx[k]+1), baseline_train_sse=total,
        shocked_sse_fixed_model=float(np.sum((Ashock@x-bshock)**2)),
        shocked_sse_refit=float(np.sum((Ashock@xshock-bshock)**2)),
        coefficient_change_l2=float(np.sqrt(np.sum((xshock-x)**2))))
    (out/'metrics.json').write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
    title = 'DEMO SINTETIS — ' if args.demo else ''
    fig, ax = plt.subplots(figsize=(12,4))
    for idx, target, a, split in [(train_idx,b,A,'Train'),(test_idx,bt,At,'Test')]:
        axis_values = [all_dates[i] for i in idx] if all_dates else idx+1
        ax.plot(axis_values,target,label=f'Aktual {split}',lw=1)
        ax.plot(axis_values,a@x,label=f'Prediksi {split}',lw=1,alpha=.8)
    ax.axvline(dates_test[0] if all_dates else len(train)+.5,color='black',ls='--',label='Batas train/test')
    ax.set(xlabel='Tanggal' if all_dates else 'Nomor observasi harga (kronologis)',ylabel='Return',title=title+'SETAR: aktual dan prediksi one-step-ahead')
    ax.legend(ncol=3); fig.tight_layout(); fig.savefig(out/'01_overlay_train_test.png',dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(train_idx+1,residual,lw=1)
    ax.scatter(train_idx[order]+1,residual[order],c='red',s=20,label='10 residual kuadrat terbesar')
    ax.axhline(0,color='black',lw=.7); ax.legend()
    ax.set(xlabel='Nomor observasi harga train',ylabel='Aktual − prediksi',title=title+'Residual train dan kontribusi outlier')
    fig.tight_layout(); fig.savefig(out/'02_residual_train.png',dpi=180); plt.close(fig)
    model = [f'Rt = {x[0]:+.10g} {x[1]:+.10g} R(t-1) {x[2]:+.10g} R(t-2), jika R(t-1) >= 0',
             f'Rt = {x[3]:+.10g} {x[4]:+.10g} R(t-1) {x[5]:+.10g} R(t-2), jika R(t-1) < 0',
             'Persamaan di atas adalah prediksi; model stokastik menambahkan epsilon_t.']
    for reg, start in [('Bullish',0),('Bearish',3)]:
        for lag in [1,2]:
            coef = x[start+lag]
            behavior = 'penerusan arah (momentum)' if coef > 0 else 'pembalikan arah' if coef < 0 else 'tidak ada efek linear'
            model.append(f'{reg}, lag {lag}: {coef:+.8g} mengindikasikan {behavior}, dengan prediktor lain tetap.')
    model.append('Tanda koefisien per lag tidak membuktikan stasioneritas/mean reversion global SETAR; dinamika kedua lag dan perpindahan rezim juga berpengaruh.')
    (out/'model_interpretation.txt').write_text('\n'.join(model),encoding='utf-8')
    print('DEMO SINTETIS, BUKAN HASIL TUGAS' if args.demo else 'HASIL DATA CSV')
    print(f'A: {A.shape}; A_test: {At.shape}; output: {out.resolve()}')
    for row in rows:
        print(row['method'], 'RMSE train:',row['rmse_train'],'RMSE test:',row['rmse_test'])
    print('Cross-check Role 3:',crosscheck)
    print('H1A subdiagonal maksimum:',metadata['H1_subdiagonal_max'])


if __name__ == '__main__':
    main()
