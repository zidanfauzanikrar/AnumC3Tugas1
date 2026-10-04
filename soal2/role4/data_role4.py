"""Return sederhana dan matriks desain SETAR dua rezim, lag dua."""
import csv
from datetime import datetime
from pathlib import Path
import numpy as np


def load_prices(path, price_column=None, date_column=None):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=',;\t')
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        rows = list(reader)
        fields = reader.fieldnames or []
    norm = {s.strip().lower().replace('_', '').replace(' ', ''): s for s in fields}
    if price_column is None:
        price_column = next((norm[k] for k in ['close', 'closingprice', 'price', 'harga', 'hargapenutupan', 'adjclose'] if k in norm), None)
    if price_column not in fields:
        raise ValueError(f'{path}: pilih kolom harga dengan --price-column. Kolom tersedia: {fields}')
    if date_column is None:
        date_column = next((norm[k] for k in ['date', 'tanggal', 'datetime'] if k in norm), None)
    prices = np.array([float(r[price_column]) for r in rows])
    if len(prices) < 4 or not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError('Perlu >= 4 harga positif dan finite; data hilang tidak dihapus otomatis.')
    dates = None
    if date_column:
        if date_column not in fields:
            raise ValueError(f'Kolom tanggal {date_column} tidak ditemukan.')
        dates = []
        for row in rows:
            raw = row[date_column].strip()
            parsed = None
            for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%d/%m/%Y', '%Y/%m/%d']:
                try:
                    parsed = datetime.strptime(raw, fmt)
                    break
                except ValueError:
                    pass
            if parsed is None:
                raise ValueError(f'Tanggal {raw!r} tidak dikenali; ubah ke YYYY-MM-DD.')
            dates.append(parsed)
        order = sorted(range(len(dates)), key=lambda i: dates[i])
        prices = prices[order]
        dates = [dates[i] for i in order]
        if len(set(dates)) != len(dates):
            raise ValueError('Ada tanggal duplikat.')
    return prices, dates


def returns(prices):
    return (prices[1:] - prices[:-1]) / prices[:-1]


def design(r):
    r = np.asarray(r, dtype=float)
    if len(r) < 3:
        raise ValueError('Perlu minimal tiga return.')
    lag1, lag2 = r[1:-1], r[:-2]
    indicator = (lag1 >= 0).astype(float)
    A = np.column_stack([indicator, indicator * lag1, indicator * lag2,
                         1 - indicator, (1 - indicator) * lag1, (1 - indicator) * lag2])
    return A, r[2:].copy()


def build_datasets(train, test, continuous=True):
    A, b = design(returns(train))
    if continuous:
        all_r = returns(np.r_[train, test])
        full_A, full_b = design(all_r)
        At, bt = full_A[len(b):], full_b[len(b):]
        test_idx = np.arange(len(train), len(train) + len(test))
    else:
        At, bt = design(returns(test))
        test_idx = np.arange(len(train) + 3, len(train) + len(test))
    return A, b, At, bt, np.arange(3, len(train)), test_idx
