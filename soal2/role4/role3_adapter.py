"""Gunakan fungsi notebook Role 3 tanpa menjalankan cell eksperimennya."""
import ast
import hashlib
import json
from math import sqrt, log10
from pathlib import Path
import numpy as np


def load_role3(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Notebook Role 3 belum ada: {path}')
    notebook = json.loads(path.read_text(encoding='utf-8'))
    functions = []
    constants = {}
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            tree = ast.parse(''.join(cell['source']))
            functions.extend(node for node in tree.body if isinstance(node, ast.FunctionDef))
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in ['LAG','THRESHOLD']:
                            constants[target.id] = ast.literal_eval(node.value)
    if constants != {'LAG': 2, 'THRESHOLD': 0.0}:
        raise ValueError('Konvensi lag/threshold Role 3 berubah; sesuaikan model Role 4.')
    namespace = {'sqrt': sqrt, 'log10': log10, **constants}
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(module, str(path), 'exec'), namespace)
    needed = ['hitung_return', 'bentuk_sistem', 'gram', 'transpose', 'matvec', 'cholesky', 'cholesky_solve']
    if not all(name in namespace for name in needed):
        raise ValueError('API notebook Role 3 berubah; sesuaikan adapter.')
    namespace['notebook_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return namespace


def verify_role3(api, prices, A, b):
    r = api['hitung_return'](prices.tolist())
    ref_A, ref_b, regimes = api['bentuk_sistem'](r)
    ref_A, ref_b = np.array(ref_A), np.array(ref_b)
    if ref_A.shape != A.shape or ref_b.shape != b.shape:
        raise ValueError('Dimensi desain tidak sama dengan Role 3.')
    err_A = float(np.max(np.abs(ref_A-A)))
    err_b = float(np.max(np.abs(ref_b-b)))
    if err_A != 0 or err_b != 0:
        raise ValueError(f'A/b tidak identik dengan Role 3: {err_A}, {err_b}')
    return dict(status='IDENTICAL_A_AND_B', max_difference_A=err_A, max_difference_b=err_b,
                regime1=regimes.count(1), regime2=regimes.count(2), notebook_sha256=api['notebook_sha256'])


def role3_normal(api, A, b):
    matrix, rhs = A.tolist(), b.tolist()
    G = api['gram'](matrix)
    c = api['matvec'](api['transpose'](matrix), rhs)
    return np.array(api['cholesky_solve'](api['cholesky'](G), c))
