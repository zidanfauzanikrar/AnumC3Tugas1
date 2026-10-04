# Soal 2 - Role 4

Jalankan dari folder `AnumC3Tugas1`:

```bash
python3 -m pip install -r soal2/role4/requirements.txt
python3 soal2/role4/run_role4.py
```

Hasil ada di `soal2/role4/output_role4/`.
Evaluasi test memakai one-step-ahead dengan riwayat aktual dari train diteruskan ke test.

Untuk menjalankan pengujian:

```bash
python3 -m unittest discover -s soal2/role4 -p 'test_*.py' -v
```
