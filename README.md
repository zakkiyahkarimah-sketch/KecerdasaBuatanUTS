# Sistem Klasifikasi Prestasi Akademik Matematika Siswa Menggunakan Decision Tree

Aplikasi web berbasis **Flask** dan **scikit-learn** yang mereplikasi dan mengembangkan publikasi ilmiah:
> **"Klasifikasi Prestasi Akademik Matematika Siswa Menggunakan Decision Tree pada Dataset Student Performance"**  
> *Elisabeth Jessica Sitanggang, Yeheskiel Ravena Damanik, Esther Laura Rumahorbo, Albert Adolf Angie MS*  
> **RIGGS: Journal of Artificial Intelligence and Digital Business**, Vol. 5 No. 2 (2026), pp. 22774–22783. DOI: `10.31004/riggs.v5i2.12193`

---

## 📌 Ringkasan Proyek & Metodologi

Aplikasi ini bertujuan untuk mendeteksi sedini mungkin tingkat capaian belajar matematika siswa ke dalam tiga tingkatan klasifikasi:
- 🔴 **Rendah**: Nilai akhir $G3 \in [0, 9]$
- 🟡 **Sedang**: Nilai akhir $G3 \in [10, 14]$
- 🟢 **Tinggi**: Nilai akhir $G3 \in [15, 20]$

### Karakteristik Dataset
- **Sumber**: Versi Kaggle yang bersumber dari UCI Machine Learning Repository (*Student Performance Data Set* - Cortez & Silva, 2008).
- **Jumlah Sampel**: 395 baris data siswa (bagian Matematika, `student-mat.csv`, pemisah `;`).
- **Kualitas Data**: 0 nilai hilang (*missing values*), 0 data duplikat.
- **Pencegahan Kebocoran Data (*Data Leakage Prevention*)**: Nilai akhir `G3` dihapus dari fitur masukan. 32 variabel sisanya (termasuk nilai evaluasi awal `G1` dan `G2`) digunakan sebagai prediktor.

---

## 📊 Ringkasan Hasil Evaluasi Model

| Metrik Evaluasi | Jurnal Acuan (Orange Data Mining) | Decision Tree (Scikit-Learn Kami) | Model Pembanding: MLP Classifier |
|---|:---:|:---:|:---:|
| **Akurasi Testing (20%)** | — | **83,12%** | 55,84% |
| **Akurasi 10-Fold CV (395 Data)** | **87,10%** (344/395 benar) | **85,33% ± 2,89%** | — |
| **Akurasi 10-Fold CV (Sisa 385 Data)** | — | **84,69% ± 5,36%** | 59,95% ± 18,67% |
| **Presisi Rata-rata Makro** | 87,10% | **86,02%** | 72,50% |
| **Recall Rata-rata Makro** | 87,10% | **84,36%** | 43,33% |
| **F1-Score Makro** | 87,10% | **85,14%** | 41,83% |
| **Uji 10 Data Baru Independen** | — | **70,00% (7/10 Benar)** | — |

*Catatan Metodologi*: Kriteria bawaan Orange *"stop when majority reaches 95%"* tidak memiliki padanan langsung di scikit-learn, sehingga diabaikan. Meskipun demikian, hasil model scikit-learn terbukti sangat konsisten dengan temuan jurnal, di mana atribut $G2$ (>87%) dan $G1$ (>3%) menjadi atribut penentu utama tanpa adanya kesalahan ekstrem antara kelas Rendah dan Tinggi.

---

## 📁 Struktur Direktori Proyek

```text
KecerdasanBuatanUTS/
├── app.py                  # Aplikasi utama web Flask
├── train.py                # Script pelatihan model & generasi visualisasi grafik
├── requirements.txt        # Dependensi pustaka Python (versi ter-pin)
├── README.md               # Dokumentasi lengkap & panduan deployment
├── .gitignore              # Mengabaikan .venv/, __pycache__/, .env
├── data/
│   ├── student-mat.csv     # Dataset asli (395 siswa, pemisah ;)
│   └── data_baru.csv       # 10 baris data independen yang disisihkan (random_state=42)
├── model/
│   ├── dt_pipeline.joblib  # Pipeline model Decision Tree
│   ├── mlp_pipeline.joblib # Pipeline model MLP Classifier
│   ├── metrics.joblib      # Metrik evaluasi & classification report
│   ├── feature_meta.joblib # Metadata fitur untuk form input
│   └── feature_names.joblib
├── static/
│   ├── css/
│   │   └── style.css       # Kustomisasi CSS modern
│   └── plots/
│       ├── distribusi_target.png
│       ├── decision_tree.png
│       ├── feature_importance.png
│       ├── confusion_matrix_dt.png
│       ├── confusion_matrix_mlp.png
│       └── model_comparison.png
└── templates/
    ├── base.html           # Layout utama (Bootstrap 5, Navbar, Footer)
    ├── index.html          # Beranda & ringkasan jurnal
    ├── dataset.html        # Eksplorasi dataset & deskripsi variabel
    ├── preprocessing.html  # Dokumentasi before/after preprocessing
    ├── training.html       # Visualisasi pohon & feature importance
    ├── evaluasi.html       # Confusion matrix & classification report
    ├── prediksi.html       # Form input 32 fitur & upload CSV batch
    └── uji_baru.html       # Pengujian 10 data baru independen
```

---

## 💻 Versi Lingkungan & Dependensi

- **Versi Python**: `Python 3.12.3` (kompatibel dengan Python 3.10+)
- **Paket Utama (`requirements.txt`)**:
  - `flask==3.1.3`
  - `pandas==3.0.6`
  - `scikit-learn==1.9.1`
  - `joblib==1.6.0`
  - `matplotlib==3.11.2`
  - `seaborn==0.13.2`

---

## 🚀 Panduan Menjalankan Secara Lokal

1. **Clone repository**:
   ```bash
   git clone https://github.com/ItsKarr007/KecerdasanBuatanUTS.git
   cd KecerdasanBuatanUTS
   ```

2. **Buat dan aktifkan virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependensi**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Latih model dan hasilkan visualisasi (opsional jika sudah ada)**:
   ```bash
   python train.py
   ```

5. **Jalankan server Flask**:
   ```bash
   python app.py
   ```
   Buka peramban di `http://127.0.0.1:5000`.

---

## ☁️ Panduan Lengkap Deployment ke PythonAnywhere (WAJIB)

Target deployment aplikasi ini adalah **PythonAnywhere** (akun gratis). Ikuti langkah-langkah terstruktur di bawah ini. Ganti `USERNAME` dengan username akun PythonAnywhere Anda:

### 1. Buka Bash Console di PythonAnywhere
Masuk ke dasbor PythonAnywhere, buka menu **Consoles** > pilih **Bash**.

### 2. Clone Repository GitHub
Jalankan perintah clone ke direktori home:
```bash
git clone https://github.com/ItsKarr007/KecerdasanBuatanUTS.git
```

### 3. Buat Virtual Environment & Install Dependensi
```bash
mkvirtualenv --python=python3.10 myenv
cd ~/KecerdasanBuatanUTS
pip install -r requirements.txt
```
*(Catatan: Anda dapat menyesuaikan versi Python misalnya `--python=python3.10` atau `--python=python3.11` sesuai versi yang tersedia di akun PythonAnywhere Anda).*

### 4. Latih Model di Server PythonAnywhere
Jalankan script pelatihan agar model `.joblib` dan plot grafik di `.png` dibuat sesuai versi scikit-learn di server:
```bash
python train.py
```

### 5. Konfigurasi Web App
1. Buka tab **Web** pada navbar dashboard PythonAnywhere.
2. Klik tombol **Add a new web app**.
3. Pilih opsi **Manual configuration** (jangan pilih wizard Flask otomatis).
4. Pilih versi Python yang sama persis dengan yang Anda gunakan saat membuat virtualenv (misal Python 3.10).

### 6. Atur Path Direktori dan Virtualenv
Pada tab **Web**:
- **Source code**: `/home/USERNAME/KecerdasanBuatanUTS`
- **Working directory**: `/home/USERNAME/KecerdasanBuatanUTS`
- **Virtualenv**: `/home/USERNAME/.virtualenvs/myenv`

### 7. Konfigurasi Berkas WSGI
Pada tab **Web**, klik tautan berkas **WSGI configuration file** (biasanya bernama `/var/www/USERNAME_pythonanywhere_com_wsgi.py`). Hapus seluruh isinya, lalu ganti dengan kode berikut:
```python
import sys
import os

# Menambahkan direktori proyek ke sys.path
path = "/home/USERNAME/KecerdasanBuatanUTS"
if path not in sys.path:
    sys.path.insert(0, path)

# Memastikan direktori kerja aplikasi benar
os.chdir(path)

# Mengimpor objek Flask bernama 'app' sebagai 'application'
from app import app as application
```
Klik tombol **Save** (Ctrl+S).

### 8. Atur Direktori Static Files
Pada bagian **Static files** di tab **Web**, tambahkan pemetaan berkas statis:
- **URL**: `/static/`
- **Directory**: `/home/USERNAME/KecerdasanBuatanUTS/static`

### 9. Reload Web App
Klik tombol hijau besar bertuliskan **Reload USERNAME.pythonanywhere.com**.
Aplikasi Anda sekarang aktif dan dapat diakses online di:
`https://USERNAME.pythonanywhere.com`

---

## 🔄 Pembaruan / Update Aplikasi

Jika ada perubahan kode di GitHub yang ingin ditarik ke server:
1. Buka **Consoles** > **Bash**:
   ```bash
   cd ~/KecerdasanBuatanUTS
   git pull origin main
   ```
2. Buka tab **Web** dan klik tombol **Reload**.

---

## 🛠️ Panduan Troubleshooting

| Masalah yang Sering Muncul | Penyebab Utama | Solusi |
|---|---|---|
| **Error 500 / Something went wrong** | Kesalahan impor modul atau path file | Buka tab **Web** > lihat bagian **Log files** > klik **Error log**. Cek pesan galat baris terakhir. |
| **ModuleNotFoundError** (misal: `No module named 'flask'`) | Path Virtualenv belum diisi atau salah | Pastikan field **Virtualenv** di tab Web berisi `/home/USERNAME/.virtualenvs/myenv`. |
| **FileNotFoundError** saat membaca data/model | Working directory belum diatur | Pastikan field **Working directory** diisi `/home/USERNAME/KecerdasanBuatanUTS` dan berkas `train.py` sudah dijalankan. |
| **Gambar grafik atau CSS tidak muncul** | Konfigurasi Static files belum dipetakan | Periksa pemetaan URL `/static/` ke `/home/USERNAME/KecerdasanBuatanUTS/static` di tab Web. |
| **Perubahan kode tidak terlihat di browser** | Belum melakukan Reload | Setiap setelah melakukan `git pull` atau edit berkas, wajib klik tombol hijau **Reload** di tab Web. |
| **Aplikasi dinonaktifkan setelah 3 bulan** | Kebijakan akun gratis PythonAnywhere | Masuk ke tab **Web** dan klik tombol **Run until 3 months from today** secara berkala. |

---

## 👨‍💻 Pengembang & Repository

- **Repository**: [https://github.com/ItsKarr007/KecerdasanBuatanUTS](https://github.com/ItsKarr007/KecerdasanBuatanUTS)
- Dibuat untuk memenuhi Tugas Ujian Tengah Semester Kecerdasan Buatan.

