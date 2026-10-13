"""
app.py
Aplikasi Web Klasifikasi Prestasi Akademik Matematika Siswa Menggunakan Decision Tree
Berdasarkan Publikasi Ilmiah RIGGS (2026) Vol. 5 No. 2, pp. 22774-22783.
"""

import os
import io
import joblib
import pandas as pd
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_file,
    session
)
from train import train_models, LABELS_ORDER

# Path absolut berbasis lokasi file app.py untuk kompatibilitas penuh dengan PythonAnywhere
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
PLOTS_DIR = os.path.join(STATIC_DIR, 'plots')

# Inisialisasi Aplikasi Flask
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'student_performance_decision_tree_secret_2026')

# Path berkas artefak
STUDENT_MAT_PATH = os.path.join(DATA_DIR, 'student-mat.csv')
DATA_BARU_PATH = os.path.join(DATA_DIR, 'data_baru.csv')
DT_MODEL_PATH = os.path.join(MODEL_DIR, 'dt_pipeline.joblib')
METRICS_PATH = os.path.join(MODEL_DIR, 'metrics.joblib')
FEATURE_META_PATH = os.path.join(MODEL_DIR, 'feature_meta.joblib')

# Simpan hasil batch prediksi di memori sesi / file sementara
BATCH_RESULTS_CACHE = []


def ensure_models_loaded():
    """Memastikan file model dan metrik sudah tersedia. Jika belum, latih otomatis."""
    if not os.path.exists(DT_MODEL_PATH) or not os.path.exists(METRICS_PATH):
        print("Model belum ditemukan. Menjalankan train_models()...")
        train_models()
    dt_pipeline = joblib.load(DT_MODEL_PATH)
    metrics = joblib.load(METRICS_PATH)
    feature_meta = joblib.load(FEATURE_META_PATH) if os.path.exists(FEATURE_META_PATH) else {}
    return dt_pipeline, metrics, feature_meta


# ==========================================================
# ROUTE 1: BERANDA (INDEX)
# ==========================================================
@app.route('/')
def index():
    try:
        _, metrics, _ = ensure_models_loaded()
        return render_template('index.html', active_page='index', metrics=metrics)
    except Exception as e:
        flash(f"Terjadi kesalahan saat memuat data beranda: {str(e)}", "danger")
        return render_template('index.html', active_page='index', metrics=None)


# ==========================================================
# ROUTE 2: DATASET
# ==========================================================
@app.route('/dataset')
def dataset_view():
    try:
        _, metrics, _ = ensure_models_loaded()
        df = pd.read_csv(STUDENT_MAT_PATH, sep=';')

        # Tambahkan Kategori_G3 untuk preview
        def categorize(g3):
            return 'Rendah' if g3 <= 9 else ('Sedang' if g3 <= 14 else 'Tinggi')

        df_preview = df.copy()
        df_preview['Kategori_G3'] = df_preview['G3'].apply(categorize)

        total_rows, total_cols = df.shape
        target_dist = df_preview['Kategori_G3'].value_counts().to_dict()

        preview_data = df_preview.head(10).to_dict(orient='records')
        preview_columns = list(df_preview.columns)

        return render_template(
            'dataset.html',
            active_page='dataset',
            total_rows=total_rows,
            total_cols=total_cols,
            target_dist=target_dist,
            preview_data=preview_data,
            preview_columns=preview_columns,
            metrics=metrics
        )
    except Exception as e:
        flash(f"Gagal memuat dataset: {str(e)}", "danger")
        return redirect(url_for('index'))


# ==========================================================
# ROUTE 3: PREPROCESSING
# ==========================================================
@app.route('/preprocessing')
def preprocessing_view():
    try:
        _, metrics, _ = ensure_models_loaded()
        return render_template('preprocessing.html', active_page='preprocessing', metrics=metrics)
    except Exception as e:
        flash(f"Gagal memuat halaman preprocessing: {str(e)}", "danger")
        return redirect(url_for('index'))


# ==========================================================
# ROUTE 4: PELATIHAN MODEL & POHON KEPUTUSAN
# ==========================================================
@app.route('/training')
def training_view():
    try:
        _, metrics, _ = ensure_models_loaded()
        return render_template('training.html', active_page='training', metrics=metrics)
    except Exception as e:
        flash(f"Gagal memuat halaman pelatihan: {str(e)}", "danger")
        return redirect(url_for('index'))


# ==========================================================
# ROUTE 5: LATIH ULANG MODEL (RETRAIN)
# ==========================================================
@app.route('/retrain', methods=['POST'])
def retrain_model():
    try:
        train_models()
        flash("Model Decision Tree & MLP berhasil dilatih ulang! Metrik dan visualisasi grafik telah diperbarui.", "success")
    except Exception as e:
        flash(f"Gagal melatih ulang model: {str(e)}", "danger")
    return redirect(url_for('training_view'))


# ==========================================================
# ROUTE 6: EVALUASI MODEL
# ==========================================================
@app.route('/evaluasi')
def evaluasi_view():
    try:
        _, metrics, _ = ensure_models_loaded()
        return render_template('evaluasi.html', active_page='evaluasi', metrics=metrics)
    except Exception as e:
        flash(f"Gagal memuat halaman evaluasi: {str(e)}", "danger")
        return redirect(url_for('index'))


# ==========================================================
# ROUTE 7: PREDIKSI (SINGLE FORM & BATCH CSV)
# ==========================================================
@app.route('/prediksi', methods=['GET', 'POST'])
def prediksi_view():
    global BATCH_RESULTS_CACHE
    dt_pipeline, _, feature_meta = ensure_models_loaded()

    prediction_result = None
    form_values = {}
    batch_results = BATCH_RESULTS_CACHE if request.method == 'GET' else []
    active_tab = 'single'

    if request.method == 'POST':
        action = request.form.get('action')

        # AKSI 1: PREDIKSI SINGLE DARI FORM 32 FITUR
        if action == 'single_predict':
            active_tab = 'single'
            form_values = request.form.to_dict()
            try:
                # Susun dataframe 1 baris dengan tipe data yang tepat
                input_data = {}
                # 17 fitur kategorikal
                cat_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
                            'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
                            'nursery', 'higher', 'internet', 'romantic']
                # 15 fitur numerik
                num_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
                            'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health',
                            'absences', 'G1', 'G2']

                for col in cat_cols:
                    input_data[col] = [str(request.form.get(col, ''))]

                for col in num_cols:
                    input_data[col] = [float(request.form.get(col, 0))]

                input_df = pd.DataFrame(input_data)

                # Jalankan prediksi dengan pipeline
                pred_label = dt_pipeline.predict(input_df)[0]
                pred_proba = dt_pipeline.predict_proba(input_df)[0]
                classes = dt_pipeline.classes_.tolist()

                # Buat dictionary probabilitas terurut
                proba_dict = {cls: float(prob) for cls, prob in zip(classes, pred_proba)}
                for label in LABELS_ORDER:
                    if label not in proba_dict:
                        proba_dict[label] = 0.0

                prediction_result = {
                    'kategori': pred_label,
                    'probabilitas': proba_dict,
                    'highest_prob': float(max(pred_proba))
                }
                flash(f"Prediksi berhasil! Siswa diprediksi berkategori '{pred_label}'.", "success")

            except Exception as e:
                flash(f"Terjadi kesalahan saat memproses data form: {str(e)}", "danger")

        # AKSI 2: PREDIKSI BATCH DARI FILE CSV
        elif action == 'batch_predict':
            active_tab = 'batch'
            if 'csv_file' not in request.files or request.files['csv_file'].filename == '':
                flash("Harap pilih berkas CSV yang valid untuk diunggah.", "warning")
            else:
                file = request.files['csv_file']
                try:
                    # Coba baca dengan delimiter titik koma lalu koma
                    content = file.read().decode('utf-8')
                    try:
                        df_upload = pd.read_csv(io.StringIO(content), sep=';')
                        if len(df_upload.columns) <= 1:
                            df_upload = pd.read_csv(io.StringIO(content), sep=',')
                    except Exception:
                        df_upload = pd.read_csv(io.StringIO(content), sep=',')

                    # Buang G3 jika ada di file upload
                    pred_df = df_upload.copy()
                    if 'G3' in pred_df.columns:
                        pred_df = pred_df.drop(columns=['G3'])
                    if 'Kategori_G3' in pred_df.columns:
                        pred_df = pred_df.drop(columns=['Kategori_G3'])

                    # Jalankan prediksi
                    preds = dt_pipeline.predict(pred_df)
                    probas = dt_pipeline.predict_proba(pred_df)
                    classes = dt_pipeline.classes_.tolist()

                    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

                    df_upload['Prediksi_Kategori'] = preds
                    df_upload['Prob_Rendah'] = [p[class_to_idx.get('Rendah', 0)] for p in probas]
                    df_upload['Prob_Sedang'] = [p[class_to_idx.get('Sedang', 0)] for p in probas]
                    df_upload['Prob_Tinggi'] = [p[class_to_idx.get('Tinggi', 0)] for p in probas]

                    batch_results = df_upload.to_dict(orient='records')
                    BATCH_RESULTS_CACHE = batch_results

                    flash(f"Berhasil memproses {len(df_upload)} baris data siswa!", "success")

                except Exception as e:
                    flash(f"Gagal memproses berkas CSV: {str(e)}", "danger")

    return render_template(
        'prediksi.html',
        active_page='prediksi',
        prediction_result=prediction_result,
        form_values=form_values,
        batch_results=batch_results,
        active_tab=active_tab,
        feature_meta=feature_meta
    )


# ==========================================================
# ROUTE 8: PENGUJIAN 10 DATA BARU INDEPENDEN
# ==========================================================
@app.route('/uji-baru')
def uji_baru_view():
    try:
        dt_pipeline, _, _ = ensure_models_loaded()

        if not os.path.exists(DATA_BARU_PATH):
            flash("Data baru belum ditemukan. Melakukan inisialisasi ulang...", "warning")
            train_models()

        df_baru = pd.read_csv(DATA_BARU_PATH, sep=';')

        # Fitur X untuk prediksi
        X_baru = df_baru.drop(columns=['G3', 'Kategori_G3'])
        y_actual = df_baru['Kategori_G3'].tolist()

        preds = dt_pipeline.predict(X_baru)
        probas = dt_pipeline.predict_proba(X_baru)

        test_rows = []
        correct_count = 0

        for idx, row in df_baru.iterrows():
            pred = preds[idx]
            actual = y_actual[idx]
            is_correct = (pred == actual)
            if is_correct:
                correct_count += 1

            test_rows.append({
                'school': row['school'],
                'sex': row['sex'],
                'age': row['age'],
                'G1': row['G1'],
                'G2': row['G2'],
                'G3': row['G3'],
                'failures': row['failures'],
                'absences': row['absences'],
                'actual_label': actual,
                'predicted_label': pred,
                'highest_prob': float(max(probas[idx])),
                'is_correct': is_correct
            })

        total_cases = len(test_rows)
        accuracy_rate = (correct_count / total_cases) * 100 if total_cases > 0 else 0

        return render_template(
            'uji_baru.html',
            active_page='uji_baru',
            test_rows=test_rows,
            total_cases=total_cases,
            correct_cases=correct_count,
            accuracy_rate=accuracy_rate
        )

    except Exception as e:
        flash(f"Gagal menguji data baru: {str(e)}", "danger")
        return redirect(url_for('index'))


# ==========================================================
# ROUTE 9: DOWNLOAD FILES
# ==========================================================
@app.route('/download/dataset')
def download_raw_dataset():
    """Mengunduh berkas asli student-mat.csv"""
    return send_file(STUDENT_MAT_PATH, as_attachment=True, download_name='student-mat.csv')


@app.route('/download/data-baru')
def download_data_baru():
    """Mengunduh 10 data baru data_baru.csv"""
    return send_file(DATA_BARU_PATH, as_attachment=True, download_name='data_baru.csv')


@app.route('/download/template-csv')
def download_template_csv():
    """Mengunduh contoh template CSV 32 fitur untuk uji batch"""
    df = pd.read_csv(STUDENT_MAT_PATH, sep=';')
    sample_df = df.drop(columns=['G3']).head(5)
    output = io.StringIO()
    sample_df.to_csv(output, index=False, sep=';')
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='template_prediksi_siswa.csv'
    )


@app.route('/download/batch-result')
def download_batch_result():
    """Mengunduh hasil prediksi batch dalam bentuk CSV"""
    global BATCH_RESULTS_CACHE
    if not BATCH_RESULTS_CACHE:
        flash("Tidak ada hasil prediksi batch untuk diunduh.", "warning")
        return redirect(url_for('prediksi_view'))

    df_out = pd.DataFrame(BATCH_RESULTS_CACHE)
    output = io.StringIO()
    df_out.to_csv(output, index=False, sep=';')
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='hasil_prediksi_batch.csv'
    )


@app.route('/download/laporan-pdf')
def download_laporan_pdf():
    """Mengunduh Laporan Resmi Komparasi Hasil Project vs Jurnal (PDF)"""
    pdf_path = os.path.join(BASE_DIR, 'Laporan_Komparasi_Hasil_UTS.pdf')
    if not os.path.exists(pdf_path):
        import subprocess
        subprocess.run(['python', os.path.join(BASE_DIR, 'generate_pdf_report.py')])
    return send_file(pdf_path, as_attachment=True, download_name='Laporan_Komparasi_Hasil_UTS.pdf')


# ==========================================================
# MENJALANKAN APLIKASI SECARA LOKAL
# ==========================================================
if __name__ == '__main__':
    # Pastikan app.run() HANYA berjalan di dalam blok __name__ == '__main__'
    app.run(host='0.0.0.0', port=5000, debug=True)

