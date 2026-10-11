"""
train.py
Script pelatihan model klasifikasi prestasi matematika siswa
berdasarkan jurnal: "Klasifikasi Prestasi Akademik Matematika Siswa Menggunakan Decision Tree pada Dataset Student Performance"
"""

import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
import joblib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Direktori basis absolut
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
PLOTS_DIR = os.path.join(STATIC_DIR, 'plots')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, 'student-mat.csv')
DATA_BARU_PATH = os.path.join(DATA_DIR, 'data_baru.csv')

LABELS_ORDER = ['Rendah', 'Sedang', 'Tinggi']


def load_and_preprocess_data():
    """Membaca dataset asli student-mat.csv dan membuat target Kategori_G3."""
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Dataset tidak ditemukan di {CSV_PATH}")

    df = pd.read_csv(CSV_PATH, sep=';')

    # Pemetaan G3 ke Kategori_G3:
    # 0-9: Rendah, 10-14: Sedang, 15-20: Tinggi
    def categorize_g3(score):
        if score <= 9:
            return 'Rendah'
        elif score <= 14:
            return 'Sedang'
        else:
            return 'Tinggi'

    df['Kategori_G3'] = df['G3'].apply(categorize_g3)
    return df


def split_data(df):
    """
    Menyisihkan 10 baris acak (random_state=42) sebagai 'data baru'
    yang TIDAK dipakai untuk training maupun testing.
    Sisa data dibagi 80:20 (stratify=Kategori_G3, random_state=42).
    """
    data_baru = df.sample(n=10, random_state=42)
    sisa_df = df.drop(data_baru.index).copy()

    # Simpan data baru ke data/data_baru.csv
    data_baru.to_csv(DATA_BARU_PATH, index=False, sep=';')
    print(f"[1/6] Data baru (10 baris) disimpan ke: {DATA_BARU_PATH}")

    # Pisahkan fitur dan target dari sisa data
    # Buang G3 untuk mencegah data leakage
    X = sisa_df.drop(columns=['G3', 'Kategori_G3'])
    y = sisa_df['Kategori_G3']

    # Train-test split 80:20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[2/6] Split data: Train={len(X_train)}, Test={len(X_test)}, Baru={len(data_baru)}")

    return sisa_df, data_baru, X_train, X_test, y_train, y_test, X, y


def generate_plots(df, dt_pipeline, preprocessor, feature_names_out,
                   y_test, y_pred_dt, y_pred_mlp, cv_dt_mean, cv_mlp_mean, test_acc_dt, test_acc_mlp):
    """Menghasilkan semua visualisasi grafik dan menyimpannya di static/plots/"""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # 1. Distribusi Kelas Target
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    target_counts = df['Kategori_G3'].value_counts()[LABELS_ORDER]
    colors = ['#dc3545', '#ffc107', '#198754']

    bars = axes[0].bar(target_counts.index, target_counts.values, color=colors, edgecolor='black', alpha=0.85)
    axes[0].set_title('Distribusi Jumlah Siswa per Kategori Prestasi (G3)', fontsize=12, fontweight='bold', pad=12)
    axes[0].set_ylabel('Jumlah Siswa', fontsize=11)
    axes[0].set_xlabel('Kategori Prestasi', fontsize=11)
    for bar in bars:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, yval + 3, f"{yval} ({yval/len(df)*100:.1f}%)",
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[0].set_ylim(0, max(target_counts.values) + 25)

    axes[1].pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=140, explode=[0.05, 0.02, 0.05],
                wedgeprops=dict(width=0.6, edgecolor='white', linewidth=2),
                textprops={'fontsize': 11, 'fontweight': 'bold'})
    axes[1].set_title('Proporsi Kategori Prestasi Siswa', fontsize=12, fontweight='bold', pad=12)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'distribusi_target.png'), dpi=180)
    plt.close()
    print("Grafik disimpan: distribusi_target.png")

    # 2. Visualisasi Pohon Keputusan
    dt_model = dt_pipeline.named_steps['classifier']
    fig, ax = plt.subplots(figsize=(22, 11))
    plot_tree(
        dt_model,
        feature_names=feature_names_out,
        class_names=LABELS_ORDER,
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
        precision=2
    )
    ax.set_title('Visualisasi Struktur Decision Tree (max_depth=5, min_samples_leaf=5)',
                 fontsize=15, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'decision_tree.png'), dpi=200)
    plt.close()
    print("Grafik disimpan: decision_tree.png")

    # 3. Feature Importance Decision Tree
    importances = dt_model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names_out).sort_values(ascending=False)
    top_features = feat_imp[feat_imp > 0].head(15)

    plt.figure(figsize=(10, 5.5))
    sns.barplot(x=top_features.values, y=top_features.index, palette='crest')
    plt.title('Top Feature Importances (Decision Tree)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Gini Importance (Nilai Kontribusi)', fontsize=11)
    plt.ylabel('Fitur', fontsize=11)
    for index, value in enumerate(top_features.values):
        plt.text(value + 0.005, index, f"{value:.4f}", va='center', fontsize=9, fontweight='bold')
    plt.xlim(0, max(top_features.values) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance.png'), dpi=180)
    plt.close()
    print("Grafik disimpan: feature_importance.png")

    # 4. Confusion Matrix Decision Tree Heatmap
    cm_dt = confusion_matrix(y_test, y_pred_dt, labels=LABELS_ORDER)
    plt.figure(figsize=(6.5, 5))
    sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABELS_ORDER, yticklabels=LABELS_ORDER,
                cbar=False, annot_kws={'size': 13, 'weight': 'bold'})
    plt.title('Confusion Matrix: Decision Tree (Data Testing 20%)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Prediksi Model', fontsize=11, fontweight='bold')
    plt.ylabel('Label Aktual', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix_dt.png'), dpi=180)
    plt.close()
    print("Grafik disimpan: confusion_matrix_dt.png")

    # 5. Confusion Matrix MLP Heatmap
    cm_mlp = confusion_matrix(y_test, y_pred_mlp, labels=LABELS_ORDER)
    plt.figure(figsize=(6.5, 5))
    sns.heatmap(cm_mlp, annot=True, fmt='d', cmap='Oranges',
                xticklabels=LABELS_ORDER, yticklabels=LABELS_ORDER,
                cbar=False, annot_kws={'size': 13, 'weight': 'bold'})
    plt.title('Confusion Matrix: MLP Classifier (Data Testing 20%)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Prediksi Model', fontsize=11, fontweight='bold')
    plt.ylabel('Label Aktual', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix_mlp.png'), dpi=180)
    plt.close()
    print("Grafik disimpan: confusion_matrix_mlp.png")

    # 6. Grafik Perbandingan Model
    plt.figure(figsize=(8.5, 4.8))
    categories = ['DT (Testing)', 'DT (10-Fold CV)', 'MLP (Testing)', 'MLP (10-Fold CV)', 'Jurnal Orange (CV)']
    accuracies = [test_acc_dt * 100, cv_dt_mean * 100, test_acc_mlp * 100, cv_mlp_mean * 100, 87.1]
    bar_colors = ['#0d6efd', '#6ea8fe', '#fd7e14', '#ffc078', '#198754']

    bars = plt.bar(categories, accuracies, color=bar_colors, edgecolor='black', width=0.52)
    plt.title('Perbandingan Akurasi Model dengan Hasil Jurnal', fontsize=13, fontweight='bold', pad=12)
    plt.ylabel('Akurasi (%)', fontsize=11)
    plt.ylim(60, 100)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.2f}%",
                 ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'model_comparison.png'), dpi=180)
    plt.close()
    print("Grafik disimpan: model_comparison.png")


def train_models():
    """Fungsi utama pelatihan model dan penyimpanan artefak."""
    print("=== MEMULAI PROSES PELATIHAN MODEL ===")
    df = load_and_preprocess_data()
    print(f"Total dataset dimuat: {len(df)} baris, {df.shape[1]} kolom")

    missing_sum = int(df.isnull().sum().sum())
    dup_sum = int(df.duplicated().sum())

    sisa_df, data_baru, X_train, X_test, y_train, y_test, X, y = split_data(df)

    categorical_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    preprocessor_dt = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
            ('num', 'passthrough', numeric_cols)
        ]
    )

    preprocessor_mlp = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
            ('num', StandardScaler(), numeric_cols)
        ]
    )

    dt_pipeline = Pipeline([
        ('preprocessor', preprocessor_dt),
        ('classifier', DecisionTreeClassifier(
            criterion='gini',
            max_depth=5,
            min_samples_leaf=5,
            min_samples_split=5,
            random_state=42
        ))
    ])

    mlp_pipeline = Pipeline([
        ('preprocessor', preprocessor_mlp),
        ('classifier', MLPClassifier(
            hidden_layer_sizes=(32, 16),
            max_iter=150,
            early_stopping=True,
            n_iter_no_change=8,
            random_state=42
        ))
    ])

    print("[3/6] Menghitung 10-Fold Stratified Cross Validation...")
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores_dt = cross_val_score(dt_pipeline, X, y, cv=cv, scoring='accuracy')

    # 10-fold CV DT pada 395 data asli (sebagai pembanding jurnal persis)
    X_all = df.drop(columns=['G3', 'Kategori_G3'])
    y_all = df['Kategori_G3']
    cv_scores_dt_all = cross_val_score(dt_pipeline, X_all, y_all, cv=cv, scoring='accuracy')

    cv_scores_mlp = cross_val_score(mlp_pipeline, X, y, cv=cv, scoring='accuracy')

    print(f"  - CV DT (385 data): {cv_scores_dt.mean():.4f} (+/- {cv_scores_dt.std():.4f})")
    print(f"  - CV DT (395 data penuh): {cv_scores_dt_all.mean():.4f} (+/- {cv_scores_dt_all.std():.4f})")
    print(f"  - CV MLP (385 data): {cv_scores_mlp.mean():.4f} (+/- {cv_scores_mlp.std():.4f})")

    print("[4/6] Melatih model pada training set (80%)...")
    dt_pipeline.fit(X_train, y_train)
    mlp_pipeline.fit(X_train, y_train)

    print("[5/6] Evaluasi pada testing set (20%)...")
    y_pred_dt = dt_pipeline.predict(X_test)
    y_pred_mlp = mlp_pipeline.predict(X_test)

    test_acc_dt = float(accuracy_score(y_test, y_pred_dt))
    prec_macro_dt = float(precision_score(y_test, y_pred_dt, labels=LABELS_ORDER, average='macro', zero_division=0))
    rec_macro_dt = float(recall_score(y_test, y_pred_dt, labels=LABELS_ORDER, average='macro', zero_division=0))
    f1_macro_dt = float(f1_score(y_test, y_pred_dt, labels=LABELS_ORDER, average='macro', zero_division=0))

    prec_per_class_dt = precision_score(y_test, y_pred_dt, labels=LABELS_ORDER, average=None, zero_division=0).tolist()
    rec_per_class_dt = recall_score(y_test, y_pred_dt, labels=LABELS_ORDER, average=None, zero_division=0).tolist()
    f1_per_class_dt = f1_score(y_test, y_pred_dt, labels=LABELS_ORDER, average=None, zero_division=0).tolist()

    report_dt_dict = classification_report(y_test, y_pred_dt, target_names=LABELS_ORDER, output_dict=True, zero_division=0)
    cm_dt = confusion_matrix(y_test, y_pred_dt, labels=LABELS_ORDER).tolist()

    test_acc_mlp = float(accuracy_score(y_test, y_pred_mlp))
    prec_macro_mlp = float(precision_score(y_test, y_pred_mlp, labels=LABELS_ORDER, average='macro', zero_division=0))
    rec_macro_mlp = float(recall_score(y_test, y_pred_mlp, labels=LABELS_ORDER, average='macro', zero_division=0))
    f1_macro_mlp = float(f1_score(y_test, y_pred_mlp, labels=LABELS_ORDER, average='macro', zero_division=0))
    cm_mlp = confusion_matrix(y_test, y_pred_mlp, labels=LABELS_ORDER).tolist()
    report_mlp_dict = classification_report(y_test, y_pred_mlp, target_names=LABELS_ORDER, output_dict=True, zero_division=0)

    preprocessor = dt_pipeline.named_steps['preprocessor']
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist()
    all_feature_names_out = cat_feature_names + numeric_cols

    dt_model = dt_pipeline.named_steps['classifier']
    importances = dt_model.feature_importances_
    feat_imp_df = pd.DataFrame({
        'feature': all_feature_names_out,
        'importance': importances
    }).sort_values('importance', ascending=False)
    top_15_features = feat_imp_df[feat_imp_df['importance'] > 0].head(15).to_dict(orient='records')

    print("[6/6] Menghasilkan grafik visualisasi...")
    generate_plots(
        df=df,
        dt_pipeline=dt_pipeline,
        preprocessor=preprocessor,
        feature_names_out=all_feature_names_out,
        y_test=y_test,
        y_pred_dt=y_pred_dt,
        y_pred_mlp=y_pred_mlp,
        cv_dt_mean=float(cv_scores_dt.mean()),
        cv_mlp_mean=float(cv_scores_mlp.mean()),
        test_acc_dt=test_acc_dt,
        test_acc_mlp=test_acc_mlp
    )

    split_info = {
        'total_data': len(df),
        'data_baru_count': len(data_baru),
        'data_baru_dist': data_baru['Kategori_G3'].value_counts().to_dict(),
        'sisa_data_count': len(sisa_df),
        'train_count': len(X_train),
        'train_dist': y_train.value_counts().to_dict(),
        'test_count': len(X_test),
        'test_dist': y_test.value_counts().to_dict(),
        'missing_values': missing_sum,
        'duplicates': dup_sum
    }

    metrics = {
        'split_info': split_info,
        'labels_order': LABELS_ORDER,
        'decision_tree': {
            'accuracy': test_acc_dt,
            'precision_macro': prec_macro_dt,
            'recall_macro': rec_macro_dt,
            'f1_macro': f1_macro_dt,
            'prec_per_class': dict(zip(LABELS_ORDER, prec_per_class_dt)),
            'rec_per_class': dict(zip(LABELS_ORDER, rec_per_class_dt)),
            'f1_per_class': dict(zip(LABELS_ORDER, f1_per_class_dt)),
            'confusion_matrix': cm_dt,
            'classification_report': report_dt_dict,
            'cv_10fold_mean': float(cv_scores_dt.mean()),
            'cv_10fold_std': float(cv_scores_dt.std()),
            'cv_10fold_scores': cv_scores_dt.tolist(),
            'cv_10fold_all_mean': float(cv_scores_dt_all.mean()),
            'cv_10fold_all_std': float(cv_scores_dt_all.std()),
            'top_features': top_15_features
        },
        'mlp': {
            'accuracy': test_acc_mlp,
            'precision_macro': prec_macro_mlp,
            'recall_macro': rec_macro_mlp,
            'f1_macro': f1_macro_mlp,
            'confusion_matrix': cm_mlp,
            'classification_report': report_mlp_dict,
            'cv_10fold_mean': float(cv_scores_mlp.mean()),
            'cv_10fold_std': float(cv_scores_mlp.std()),
            'cv_10fold_scores': cv_scores_mlp.tolist()
        },
        'jurnal_reference': {
            'accuracy': 0.871,
            'precision': 0.871,
            'recall': 0.871,
            'f1_score': 0.871,
            'auc': 0.934,
            'mcc': 0.791,
            'correct_cases': 344,
            'total_cases': 395
        }
    }

    feature_meta = {}
    for col in categorical_cols:
        feature_meta[col] = {
            'type': 'categorical',
            'options': sorted(df[col].unique().tolist()),
            'default': str(df[col].mode()[0])
        }
    for col in numeric_cols:
        feature_meta[col] = {
            'type': 'numeric',
            'min': int(df[col].min()),
            'max': int(df[col].max()),
            'default': int(round(df[col].median())),
            'mean': float(df[col].mean())
        }

    joblib.dump(dt_pipeline, os.path.join(MODEL_DIR, 'dt_pipeline.joblib'))
    joblib.dump(mlp_pipeline, os.path.join(MODEL_DIR, 'mlp_pipeline.joblib'))
    joblib.dump(metrics, os.path.join(MODEL_DIR, 'metrics.joblib'))
    joblib.dump(feature_meta, os.path.join(MODEL_DIR, 'feature_meta.joblib'))
    joblib.dump(all_feature_names_out, os.path.join(MODEL_DIR, 'feature_names.joblib'))

    print("=== MODEL DAN METRIK BERHASIL DISIMPAN KE FOLDER model/ ===")
    return metrics


if __name__ == '__main__':
    train_models()
