
import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin

# =====================================================
# KONFIGURASI PATH
# =====================================================
ARTIFACT_DIR = r"C:\Users\USER\Documents\Codingan Skripsi\ann_improved"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "best_ann_model_tuned.pt")
PREPROCESSOR_PATH = os.path.join(ARTIFACT_DIR, "preprocessor.pkl")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.pkl")
FEATURE_COLUMNS_PATH = os.path.join(ARTIFACT_DIR, "feature_columns.pkl")
RAW_FEATURE_COLUMNS_PATH = os.path.join(ARTIFACT_DIR, "raw_feature_columns.pkl")

CLEAN_TEST_PATH = os.path.join(ARTIFACT_DIR, "clean_test_best_config.csv")
TEST_PREDICTIONS_PATH = os.path.join(ARTIFACT_DIR, "test_predictions_best_tuned.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# KONSTANTA FITUR
# =====================================================
NUMERIC_FEATURES = [
    "log_harga", "is_non_brand", "berat_kg", "tebal_mm", "jumlah_lembar", "ukuran_cm",
    "panjang_meter", "yard", "box_panjang_cm", "box_lebar_cm", "box_tinggi_cm",
    "log_volume_box_cm3", "lebar_cm_roll", "panjang_meter_roll", "lebar_pair_yard",
    "panjang_yard_pair", "jumlah_dimensi_x", "panjang_nama", "jumlah_token", "is_bflute",
    "is_cflute", "is_foam", "is_kraft", "has_box_dimension", "has_roll_dimension", "has_yard_dimension",
]

CATEGORICAL_FEATURES = [
    "kategori_produk", "jenis_produk", "merek", "warna", "material", "bentuk_kemasan", "family_template",
]

TEXT_FEATURE = "nama_produk_normalized"
RAW_FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TEXT_FEATURE]

# =====================================================
# CUSTOM TRANSFORMER
# Wajib ada supaya joblib.load(preprocessor.pkl) bisa membaca class ini.
# =====================================================
class SafeTfidfSVDVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=3000, min_df=3, ngram_range=(1, 2), n_components=32, random_state=42):
        self.max_features = max_features
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.n_components = n_components
        self.random_state = random_state

    def _to_series(self, X):
        if isinstance(X, pd.DataFrame):
            s = X.iloc[:, 0]
        elif isinstance(X, pd.Series):
            s = X
        else:
            arr = np.asarray(X)
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr[:, 0]
            s = pd.Series(arr)
        return s.fillna("").astype(str)

    def fit(self, X, y=None):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        texts = self._to_series(X)
        self.vectorizer_ = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            sublinear_tf=True
        )
        self.use_zero_output_ = False
        self.svd_ = None
        self.output_dim_ = 1

        try:
            X_tfidf = self.vectorizer_.fit_transform(texts)
        except ValueError:
            self.use_zero_output_ = True
            return self

        n_features = X_tfidf.shape[1]

        if n_features <= 0:
            self.use_zero_output_ = True
            return self

        n_components = min(self.n_components, n_features - 1)

        if n_components >= 1:
            self.svd_ = TruncatedSVD(n_components=n_components, random_state=self.random_state)
            self.svd_.fit(X_tfidf)
            self.output_dim_ = n_components
        else:
            self.output_dim_ = n_features

        return self

    def transform(self, X):
        texts = self._to_series(X)

        if getattr(self, "use_zero_output_", False):
            return np.zeros((len(texts), 1), dtype=np.float32)

        X_tfidf = self.vectorizer_.transform(texts)

        if getattr(self, "svd_", None) is not None:
            return np.asarray(self.svd_.transform(X_tfidf), dtype=np.float32)

        return np.asarray(X_tfidf.toarray(), dtype=np.float32)

# =====================================================
# HELPER
# =====================================================
def standardize_colname(col: str) -> str:
    col = str(col).strip().lower()
    col = re.sub(r"[^a-z0-9]+", "_", col)
    col = re.sub(r"_+", "_", col).strip("_")
    return col

def find_column_by_standardized_name(df: pd.DataFrame, candidates):
    candidate_set = set(candidates)

    for col in df.columns:
        if standardize_colname(col) in candidate_set:
            return col

    return None

def add_indicator_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in [
        "box_panjang_cm", "box_lebar_cm", "box_tinggi_cm",
        "lebar_cm_roll", "panjang_meter_roll", "yard",
        "lebar_pair_yard", "panjang_yard_pair"
    ]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["has_box_dimension"] = (
        (df[["box_panjang_cm", "box_lebar_cm", "box_tinggi_cm"]].fillna(0) > 0).any(axis=1)
    ).astype(int)

    df["has_roll_dimension"] = (
        (df[["lebar_cm_roll", "panjang_meter_roll"]].fillna(0) > 0).any(axis=1)
    ).astype(int)

    df["has_yard_dimension"] = (
        (df[["yard", "lebar_pair_yard", "panjang_yard_pair"]].fillna(0) > 0).any(axis=1)
    ).astype(int)

    return df

def prepare_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [standardize_colname(c) for c in df.columns]
    df = add_indicator_features(df)

    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = "lainnya"
        df[col] = df[col].fillna("lainnya").astype(str).str.strip()
        df.loc[df[col] == "", col] = "lainnya"

    if TEXT_FEATURE not in df.columns:
        df[TEXT_FEATURE] = ""

    df[TEXT_FEATURE] = df[TEXT_FEATURE].fillna("").astype(str)

    return df[RAW_FEATURE_COLUMNS].copy()

def clip_rating_predictions(preds: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(preds, dtype=float), 1.0, 5.0)

def round_to_half(preds: np.ndarray) -> np.ndarray:
    preds = clip_rating_predictions(preds)
    return np.round(preds * 2.0) / 2.0

def read_uploaded_file(uploaded_file):
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)

def get_prediction_columns(df: pd.DataFrame):
    """
    Mendukung dua format:
    1. Output Streamlit:
       prediksi_rating_raw, prediksi_rating_clipped, prediksi_rating_halfstep
    2. Output training:
       predicted_review_raw, predicted_review_clipped, predicted_review_halfstep
    """
    raw_col = find_column_by_standardized_name(
        df,
        ["prediksi_rating_raw", "predicted_review_raw"]
    )
    clipped_col = find_column_by_standardized_name(
        df,
        ["prediksi_rating_clipped", "predicted_review_clipped"]
    )
    halfstep_col = find_column_by_standardized_name(
        df,
        ["prediksi_rating_halfstep", "predicted_review_halfstep"]
    )

    return raw_col, clipped_col, halfstep_col

def calculate_prediction_metrics(result: pd.DataFrame):
    """
    Menghitung metrik evaluasi prediksi rating.

    Metrik:
    - Akurasi Tepat
    - Akurasi ±0.5
    - RMSE
    - MAE
    """
    actual_col = find_column_by_standardized_name(
        result,
        ["review", "rating", "actual_review", "nilai_review"]
    )

    if actual_col is None:
        return None

    raw_col, clipped_col, halfstep_col = get_prediction_columns(result)

    if clipped_col is None or halfstep_col is None:
        return None

    y_true = pd.to_numeric(result[actual_col], errors="coerce")
    y_pred_clipped = pd.to_numeric(result[clipped_col], errors="coerce")
    y_pred_halfstep = pd.to_numeric(result[halfstep_col], errors="coerce")

    valid_mask = y_true.notna() & y_pred_clipped.notna() & y_pred_halfstep.notna()

    if valid_mask.sum() == 0:
        return None

    y_true = y_true[valid_mask].to_numpy(dtype=float)
    y_pred_clipped = y_pred_clipped[valid_mask].to_numpy(dtype=float)
    y_pred_halfstep = y_pred_halfstep[valid_mask].to_numpy(dtype=float)

    exact_accuracy = np.mean(np.isclose(y_true, y_pred_halfstep)) * 100
    tolerance_accuracy = np.mean(np.abs(y_true - y_pred_halfstep) <= 0.5) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred_clipped) ** 2))
    mae = np.mean(np.abs(y_true - y_pred_clipped))

    return {
        "actual_col": actual_col,
        "clipped_col": clipped_col,
        "halfstep_col": halfstep_col,
        "n_data": int(len(y_true)),
        "exact_accuracy": float(exact_accuracy),
        "tolerance_accuracy": float(tolerance_accuracy),
        "rmse": float(rmse),
        "mae": float(mae),
    }

def display_metrics(metrics: dict, title="Performa Model"):
    st.subheader(title)

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        label="Akurasi Tepat",
        value=f"{metrics['exact_accuracy']:.2f}%"
    )

    m2.metric(
        label="Akurasi ±0.5",
        value=f"{metrics['tolerance_accuracy']:.2f}%"
    )

    m3.metric(
        label="RMSE",
        value=f"{metrics['rmse']:.4f}"
    )

    m4.metric(
        label="MAE",
        value=f"{metrics['mae']:.4f}"
    )

    st.success(
        f"Evaluasi dihitung berdasarkan kolom `{metrics['actual_col']}` "
        f"pada {metrics['n_data']} data valid. "
        f"Akurasi ±0.5 berarti prediksi dianggap benar jika selisihnya maksimal 0.5 dari rating asli."
    )

def display_prediction_distribution(result: pd.DataFrame):
    raw_col, clipped_col, halfstep_col = get_prediction_columns(result)

    if raw_col is None and clipped_col is None and halfstep_col is None:
        return

    st.subheader("Distribusi Hasil Prediksi")

    fig, ax = plt.subplots(figsize=(10, 4))
    bins = np.arange(1, 5.5, 0.5)

    if raw_col is not None:
        ax.hist(result[raw_col], bins=bins, alpha=0.5, label="Raw")

    if clipped_col is not None:
        ax.hist(result[clipped_col], bins=bins, alpha=0.5, label="Clipped")

    if halfstep_col is not None:
        ax.hist(result[halfstep_col], bins=bins, alpha=0.5, label="Half-Step")

    ax.set_title("Distribusi Rating Prediksi")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Jumlah Produk")
    ax.set_xticks(bins)
    ax.legend()

    st.pyplot(fig)

# =====================================================
# MODEL
# =====================================================
def make_norm_layer(norm_name: str, dim: int):
    if norm_name == "batchnorm":
        return nn.BatchNorm1d(dim)
    if norm_name == "layernorm":
        return nn.LayerNorm(dim)
    if norm_name == "none":
        return nn.Identity()
    raise ValueError(f"norm_name tidak dikenali: {norm_name}")

def make_activation(activation_name: str):
    if activation_name == "relu":
        return nn.ReLU()
    if activation_name == "silu":
        return nn.SiLU()
    raise ValueError(f"activation_name tidak dikenali: {activation_name}")

class ANNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout, norm_name="layernorm", activation_name="silu"):
        super().__init__()

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(make_norm_layer(norm_name, hidden_dim))
            layers.append(make_activation(activation_name))

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# =====================================================
# LOAD ARTIFACTS
# =====================================================
@st.cache_resource
def load_artifacts():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    y_scaler = joblib.load(SCALER_PATH)

    if os.path.exists(FEATURE_COLUMNS_PATH):
        feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    else:
        feature_columns = []

    input_dim = int(checkpoint["input_dim"])
    config = checkpoint["model_config"]

    model = ANNRegressor(
        input_dim=input_dim,
        hidden_layers=config["hidden_layers"],
        dropout=config["dropout"],
        norm_name=config["norm_name"],
        activation_name=config["activation_name"]
    )

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(DEVICE).eval()

    return {
        "model": model,
        "preprocessor": preprocessor,
        "y_scaler": y_scaler,
        "feature_columns": feature_columns,
        "checkpoint": checkpoint,
        "config": config,
        "input_dim": input_dim,
    }

def predict_rating(df_raw: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    model = artifacts["model"]
    preprocessor = artifacts["preprocessor"]
    y_scaler = artifacts["y_scaler"]
    input_dim = artifacts["input_dim"]

    df_input = prepare_input_dataframe(df_raw)
    x_processed = preprocessor.transform(df_input).astype(np.float32)

    if x_processed.shape[1] != input_dim:
        raise ValueError(
            f"Jumlah fitur hasil preprocessing tidak sesuai. "
            f"Model membutuhkan {input_dim}, tetapi hasil preprocessor {x_processed.shape[1]}."
        )

    x_tensor = torch.from_numpy(x_processed).to(DEVICE)

    with torch.no_grad():
        pred_scaled = model(x_tensor).cpu().numpy().reshape(-1, 1)

    pred_rating = y_scaler.inverse_transform(pred_scaled).ravel()

    result = df_raw.copy()
    result["prediksi_rating_raw"] = pred_rating
    result["prediksi_rating_clipped"] = clip_rating_predictions(pred_rating)
    result["prediksi_rating_halfstep"] = round_to_half(pred_rating)

    return result

def process_test_dataframe(df_test: pd.DataFrame, artifacts: dict):
    """
    Jika file sudah berisi hasil prediksi test dari training
    seperti test_predictions_best_tuned.csv, maka tidak perlu prediksi ulang.

    Jika file masih berupa clean_test_best_config.csv,
    maka model akan memprediksi ulang lalu metrik dihitung.
    """
    raw_col, clipped_col, halfstep_col = get_prediction_columns(df_test)

    already_has_predictions = clipped_col is not None and halfstep_col is not None

    if already_has_predictions:
        result = df_test.copy()
        mode = "File sudah berisi hasil prediksi. Sistem menghitung ulang metrik dari kolom prediksi yang ada."
    else:
        result = predict_rating(df_test, artifacts)
        mode = "File berupa data test mentah. Sistem menjalankan prediksi ulang menggunakan model ANN."

    metrics = calculate_prediction_metrics(result)
    return result, metrics, mode

# =====================================================
# STREAMLIT UI
# =====================================================
st.set_page_config(page_title="Prediksi Rating ANN", layout="wide")

st.markdown("""
<style>
.stButton>button {
    background-color: #fff;
    color: #e07a5f;
    border: 2px solid #e07a5f;
    border-radius: 10px;
    padding: 0.4em 1em;
    font-weight: bold;
}
.stButton>button:hover {
    background-color: #e07a5f;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("Menu")
menu = st.sidebar.radio(
    "Pilih halaman",
    ["Home", "Uji Data Test", "Prediksi CSV", "Template Input", "Info Model"]
)

try:
    artifacts = load_artifacts()
    st.sidebar.success("Artifact berhasil dimuat")
    st.sidebar.write("Device:", DEVICE)
    st.sidebar.write("Input dim:", artifacts["input_dim"])
except Exception as e:
    st.sidebar.error("Artifact gagal dimuat")
    st.sidebar.code(str(e))
    st.stop()

# =====================================================
# MENU STREAMLIT
# =====================================================
if menu == "Home":
    st.title("Prediksi Rating Produk E-Commerce Menggunakan ANN")

    st.write(
        "Website ini menggunakan model ANN hasil training. "
        "Data mentah diproses terlebih dahulu menggunakan preprocessor.pkl, "
        "lalu hasil fiturnya dimasukkan ke model ANN."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Input Model", artifacts["input_dim"])
    c2.metric("Best Val RMSE", artifacts["checkpoint"].get("best_val_rmse", "-"))
    c3.metric("Best Val MAE", artifacts["checkpoint"].get("best_val_mae", "-"))

    st.markdown("---")
    st.subheader("Cara Menguji Data Test")

    st.write("""
    Gunakan menu **Uji Data Test** untuk memasukkan data test dari hasil training.

    File yang bisa digunakan:

    1. `clean_test_best_config.csv`  
       Berisi data test bersih. Sistem akan menjalankan prediksi ulang.

    2. `test_predictions_best_tuned.csv`  
       Berisi hasil prediksi data test yang sudah dibuat saat training. Sistem hanya menghitung ulang metrik.

    File tersebut biasanya ada di folder:
    """)

    st.code(ARTIFACT_DIR)

elif menu == "Uji Data Test":
    st.title("Uji Data Test Model")

    st.write(
        "Menu ini digunakan untuk menguji model menggunakan data test yang benar. "
        "Disarankan menggunakan `clean_test_best_config.csv` atau `test_predictions_best_tuned.csv` "
        "dari folder output training."
    )

    sumber_data = st.radio(
        "Pilih sumber data test",
        ["Upload file test", "Ambil otomatis dari folder ann_improved"],
        horizontal=True
    )

    df_test = None
    sumber_label = None

    if sumber_data == "Upload file test":
        uploaded_test = st.file_uploader(
            "Upload clean_test_best_config.csv atau test_predictions_best_tuned.csv",
            type=["csv", "xlsx"],
            key="upload_test"
        )

        if uploaded_test is not None:
            df_test = read_uploaded_file(uploaded_test)
            sumber_label = uploaded_test.name

    else:
        pilihan_file = st.selectbox(
            "Pilih file test dari folder artifact",
            [
                "clean_test_best_config.csv",
                "test_predictions_best_tuned.csv"
            ]
        )

        local_path = CLEAN_TEST_PATH if pilihan_file == "clean_test_best_config.csv" else TEST_PREDICTIONS_PATH

        st.code(local_path)

        if os.path.exists(local_path):
            df_test = pd.read_csv(local_path)
            sumber_label = local_path
        else:
            st.error(
                f"File tidak ditemukan: {local_path}. "
                "Pastikan training sudah selesai dan file output test sudah tersimpan."
            )

    if df_test is not None:
        st.subheader("Preview Data Test")
        st.caption(f"Sumber data: {sumber_label}")
        st.dataframe(df_test.head(20), use_container_width=True)

        if st.button("UJI MODEL DENGAN DATA TEST"):
            try:
                result, metrics, mode_message = process_test_dataframe(df_test, artifacts)

                st.info(mode_message)

                if metrics is not None:
                    display_metrics(metrics, title="Performa Model pada Data Test")
                else:
                    st.warning(
                        "Metrik tidak dapat dihitung karena file tidak memiliki kolom rating asli "
                        "atau kolom prediksi yang sesuai."
                    )

                st.subheader("Hasil Data Test")
                st.dataframe(result.head(50), use_container_width=True)

                csv_data = result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Hasil Uji Data Test",
                    csv_data,
                    "hasil_uji_data_test.csv",
                    "text/csv"
                )

                display_prediction_distribution(result)

            except Exception as e:
                st.error("Terjadi error saat menguji data test.")
                st.code(str(e))

elif menu == "Prediksi CSV":
    st.title("Prediksi dari File CSV/XLSX")

    st.write(
        "Upload file data produk mentah. "
        "Kolom akan distandardisasi otomatis mengikuti format training."
    )

    uploaded_file = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx"], key="upload_prediksi")

    if uploaded_file is not None:
        df = read_uploaded_file(uploaded_file)

        st.subheader("Preview Data Upload")
        st.dataframe(df.head(20), use_container_width=True)

        if st.button("PROSES PREDIKSI"):
            try:
                result = predict_rating(df, artifacts)

                metrics = calculate_prediction_metrics(result)

                if metrics is not None:
                    display_metrics(metrics, title="Performa Model pada Data Upload")
                else:
                    st.warning(
                        "Metrik evaluasi tidak dapat dihitung karena file yang diupload tidak memiliki kolom rating asli "
                        "seperti Review, rating, actual_review, atau nilai_review. "
                        "Sistem tetap menampilkan hasil prediksi."
                    )

                st.subheader("Hasil Prediksi")
                st.dataframe(result.head(50), use_container_width=True)

                csv_data = result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Hasil Prediksi",
                    csv_data,
                    "hasil_prediksi_rating.csv",
                    "text/csv"
                )

                display_prediction_distribution(result)

            except Exception as e:
                st.error("Terjadi error saat memproses prediksi.")
                st.code(str(e))

elif menu == "Template Input":
    st.title("Template Input CSV")

    template = pd.DataFrame(columns=RAW_FEATURE_COLUMNS)
    st.dataframe(template, use_container_width=True)

    csv_template = template.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Template CSV",
        csv_template,
        "template_input_prediksi.csv",
        "text/csv"
    )

elif menu == "Info Model":
    st.title("Informasi Model dan Artifact")

    st.subheader("Path Artifact")
    st.code(
        f"MODEL_PATH = {MODEL_PATH}\n"
        f"PREPROCESSOR_PATH = {PREPROCESSOR_PATH}\n"
        f"SCALER_PATH = {SCALER_PATH}\n"
        f"FEATURE_COLUMNS_PATH = {FEATURE_COLUMNS_PATH}\n"
        f"CLEAN_TEST_PATH = {CLEAN_TEST_PATH}\n"
        f"TEST_PREDICTIONS_PATH = {TEST_PREDICTIONS_PATH}"
    )

    st.subheader("Konfigurasi Model")
    st.json(artifacts["config"])

    st.subheader("Checkpoint")
    st.json({
        "input_dim": artifacts["checkpoint"].get("input_dim"),
        "best_epoch": artifacts["checkpoint"].get("best_epoch"),
        "best_val_rmse": artifacts["checkpoint"].get("best_val_rmse"),
        "best_val_mae": artifacts["checkpoint"].get("best_val_mae"),
        "objective_score": artifacts["checkpoint"].get("objective_score"),
        "sample_weight_mode": artifacts["checkpoint"].get("sample_weight_mode"),
    })

    st.subheader("Daftar Fitur Akhir Setelah Preprocessing")
    feature_columns = artifacts["feature_columns"]
    st.write(f"Jumlah fitur akhir: {len(feature_columns)}")
    st.dataframe(pd.DataFrame({"feature": feature_columns}), use_container_width=True)
