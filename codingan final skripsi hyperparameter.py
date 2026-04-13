from __future__ import annotations

import json
import random
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# =========================================================
# KONFIGURASI UTAMA
# =========================================================
SEED = 42
DATA_PATH = "dataset5000ready.csv"
OUTPUT_DIR = Path("hasil_ann_eda_aligned")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()
MIN_DELTA = 1e-4

# text config
TFIDF_MIN_DF = 3
TFIDF_MAX_FEATURES = 2000
TFIDF_NGRAM_RANGE = (3, 5)
TEXT_SVD_COMPONENTS = 96

# =========================================================
# SAFE HYPERPARAMETER RANGE SESUAI EDA
# =========================================================
SAFE_HPARAM_RANGES = {
    "tfidf_min_df": [2, 3, 4, 5],
    "tfidf_max_features": [1500, 2000, 2500],
    "tfidf_ngram_range": [(3, 5)],
    "text_svd_components": [64, 96, 128, 160],
    "hidden_layers": [
        [256, 128],
        [320, 160],
        [384, 192, 96],
        [448, 224, 112],
        [512, 256, 128],
    ],
    "dropout": [0.05, 0.10, 0.15, 0.20],
    "learning_rate": [3e-4, 5e-4, 7e-4, 1e-3],
    "weight_decay": [1e-6, 5e-6, 1e-5, 2e-5, 5e-5],
    "batch_size": [32, 48, 64],
    "max_epochs": [160, 180, 200, 220],
    "patience": [15, 18, 20, 24],
    "use_batchnorm": [True],
    "loss_name": ["huber"],
}

# =========================================================
# KANDIDAT ANN YANG PALING MASUK AKAL UNTUK PIPELINE EDA
# =========================================================
# Fokus:
# - model tidak terlalu kecil agar bisa menangkap interaksi text + tabular
# - tidak terlalu besar agar tidak overfit pada distribution shift
# - learning rate moderat, dropout rendah-sedang, batchnorm aktif
ANN_CANDIDATES = [
    {
        "name": "ann_safe_01",
        "hidden_layers": [256, 128],
        "dropout": 0.05,
        "learning_rate": 1.0e-3,
        "weight_decay": 1.0e-5,
        "batch_size": 64,
        "max_epochs": 160,
        "patience": 15,
        "use_batchnorm": True,
        "loss_name": "huber",
    },
    {
        "name": "ann_safe_02",
        "hidden_layers": [320, 160],
        "dropout": 0.10,
        "learning_rate": 8.0e-4,
        "weight_decay": 1.0e-5,
        "batch_size": 64,
        "max_epochs": 180,
        "patience": 18,
        "use_batchnorm": True,
        "loss_name": "huber",
    },
    {
        "name": "ann_safe_03",
        "hidden_layers": [384, 192, 96],
        "dropout": 0.10,
        "learning_rate": 7.0e-4,
        "weight_decay": 1.0e-5,
        "batch_size": 48,
        "max_epochs": 200,
        "patience": 20,
        "use_batchnorm": True,
        "loss_name": "huber",
    },
    {
        "name": "ann_safe_04",
        "hidden_layers": [384, 192, 96],
        "dropout": 0.15,
        "learning_rate": 5.0e-4,
        "weight_decay": 2.0e-5,
        "batch_size": 48,
        "max_epochs": 200,
        "patience": 20,
        "use_batchnorm": True,
        "loss_name": "huber",
    },
    {
        "name": "ann_safe_05",
        "hidden_layers": [448, 224, 112],
        "dropout": 0.15,
        "learning_rate": 5.0e-4,
        "weight_decay": 2.0e-5,
        "batch_size": 48,
        "max_epochs": 200,
        "patience": 20,
        "use_batchnorm": True,
        "loss_name": "huber",
    },
    {
        "name": "ann_safe_06",
        "hidden_layers": [512, 256, 128],
        "dropout": 0.20,
        "learning_rate": 3.0e-4,
        "weight_decay": 5.0e-5,
        "batch_size": 32,
        "max_epochs": 220,
        "patience": 24,
        "use_batchnorm": True,
        "loss_name": "huber",
    },
]

# konfigurasi awal yang paling saya rekomendasikan
RECOMMENDED_START_CONFIG = {
    "text_svd_components": 96,
    "hidden_layers": [384, 192, 96],
    "dropout": 0.10,
    "learning_rate": 7.0e-4,
    "weight_decay": 1.0e-5,
    "batch_size": 48,
    "max_epochs": 200,
    "patience": 20,
    "use_batchnorm": True,
    "loss_name": "huber",
}

RIDGE_ALPHAS = [3, 10, 30, 50, 100, 150, 300]


# =========================================================
# UTILITAS
# =========================================================
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def write_json(obj: Dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def standardize_colname(col: str) -> str:
    col = str(col).strip().lower()
    col = re.sub(r"[^a-z0-9]+", "_", col)
    col = re.sub(r"_+", "_", col).strip("_")
    return col


def clip_rating_predictions(preds: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(preds, dtype=float), 1.0, 5.0)


def compute_error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = clip_rating_predictions(y_pred)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"rmse": rmse, "mae": mae}


def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# =========================================================
# ENCODER KATEGORI HIGH-CARDINALITY
# =========================================================
class FrequencyEncoder:
    def fit(self, X: pd.DataFrame):
        X = pd.DataFrame(X).copy()
        self.cols_ = X.columns.tolist()
        n = len(X)
        self.maps_ = {}

        for col in self.cols_:
            vc = X[col].astype(str).fillna("__NA__").value_counts(dropna=False)
            self.maps_[col] = (vc / n).to_dict()

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(X).copy()
        encoded = []

        for col in self.cols_:
            arr = (
                X[col]
                .astype(str)
                .fillna("__NA__")
                .map(self.maps_[col])
                .fillna(0.0)
                .astype(float)
                .to_numpy()
                .reshape(-1, 1)
            )
            encoded.append(arr)

        return np.hstack(encoded).astype(np.float32)


# =========================================================
# PREPROCESSOR HYBRID ANN
# =========================================================
class HybridANNPreprocessor:
    def __init__(
        self,
        numeric_features: List[str],
        low_cardinality_features: List[str],
        high_cardinality_features: List[str],
        text_feature: str,
        tfidf_min_df: int = 3,
        tfidf_max_features: int = 2000,
        tfidf_ngram_range=(3, 5),
        text_svd_components: int = 128,
    ):
        self.numeric_features = numeric_features
        self.low_cardinality_features = low_cardinality_features
        self.high_cardinality_features = high_cardinality_features
        self.text_feature = text_feature
        self.tfidf_min_df = tfidf_min_df
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_ngram_range = tfidf_ngram_range
        self.text_svd_components = text_svd_components

        self.num_imputer = SimpleImputer(strategy="median")
        self.num_scaler = StandardScaler()

        self.low_imputer = SimpleImputer(strategy="most_frequent")
        self.low_ohe = make_onehot_encoder()

        self.high_imputer = SimpleImputer(strategy="most_frequent")
        self.high_freq = FrequencyEncoder()

        self.text_tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.tfidf_ngram_range,
            min_df=self.tfidf_min_df,
            max_features=self.tfidf_max_features,
        )
        self.text_svd = None
        self.use_svd_ = False

        self.final_scaler = StandardScaler()

    def fit(self, X: pd.DataFrame):
        X = X.copy()

        # numeric
        X_num = self.num_imputer.fit_transform(X[self.numeric_features])
        self.num_scaler.fit(X_num)

        # low-card categorical
        X_low = self.low_imputer.fit_transform(X[self.low_cardinality_features])
        self.low_ohe.fit(X_low)

        # high-card categorical
        X_high = self.high_imputer.fit_transform(X[self.high_cardinality_features])
        X_high_df = pd.DataFrame(X_high, columns=self.high_cardinality_features)
        self.high_freq.fit(X_high_df)

        # text
        X_text_sparse = self.text_tfidf.fit_transform(
            X[self.text_feature].fillna("").astype(str)
        )

        n_text_features = X_text_sparse.shape[1]
        if n_text_features >= 3:
            n_components = min(self.text_svd_components, n_text_features - 1)
            self.text_svd = TruncatedSVD(n_components=n_components, random_state=SEED)
            self.text_svd.fit(X_text_sparse)
            self.use_svd_ = True
        else:
            self.use_svd_ = False

        X_dense = self._build_dense_features(X, fit_mode=True)
        self.final_scaler.fit(X_dense)

        return self

    def _text_to_dense(self, X_text_series: pd.Series) -> np.ndarray:
        X_text_sparse = self.text_tfidf.transform(
            X_text_series.fillna("").astype(str)
        )
        if self.use_svd_:
            return self.text_svd.transform(X_text_sparse).astype(np.float32)
        return X_text_sparse.toarray().astype(np.float32)

    def _build_dense_features(self, X: pd.DataFrame, fit_mode: bool = False) -> np.ndarray:
        X = X.copy()

        X_num = self.num_scaler.transform(
            self.num_imputer.transform(X[self.numeric_features])
        ).astype(np.float32)

        X_low = self.low_ohe.transform(
            self.low_imputer.transform(X[self.low_cardinality_features])
        )
        if hasattr(X_low, "toarray"):
            X_low = X_low.toarray()
        X_low = np.asarray(X_low, dtype=np.float32)

        X_high = self.high_freq.transform(
            pd.DataFrame(
                self.high_imputer.transform(X[self.high_cardinality_features]),
                columns=self.high_cardinality_features,
            )
        ).astype(np.float32)

        X_text = self._text_to_dense(X[self.text_feature])

        X_dense = np.hstack([X_num, X_low, X_high, X_text]).astype(np.float32)
        return X_dense

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_dense = self._build_dense_features(X)
        X_dense = self.final_scaler.transform(X_dense).astype(np.float32)
        return X_dense


# =========================================================
# FEATURE ENGINEERING SESUAI EDA
# =========================================================
def add_indicator_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["has_box_dimension"] = (
        (df[["box_panjang_cm", "box_lebar_cm", "box_tinggi_cm"]].fillna(0) > 0).any(axis=1)
    ).astype(int)

    df["has_roll_dimension"] = (
        (df[["lebar_cm_roll", "panjang_meter_roll"]].fillna(0) > 0).any(axis=1)
    ).astype(int)

    df["has_yard_dimension"] = (
        (df[["yard", "panjang_yard_pair", "lebar_pair_yard"]].fillna(0) > 0).any(axis=1)
    ).astype(int)

    return df


def prepare_feature_frame(df: pd.DataFrame):
    df = add_indicator_features(df)

    # drop fitur yang sangat multikolinear berdasarkan EDA
    numeric_features = [
        "log_harga",
        "is_non_brand",
        "berat_kg",
        "tebal_mm",
        "jumlah_lembar",
        "ukuran_cm",
        "panjang_meter",
        "lebar_cm_roll",
        "log_volume_box_cm3",
        "jumlah_dimensi_x",
        "panjang_nama",
        "jumlah_token",
        "is_bflute",
        "is_cflute",
        "is_foam",
        "is_kraft",
        "has_box_dimension",
        "has_roll_dimension",
        "has_yard_dimension",
    ]

    low_cardinality_features = [
        "kategori_produk",
        "jenis_produk",
        "warna",
        "material",
        "bentuk_kemasan",
    ]

    high_cardinality_features = [
        "merek",
        "family_template",
    ]

    text_feature = "nama_produk_normalized"

    selected_cols = (
        numeric_features
        + low_cardinality_features
        + high_cardinality_features
        + [text_feature]
    )

    X = df[selected_cols].copy()
    y = df["review"].astype(float).to_numpy()

    return (
        X,
        y,
        numeric_features,
        low_cardinality_features,
        high_cardinality_features,
        text_feature,
    )


# =========================================================
# LOAD DAN SPLIT DATA
# =========================================================
def load_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [standardize_colname(c) for c in df.columns]

    required_cols = [
        "kategori_produk",
        "review",
        "family_template",
        "jenis_produk",
        "merek",
        "warna",
        "material",
        "bentuk_kemasan",
        "log_harga",
        "is_non_brand",
        "berat_kg",
        "tebal_mm",
        "jumlah_lembar",
        "ukuran_cm",
        "panjang_meter",
        "yard",
        "box_panjang_cm",
        "box_lebar_cm",
        "box_tinggi_cm",
        "log_volume_box_cm3",
        "lebar_cm_roll",
        "panjang_meter_roll",
        "lebar_pair_yard",
        "panjang_yard_pair",
        "is_bflute",
        "is_cflute",
        "is_foam",
        "is_kraft",
        "jumlah_dimensi_x",
        "panjang_nama",
        "jumlah_token",
        "nama_produk_normalized",
        "split_grouped",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

    return df


def split_from_column(df: pd.DataFrame):
    train_df = df.loc[df["split_grouped"] == "train"].copy()
    val_df = df.loc[df["split_grouped"] == "validation"].copy()
    test_df = df.loc[df["split_grouped"] == "test"].copy()

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("Split train/validation/test tidak lengkap di split_grouped.")

    return train_df, val_df, test_df


# =========================================================
# MODEL ANN
# =========================================================
class ANNRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: Sequence[int],
        dropout: float,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        current_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, int(hidden_dim)))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(int(hidden_dim)))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            current_dim = int(hidden_dim)

        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.network(x)


def make_criterion(loss_name: str):
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "huber":
        return nn.SmoothL1Loss(beta=0.25)
    raise ValueError(f"loss_name tidak dikenali: {loss_name}")


def make_loader(X_array: np.ndarray, y_array: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    X_array = np.ascontiguousarray(X_array, dtype=np.float32)
    y_array = np.ascontiguousarray(y_array, dtype=np.float32).reshape(-1, 1)

    dataset = TensorDataset(
        torch.from_numpy(X_array),
        torch.from_numpy(y_array),
    )

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )


def inverse_transform_target(y_scaled: np.ndarray, y_scaler: StandardScaler) -> np.ndarray:
    return y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()


def evaluate_loader(
    model: ANNRegressor,
    loader: DataLoader,
    criterion,
    y_scaler: StandardScaler,
):
    model.eval()

    total_loss = 0.0
    preds_scaled = []
    targets_scaled = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            pred = model(xb)
            loss = criterion(pred, yb)
            total_loss += loss.item() * len(xb)

            preds_scaled.append(pred.cpu().numpy().ravel())
            targets_scaled.append(yb.cpu().numpy().ravel())

    avg_loss = total_loss / len(loader.dataset)

    preds_scaled = np.concatenate(preds_scaled)
    targets_scaled = np.concatenate(targets_scaled)

    preds = inverse_transform_target(preds_scaled, y_scaler)
    targets = inverse_transform_target(targets_scaled, y_scaler)

    metrics = compute_error_metrics(targets, preds)
    return avg_loss, metrics, preds, targets


def train_model(
    model: ANNRegressor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    y_scaler: StandardScaler,
    learning_rate: float,
    weight_decay: float,
    loss_name: str,
    max_epochs: int,
    patience: int,
    min_delta: float = 1e-4,
    verbose_prefix: str = "",
):
    criterion = make_criterion(loss_name)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    model.to(DEVICE)

    best_state = deepcopy(model.state_dict())
    best_val_rmse = float("inf")
    best_val_mae = float("inf")
    best_epoch = 0
    patience_counter = 0

    history = {
        "train_rmse": [],
        "val_rmse": [],
        "train_mae": [],
        "val_mae": [],
        "learning_rate": [],
    }

    for epoch in range(1, int(max_epochs) + 1):
        model.train()

        train_preds_scaled = []
        train_targets_scaled = []

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_preds_scaled.append(pred.detach().cpu().numpy().ravel())
            train_targets_scaled.append(yb.detach().cpu().numpy().ravel())

        train_preds_scaled = np.concatenate(train_preds_scaled)
        train_targets_scaled = np.concatenate(train_targets_scaled)

        train_preds = inverse_transform_target(train_preds_scaled, y_scaler)
        train_targets = inverse_transform_target(train_targets_scaled, y_scaler)
        train_metrics = compute_error_metrics(train_targets, train_preds)

        _, val_metrics, _, _ = evaluate_loader(model, val_loader, criterion, y_scaler)

        history["train_rmse"].append(train_metrics["rmse"])
        history["val_rmse"].append(val_metrics["rmse"])
        history["train_mae"].append(train_metrics["mae"])
        history["val_mae"].append(val_metrics["mae"])
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))

        scheduler.step(val_metrics["rmse"])

        improved = False
        if val_metrics["rmse"] < best_val_rmse - min_delta:
            improved = True
        elif abs(val_metrics["rmse"] - best_val_rmse) <= min_delta and val_metrics["mae"] < best_val_mae:
            improved = True

        if improved:
            best_val_rmse = float(val_metrics["rmse"])
            best_val_mae = float(val_metrics["mae"])
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"{verbose_prefix}Epoch {epoch:03d} | "
            f"Train RMSE: {train_metrics['rmse']:.4f} | "
            f"Val RMSE: {val_metrics['rmse']:.4f} | "
            f"Train MAE: {train_metrics['mae']:.4f} | "
            f"Val MAE: {val_metrics['mae']:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if patience_counter >= int(patience):
            print(f"{verbose_prefix}Early stopping pada epoch {epoch}")
            break

    model.load_state_dict(best_state)

    return model, history, {"val_rmse": best_val_rmse, "val_mae": best_val_mae}, int(best_epoch)


# =========================================================
# BASELINE RIDGE HYBRID
# =========================================================
def fit_ridge_baseline(X_train, y_train, X_val, y_val, X_test, y_test):
    rows = []
    best_model = None
    best_alpha = None
    best_val_metrics = None

    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        val_metrics = compute_error_metrics(y_val, val_pred)
        rows.append({"alpha": alpha, "val_rmse": val_metrics["rmse"], "val_mae": val_metrics["mae"]})

        if best_val_metrics is None or (val_metrics["rmse"], val_metrics["mae"]) < (best_val_metrics["rmse"], best_val_metrics["mae"]):
            best_model = model
            best_alpha = alpha
            best_val_metrics = val_metrics

    test_pred = best_model.predict(X_test)
    test_metrics = compute_error_metrics(y_test, test_pred)

    results_df = pd.DataFrame(rows).sort_values(by=["val_rmse", "val_mae"]).reset_index(drop=True)
    return best_model, best_alpha, best_val_metrics, test_metrics, clip_rating_predictions(test_pred), results_df


# =========================================================
# VISUALISASI
# =========================================================
def plot_training_history(history: Dict[str, List[float]], output_dir: Path, prefix: str = "ann"):
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["train_rmse"]) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_rmse"], label="Train RMSE")
    plt.plot(epochs, history["val_rmse"], label="Validation RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title(f"Train vs Validation RMSE - {prefix}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"epoch_rmse_{prefix}.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_mae"], label="Train MAE")
    plt.plot(epochs, history["val_mae"], label="Validation MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title(f"Train vs Validation MAE - {prefix}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"epoch_mae_{prefix}.png", dpi=150)
    plt.close()


def plot_metric_comparison(metrics: Dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = list(metrics.keys())
    rmse_values = [metrics[m]["rmse"] for m in model_names]
    mae_values = [metrics[m]["mae"] for m in model_names]

    plt.figure(figsize=(9, 5))
    plt.bar(model_names, rmse_values)
    plt.ylabel("RMSE")
    plt.title("Perbandingan RMSE")
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_rmse.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(model_names, mae_values)
    plt.ylabel("MAE")
    plt.title("Perbandingan MAE")
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_mae.png", dpi=150)
    plt.close()


def plot_prediction_scatter(y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path, prefix: str = "ann_test"):
    output_dir.mkdir(parents=True, exist_ok=True)

    y_pred = clip_rating_predictions(y_pred)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.65)
    low = min(float(np.min(y_true)), float(np.min(y_pred)))
    high = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([low, high], [low, high])
    plt.xlabel("Actual Review")
    plt.ylabel("Predicted Review")
    plt.title(f"Actual vs Predicted - {prefix}")
    plt.tight_layout()
    plt.savefig(output_dir / f"actual_vs_predicted_{prefix}.png", dpi=150)
    plt.close()


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    seed_everything(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(DATA_PATH)
    train_df, val_df, test_df = split_from_column(df)

    # dedup train sesuai EDA
    train_before = len(train_df)
    train_df = train_df.drop_duplicates().copy()
    train_after = len(train_df)

    # feature frame
    (
        X_train_df,
        y_train,
        numeric_features,
        low_cardinality_features,
        high_cardinality_features,
        text_feature,
    ) = prepare_feature_frame(train_df)

    (
        X_val_df,
        y_val,
        _,
        _,
        _,
        _,
    ) = prepare_feature_frame(val_df)

    (
        X_test_df,
        y_test,
        _,
        _,
        _,
        _,
    ) = prepare_feature_frame(test_df)

    # hybrid preprocessor
    preprocessor = HybridANNPreprocessor(
        numeric_features=numeric_features,
        low_cardinality_features=low_cardinality_features,
        high_cardinality_features=high_cardinality_features,
        text_feature=text_feature,
        tfidf_min_df=TFIDF_MIN_DF,
        tfidf_max_features=TFIDF_MAX_FEATURES,
        tfidf_ngram_range=TFIDF_NGRAM_RANGE,
        text_svd_components=TEXT_SVD_COMPONENTS,
    )
    preprocessor.fit(X_train_df)

    X_train = preprocessor.transform(X_train_df)
    X_val = preprocessor.transform(X_val_df)
    X_test = preprocessor.transform(X_test_df)

    # target scaling
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).ravel()

    # baseline mean (validation only)
    baseline_val_pred = np.full(len(y_val), y_train.mean(), dtype=float)
    baseline_val_metrics = compute_error_metrics(y_val, baseline_val_pred)

    # ridge hybrid baseline
    ridge_model, ridge_alpha, ridge_val_metrics, ridge_test_metrics, ridge_test_pred, ridge_grid_df = fit_ridge_baseline(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )
    ridge_grid_df.to_csv(OUTPUT_DIR / "ridge_grid_results.csv", index=False)

    # train ANN candidates
    candidate_rows = []
    best_ann_model = None
    best_ann_history = None
    best_ann_config = None
    best_ann_best_epoch = None
    best_ann_val_metrics = None

    for idx, cfg in enumerate(ANN_CANDIDATES, start=1):
        print("\n" + "=" * 70)
        print(f"TRAIN ANN CANDIDATE {idx}/{len(ANN_CANDIDATES)} -> {cfg['name']}")
        print("=" * 70)

        seed_everything(SEED + idx)

        train_loader = make_loader(
            X_train,
            y_train_scaled,
            batch_size=cfg["batch_size"],
            shuffle=True,
        )
        val_loader = make_loader(
            X_val,
            y_val_scaled,
            batch_size=cfg["batch_size"],
            shuffle=False,
        )

        model = ANNRegressor(
            input_dim=X_train.shape[1],
            hidden_layers=cfg["hidden_layers"],
            dropout=cfg["dropout"],
            use_batchnorm=cfg["use_batchnorm"],
        )

        model, history, best_metrics, best_epoch = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            y_scaler=y_scaler,
            learning_rate=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
            loss_name=cfg["loss_name"],
            max_epochs=cfg["max_epochs"],
            patience=cfg["patience"],
            min_delta=MIN_DELTA,
            verbose_prefix=f"[{cfg['name']}] ",
        )

        candidate_rows.append(
            {
                "name": cfg["name"],
                "hidden_layers": str(cfg["hidden_layers"]),
                "dropout": cfg["dropout"],
                "learning_rate": cfg["learning_rate"],
                "weight_decay": cfg["weight_decay"],
                "batch_size": cfg["batch_size"],
                "max_epochs": cfg["max_epochs"],
                "patience": cfg["patience"],
                "use_batchnorm": cfg["use_batchnorm"],
                "loss_name": cfg["loss_name"],
                "best_epoch": best_epoch,
                "val_rmse": best_metrics["val_rmse"],
                "val_mae": best_metrics["val_mae"],
            }
        )

        if best_ann_val_metrics is None or (best_metrics["val_rmse"], best_metrics["val_mae"]) < (best_ann_val_metrics["val_rmse"], best_ann_val_metrics["val_mae"]):
            best_ann_model = model
            best_ann_history = history
            best_ann_config = cfg
            best_ann_best_epoch = best_epoch
            best_ann_val_metrics = best_metrics

    candidate_df = pd.DataFrame(candidate_rows).sort_values(by=["val_rmse", "val_mae"]).reset_index(drop=True)
    candidate_df.to_csv(OUTPUT_DIR / "ann_candidate_results.csv", index=False)

    # plot best validation history
    plot_training_history(best_ann_history, OUTPUT_DIR, prefix="best_ann_validation")

    # refit final ANN on train + validation
    trainval_df = pd.concat([train_df, val_df], axis=0, ignore_index=True).drop_duplicates().copy()

    (
        X_trainval_df,
        y_trainval,
        numeric_features,
        low_cardinality_features,
        high_cardinality_features,
        text_feature,
    ) = prepare_feature_frame(trainval_df)

    final_preprocessor = HybridANNPreprocessor(
        numeric_features=numeric_features,
        low_cardinality_features=low_cardinality_features,
        high_cardinality_features=high_cardinality_features,
        text_feature=text_feature,
        tfidf_min_df=TFIDF_MIN_DF,
        tfidf_max_features=TFIDF_MAX_FEATURES,
        tfidf_ngram_range=TFIDF_NGRAM_RANGE,
        text_svd_components=TEXT_SVD_COMPONENTS,
    )
    final_preprocessor.fit(X_trainval_df)

    X_trainval = final_preprocessor.transform(X_trainval_df)
    X_test_final = final_preprocessor.transform(X_test_df)

    y_trainval_scaler = StandardScaler()
    y_trainval_scaled = y_trainval_scaler.fit_transform(y_trainval.reshape(-1, 1)).ravel()
    final_train_loader = make_loader(
        X_trainval,
        y_trainval_scaled,
        batch_size=best_ann_config["batch_size"],
        shuffle=True,
    )

    final_test_loader = make_loader(
        X_test_final,
        y_trainval_scaler.transform(y_test.reshape(-1, 1)).ravel(),
        batch_size=best_ann_config["batch_size"],
        shuffle=False,
    )

    final_ann_model = ANNRegressor(
        input_dim=X_trainval.shape[1],
        hidden_layers=best_ann_config["hidden_layers"],
        dropout=best_ann_config["dropout"],
        use_batchnorm=best_ann_config["use_batchnorm"],
    )

    # refit fixed epochs sesuai best epoch validation
    criterion = make_criterion(best_ann_config["loss_name"])
    optimizer = torch.optim.AdamW(
        final_ann_model.parameters(),
        lr=float(best_ann_config["learning_rate"]),
        weight_decay=float(best_ann_config["weight_decay"]),
    )

    final_ann_model.to(DEVICE)

    refit_history = {
        "train_rmse": [],
        "train_mae": [],
    }

    for epoch in range(1, int(best_ann_best_epoch) + 1):
        final_ann_model.train()

        preds_scaled = []
        targets_scaled = []

        for xb, yb in final_train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            pred = final_ann_model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_ann_model.parameters(), max_norm=1.0)
            optimizer.step()

            preds_scaled.append(pred.detach().cpu().numpy().ravel())
            targets_scaled.append(yb.detach().cpu().numpy().ravel())

        preds_scaled = np.concatenate(preds_scaled)
        targets_scaled = np.concatenate(targets_scaled)

        preds = inverse_transform_target(preds_scaled, y_trainval_scaler)
        targets = inverse_transform_target(targets_scaled, y_trainval_scaler)
        train_metrics = compute_error_metrics(targets, preds)

        refit_history["train_rmse"].append(train_metrics["rmse"])
        refit_history["train_mae"].append(train_metrics["mae"])

        print(
            f"[ANN-REFIT] Epoch {epoch:03d} | "
            f"Train RMSE: {train_metrics['rmse']:.4f} | "
            f"Train MAE: {train_metrics['mae']:.4f}"
        )

    _, final_ann_test_metrics, final_ann_test_pred, final_ann_test_true = evaluate_loader(
        final_ann_model,
        final_test_loader,
        criterion,
        y_trainval_scaler,
    )

    # save predictions
    pred_df = pd.DataFrame(
        {
            "y_true": final_ann_test_true,
            "y_pred_ann": clip_rating_predictions(final_ann_test_pred),
            "y_pred_ridge": ridge_test_pred,
            "residual_ann": final_ann_test_true - clip_rating_predictions(final_ann_test_pred),
            "residual_ridge": final_ann_test_true - ridge_test_pred,
        }
    )
    pred_df.to_csv(OUTPUT_DIR / "test_predictions_ann_vs_ridge.csv", index=False)

    # plots
    plot_prediction_scatter(final_ann_test_true, final_ann_test_pred, OUTPUT_DIR, prefix="ann_test")
    plot_prediction_scatter(final_ann_test_true, ridge_test_pred, OUTPUT_DIR, prefix="ridge_test")

    comparison_metrics = {
        "BaselineMean_Val": baseline_val_metrics,
        "HybridRidge_Val": ridge_val_metrics,
        "HybridRidge_Test": ridge_test_metrics,
        "HybridANN_Val": {
            "rmse": float(best_ann_val_metrics["val_rmse"]),
            "mae": float(best_ann_val_metrics["val_mae"]),
        },
        "HybridANN_Test": final_ann_test_metrics,
    }
    plot_metric_comparison(comparison_metrics, OUTPUT_DIR)

    # summary
    summary = {
        "seed": SEED,
        "data_path": str(DATA_PATH),
        "device": DEVICE,
        "n_rows_total": int(len(df)),
        "split_sizes": {
            "train_before_dedup": int(train_before),
            "train_after_dedup": int(train_after),
            "validation": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "preprocessing": {
            "numeric_features": numeric_features,
            "low_cardinality_features": low_cardinality_features,
            "high_cardinality_features": high_cardinality_features,
            "text_feature": text_feature,
            "tfidf_min_df": TFIDF_MIN_DF,
            "tfidf_max_features": TFIDF_MAX_FEATURES,
            "tfidf_ngram_range": list(TFIDF_NGRAM_RANGE),
            "text_svd_components_requested": TEXT_SVD_COMPONENTS,
        },
        "ridge_baseline": {
            "best_alpha": ridge_alpha,
            "validation_metrics": ridge_val_metrics,
            "test_metrics": ridge_test_metrics,
        },
        "best_ann_config": best_ann_config,
        "best_ann_best_epoch": int(best_ann_best_epoch),
        "best_ann_validation_metrics": {
            "rmse": float(best_ann_val_metrics["val_rmse"]),
            "mae": float(best_ann_val_metrics["val_mae"]),
        },
        "best_ann_test_metrics": final_ann_test_metrics,
        "note": (
            "ANN ini mengikuti EDA: dedup train, drop fitur multikolinear, "
            "gunakan indikator sparse feature, one-hot untuk kategori low-cardinality, "
            "frequency encoding untuk kategori high-cardinality, "
            "dan TF-IDF char n-gram + SVD untuk nama produk."
        ),
    }
    write_json(summary, OUTPUT_DIR / "run_summary.json")

    print("\n=== FINAL ANN SESUAI EDA ===")
    print("Best ANN config     :", best_ann_config["name"])
    print("Best validation RMSE:", best_ann_val_metrics["val_rmse"])
    print("Best validation MAE :", best_ann_val_metrics["val_mae"])
    print("Test RMSE           :", final_ann_test_metrics["rmse"])
    print("Test MAE            :", final_ann_test_metrics["mae"])
    print("Output directory    :", OUTPUT_DIR.resolve())
