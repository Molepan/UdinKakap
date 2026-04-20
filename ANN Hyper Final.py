import os
import re
import json
import random
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# =========================================================
# CONFIG
# =========================================================
DATA_FILE = "dataset5000ready.csv"
OUTPUT_DIR = "ann_stable_final"

RANDOM_SEED = 124
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()

# fokus ke stabilitas, bukan random search dulu
MAX_EPOCHS = 120
PATIENCE = 15
MIN_DELTA = 5e-5

# preprocessing text
TEXT_MAX_FEATURES = 3000
TEXT_MIN_DF = 3
TEXT_SVD_COMPONENTS = 48

# regularisasi / stabilitas
CATEGORY_MIN_COUNT = 8
USE_WEIGHTED_SAMPLER = False
DROP_EXACT_DUPLICATES = False

# config tetap yang lebih aman dan halus
CONFIG = {
    "batch_size": 64,
    "learning_rate": 5e-5,
    "weight_decay": 2e-5,
    "hidden_layers": [160, 80, 40],
    "dropout": 0.02,
    "loss_name": "huber",          # huber / mse
    "norm_name": "layernorm",      # layernorm / batchnorm / none
    "activation_name": "silu",     # silu / relu
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# HELPER
# =========================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def standardize_colname(col: str) -> str:
    col = str(col).strip().lower()
    col = re.sub(r"[^a-z0-9]+", "_", col)
    col = re.sub(r"_+", "_", col).strip("_")
    return col


def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def clip_rating_predictions(preds: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(preds, dtype=float), 1.0, 5.0)


def round_to_half(preds: np.ndarray) -> np.ndarray:
    preds = clip_rating_predictions(preds)
    return np.round(preds * 2.0) / 2.0


def inverse_transform_target(y_scaled, y_scaler):
    y_scaled = np.asarray(y_scaled).reshape(-1, 1)
    return y_scaler.inverse_transform(y_scaled).ravel()


def make_criterion(loss_name: str):
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "huber":
        return nn.SmoothL1Loss(beta=0.25)
    raise ValueError(f"loss_name tidak dikenali: {loss_name}")


def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def write_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(obj), f, indent=2, ensure_ascii=False)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true, dtype=float)
    y_pred_raw = np.asarray(y_pred, dtype=float)
    y_pred_clipped = clip_rating_predictions(y_pred_raw)
    y_pred_half = round_to_half(y_pred_raw)

    return {
        "raw_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred_raw))),
        "raw_mae": float(mean_absolute_error(y_true, y_pred_raw)),
        "clipped_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred_clipped))),
        "clipped_mae": float(mean_absolute_error(y_true, y_pred_clipped)),
        "halfstep_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred_half))),
        "halfstep_mae": float(mean_absolute_error(y_true, y_pred_half)),
        "halfstep_accuracy": float(np.mean(y_true == y_pred_half)),
        "within_half_point": float(np.mean(np.abs(y_true - y_pred_half) <= 0.5)),
    }


# =========================================================
# DATASET
# =========================================================
def load_processed_dataset(path: str):
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

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Kolom wajib tidak ditemukan pada dataset: {missing_cols}")

    df["split_grouped"] = df["split_grouped"].astype(str).str.strip().str.lower()
    df["review"] = pd.to_numeric(df["review"], errors="coerce")
    df = df.loc[df["review"].between(1.0, 5.0, inclusive="both")].copy()

    return df.reset_index(drop=True)


def split_by_split_grouped(df: pd.DataFrame):
    train_df = df.loc[df["split_grouped"] == "train"].copy()
    val_df = df.loc[df["split_grouped"] == "validation"].copy()
    test_df = df.loc[df["split_grouped"] == "test"].copy()

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("Kolom split_grouped harus berisi train, validation, dan test.")

    return train_df, val_df, test_df


def add_indicator_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

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


NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = [
    "kategori_produk",
    "jenis_produk",
    "merek",
    "warna",
    "material",
    "bentuk_kemasan",
    "family_template",
]

TEXT_FEATURE = "nama_produk_normalized"


def apply_train_based_rare_category_mapping(train_df, val_df, test_df, cat_cols, min_count=8):
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    for col in cat_cols:
        train_df[col] = train_df[col].fillna("lainnya").astype(str).str.strip()
        val_df[col] = val_df[col].fillna("lainnya").astype(str).str.strip()
        test_df[col] = test_df[col].fillna("lainnya").astype(str).str.strip()

        counts = train_df[col].value_counts(dropna=False)
        allowed = set(counts[counts >= min_count].index.tolist())

        train_df[col] = train_df[col].where(train_df[col].isin(allowed), "lainnya")
        val_df[col] = val_df[col].where(val_df[col].isin(allowed), "lainnya")
        test_df[col] = test_df[col].where(test_df[col].isin(allowed), "lainnya")

    return train_df, val_df, test_df


def build_feature_table(df: pd.DataFrame):
    data = add_indicator_features(df.copy())

    if DROP_EXACT_DUPLICATES:
        data = data.drop_duplicates().reset_index(drop=True)
    else:
        data = data.reset_index(drop=True)

    for col in NUMERIC_FEATURES:
        if col not in data.columns:
            data[col] = 0.0
        data[col] = pd.to_numeric(data[col], errors="coerce")

    for col in CATEGORICAL_FEATURES:
        if col not in data.columns:
            data[col] = "lainnya"
        data[col] = data[col].fillna("lainnya").astype(str)

    if TEXT_FEATURE not in data.columns:
        data[TEXT_FEATURE] = ""
    data[TEXT_FEATURE] = data[TEXT_FEATURE].fillna("").astype(str)

    y = pd.to_numeric(data["review"], errors="coerce").astype(float).to_numpy()
    valid_mask = np.isfinite(y)

    data = data.loc[valid_mask].reset_index(drop=True)
    y = y[valid_mask]

    X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TEXT_FEATURE]].copy()
    return X, y, data


# =========================================================
# EDA
# =========================================================
def build_eda_summary(full_df, train_df, val_df, test_df):
    summary = {
        "shape_full": [int(full_df.shape[0]), int(full_df.shape[1])],
        "shape_train": [int(train_df.shape[0]), int(train_df.shape[1])],
        "shape_val": [int(val_df.shape[0]), int(val_df.shape[1])],
        "shape_test": [int(test_df.shape[0]), int(test_df.shape[1])],
        "split_counts": full_df["split_grouped"].value_counts().to_dict(),
        "review_distribution_full": full_df["review"].value_counts().sort_index().to_dict(),
        "review_distribution_train": train_df["review"].value_counts().sort_index().to_dict(),
        "review_distribution_val": val_df["review"].value_counts().sort_index().to_dict(),
        "review_distribution_test": test_df["review"].value_counts().sort_index().to_dict(),
        "review_mean_by_split": {
            "train": float(train_df["review"].mean()),
            "validation": float(val_df["review"].mean()),
            "test": float(test_df["review"].mean()),
        },
        "review_std_by_split": {
            "train": float(train_df["review"].std()),
            "validation": float(val_df["review"].std()),
            "test": float(test_df["review"].std()),
        },
        "missing_ratio_top15": full_df.isna().mean().sort_values(ascending=False).head(15).to_dict(),
        "duplicated_rows_full": int(full_df.duplicated().sum()),
        "duplicated_name_products_full": int(full_df["nama_produk_normalized"].duplicated().sum()),
    }
    return summary


# =========================================================
# TEXT TRANSFORMER YANG LEBIH TAHAN ERROR
# =========================================================
class SafeTfidfSVDVectorizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        max_features=3000,
        min_df=3,
        ngram_range=(1, 2),
        n_components=48,
        random_state=42,
    ):
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
        texts = self._to_series(X)

        self.vectorizer_ = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            sublinear_tf=True,
        )

        self.use_zero_output_ = False
        self.svd_ = None
        self.output_dim_ = 1

        try:
            X_tfidf = self.vectorizer_.fit_transform(texts)
        except ValueError:
            self.use_zero_output_ = True
            self.output_dim_ = 1
            return self

        n_features = X_tfidf.shape[1]

        if n_features <= 0:
            self.use_zero_output_ = True
            self.output_dim_ = 1
            return self

        if n_features == 1:
            self.output_dim_ = 1
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

        if self.use_zero_output_:
            return np.zeros((len(texts), 1), dtype=np.float32)

        X_tfidf = self.vectorizer_.transform(texts)

        if self.svd_ is not None:
            X_out = self.svd_.transform(X_tfidf)
            return np.asarray(X_out, dtype=np.float32)

        return np.asarray(X_tfidf.toarray(), dtype=np.float32)


# =========================================================
# PREPROCESSOR
# =========================================================
def build_preprocessor():
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_onehot_encoder()),
    ])

    text_transformer = Pipeline(steps=[
        ("tfidf_svd", SafeTfidfSVDVectorizer(
            max_features=TEXT_MAX_FEATURES,
            min_df=TEXT_MIN_DF,
            ngram_range=(1, 2),
            n_components=TEXT_SVD_COMPONENTS,
            random_state=RANDOM_SEED,
        ))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("txt", text_transformer, [TEXT_FEATURE]),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    return preprocessor


# =========================================================
# ANN MODEL
# =========================================================
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
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.network(x)


# =========================================================
# EVALUASI LOADER
# =========================================================
def evaluate_loader(model, loader, criterion, y_scaler):
    model.eval()
    loss_sum = 0.0
    preds_scaled = []
    targets_scaled = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            pred = model(xb)
            loss = criterion(pred, yb)

            loss_sum += loss.item() * len(xb)
            preds_scaled.append(pred.detach().cpu().numpy().ravel())
            targets_scaled.append(yb.detach().cpu().numpy().ravel())

    avg_loss = loss_sum / len(loader.dataset)

    preds_scaled = np.concatenate(preds_scaled)
    targets_scaled = np.concatenate(targets_scaled)

    preds = inverse_transform_target(preds_scaled, y_scaler)
    targets = inverse_transform_target(targets_scaled, y_scaler)

    metrics = evaluate_predictions(targets, preds)
    return avg_loss, metrics, preds, targets


# =========================================================
# TRAINING
# =========================================================
def build_train_loader(X_train_processed, y_train_scaled, y_train_raw, batch_size):
    train_dataset = TensorDataset(
        torch.from_numpy(X_train_processed),
        torch.from_numpy(y_train_scaled),
    )

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    if USE_WEIGHTED_SAMPLER:
        counts = pd.Series(y_train_raw).value_counts().to_dict()
        sample_weights = np.array([1.0 / counts[v] for v in y_train_raw], dtype=np.float64)

        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )

    return train_loader


def train_model(model, train_loader, train_eval_loader, val_loader, y_scaler, config):
    criterion = make_criterion(config["loss_name"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        amsgrad=True,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    best_val_rmse = float("inf")
    best_val_mae = float("inf")
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_clipped_rmse": [],
        "val_clipped_rmse": [],
        "train_clipped_mae": [],
        "val_clipped_mae": [],
        "learning_rate": [],
    }

    model.to(DEVICE)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * len(xb)

        train_loss = train_loss_sum / len(train_loader.dataset)

        fair_train_loss, fair_train_metrics, _, _ = evaluate_loader(
            model=model,
            loader=train_eval_loader,
            criterion=criterion,
            y_scaler=y_scaler,
        )

        val_loss, val_metrics, _, _ = evaluate_loader(
            model=model,
            loader=val_loader,
            criterion=criterion,
            y_scaler=y_scaler,
        )

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_clipped_rmse"].append(float(fair_train_metrics["clipped_rmse"]))
        history["val_clipped_rmse"].append(float(val_metrics["clipped_rmse"]))
        history["train_clipped_mae"].append(float(fair_train_metrics["clipped_mae"]))
        history["val_clipped_mae"].append(float(val_metrics["clipped_mae"]))
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))

        # scheduler diselaraskan ke metric utama, bukan val_loss
        scheduler.step(val_metrics["clipped_rmse"])

        improved = False
        if val_metrics["clipped_rmse"] < best_val_rmse - MIN_DELTA:
            improved = True
        elif abs(val_metrics["clipped_rmse"] - best_val_rmse) <= MIN_DELTA and val_metrics["clipped_mae"] < best_val_mae:
            improved = True

        if improved:
            best_val_rmse = val_metrics["clipped_rmse"]
            best_val_mae = val_metrics["clipped_mae"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d} | "
            f"Train RMSE: {fair_train_metrics['clipped_rmse']:.4f} | "
            f"Val RMSE: {val_metrics['clipped_rmse']:.4f} | "
            f"Train MAE: {fair_train_metrics['clipped_mae']:.4f} | "
            f"Val MAE: {val_metrics['clipped_mae']:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        if patience_counter >= PATIENCE:
            print(f"Early stopping pada epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model, history, best_val_rmse, best_val_mae, best_epoch


# =========================================================
# PLOTTING
# =========================================================
def plot_training_history(history, output_dir, prefix="best"):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_clipped_rmse"], label="Train RMSE")
    plt.plot(history["val_clipped_rmse"], label="Validation RMSE")
    plt.title("RMSE over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"training_history_{prefix}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    return plot_path


# =========================================================
# MAIN
# =========================================================
def main():
    seed_everything(RANDOM_SEED)

    df = load_processed_dataset(DATA_FILE)
    train_df, val_df, test_df = split_by_split_grouped(df)

    eda_summary = build_eda_summary(df, train_df, val_df, test_df)
    write_json(eda_summary, os.path.join(OUTPUT_DIR, "eda_summary.json"))

    print("=" * 70)
    print("Ringkasan Dataset")
    print("=" * 70)
    print(f"Data file               : {DATA_FILE}")
    print(f"Train / Val / Test      : {len(train_df)} / {len(val_df)} / {len(test_df)}")
    print(f"Review mean train       : {train_df['review'].mean():.4f}")
    print(f"Review mean val         : {val_df['review'].mean():.4f}")
    print(f"Review mean test        : {test_df['review'].mean():.4f}")
    print(f"Config                  : {CONFIG}")
    print("=" * 70)

    # mapping rare category hanya berdasarkan train
    train_df, val_df, test_df = apply_train_based_rare_category_mapping(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        cat_cols=CATEGORICAL_FEATURES,
        min_count=CATEGORY_MIN_COUNT,
    )

    X_train, y_train, clean_train_df = build_feature_table(train_df)
    X_val, y_val, clean_val_df = build_feature_table(val_df)
    X_test, y_test, clean_test_df = build_feature_table(test_df)

    clean_train_df.to_csv(os.path.join(OUTPUT_DIR, "clean_train.csv"), index=False)
    clean_val_df.to_csv(os.path.join(OUTPUT_DIR, "clean_val.csv"), index=False)
    clean_test_df.to_csv(os.path.join(OUTPUT_DIR, "clean_test.csv"), index=False)

    preprocessor = build_preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train).astype(np.float32)
    X_val_processed = preprocessor.transform(X_val).astype(np.float32)
    X_test_processed = preprocessor.transform(X_test).astype(np.float32)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).astype(np.float32)
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).astype(np.float32)
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).astype(np.float32)

    input_dim = X_train_processed.shape[1]
    print(f"Jumlah fitur setelah preprocessing: {input_dim}")

    train_loader = build_train_loader(
        X_train_processed=X_train_processed,
        y_train_scaled=y_train_scaled,
        y_train_raw=y_train,
        batch_size=CONFIG["batch_size"],
    )

    train_eval_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train_processed),
            torch.from_numpy(y_train_scaled),
        ),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_val_processed),
            torch.from_numpy(y_val_scaled),
        ),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    test_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_test_processed),
            torch.from_numpy(y_test_scaled),
        ),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    model = ANNRegressor(
        input_dim=input_dim,
        hidden_layers=CONFIG["hidden_layers"],
        dropout=CONFIG["dropout"],
        norm_name=CONFIG["norm_name"],
        activation_name=CONFIG["activation_name"],
    )

    model, history, best_val_rmse, best_val_mae, best_epoch = train_model(
        model=model,
        train_loader=train_loader,
        train_eval_loader=train_eval_loader,
        val_loader=val_loader,
        y_scaler=y_scaler,
        config=CONFIG,
    )

    criterion = make_criterion(CONFIG["loss_name"])
    test_loss, test_metrics, test_preds, test_targets = evaluate_loader(
        model=model,
        loader=test_loader,
        criterion=criterion,
        y_scaler=y_scaler,
    )

    print("\n" + "=" * 70)
    print("HASIL AKHIR")
    print("=" * 70)
    print(f"Best Validation RMSE : {best_val_rmse:.4f}")
    print(f"Best Validation MAE  : {best_val_mae:.4f}")
    print(f"Best Epoch           : {best_epoch}")
    print(f"Test RMSE clipped    : {test_metrics['clipped_rmse']:.4f}")
    print(f"Test MAE clipped     : {test_metrics['clipped_mae']:.4f}")
    print(f"Within ±0.5          : {test_metrics['within_half_point']:.4f}")
    print("=" * 70)

    plot_path = plot_training_history(history, OUTPUT_DIR, prefix="stable_final")

    metrics = {
        "data_file": DATA_FILE,
        "device": DEVICE,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "input_dim_after_preprocessing": int(input_dim),
        "best_epoch": int(best_epoch),
        "best_val_rmse": float(best_val_rmse),
        "best_val_mae": float(best_val_mae),
        "test_loss": float(test_loss),
        "test_metrics": test_metrics,
        "model_config": CONFIG,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "min_delta": MIN_DELTA,
        "use_weighted_sampler": USE_WEIGHTED_SAMPLER,
        "drop_exact_duplicates": DROP_EXACT_DUPLICATES,
        "text_max_features": TEXT_MAX_FEATURES,
        "text_min_df": TEXT_MIN_DF,
        "text_svd_components": TEXT_SVD_COMPONENTS,
        "category_min_count": CATEGORY_MIN_COUNT,
    }

    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    write_json(metrics, metrics_path)

    predictions_df = pd.DataFrame({
        "actual_review": test_targets,
        "predicted_review_raw": test_preds,
        "predicted_review_clipped": clip_rating_predictions(test_preds),
        "predicted_review_halfstep": round_to_half(test_preds),
        "abs_error_clipped": np.abs(test_targets - clip_rating_predictions(test_preds)),
        "abs_error_halfstep": np.abs(test_targets - round_to_half(test_preds)),
    })
    predictions_path = os.path.join(OUTPUT_DIR, "test_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)

    history_path = os.path.join(OUTPUT_DIR, "history.json")
    write_json(history, history_path)

    model_path = os.path.join(OUTPUT_DIR, "best_ann_model.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "model_config": CONFIG,
            "best_epoch": best_epoch,
            "best_val_rmse": best_val_rmse,
            "best_val_mae": best_val_mae,
        },
        model_path,
    )

    print("\nFile output tersimpan:")
    print(f"- EDA summary        : {os.path.join(OUTPUT_DIR, 'eda_summary.json')}")
    print(f"- Clean train        : {os.path.join(OUTPUT_DIR, 'clean_train.csv')}")
    print(f"- Clean val          : {os.path.join(OUTPUT_DIR, 'clean_val.csv')}")
    print(f"- Clean test         : {os.path.join(OUTPUT_DIR, 'clean_test.csv')}")
    print(f"- Plot history       : {plot_path}")
    print(f"- Metrics            : {metrics_path}")
    print(f"- History            : {history_path}")
    print(f"- Prediksi test      : {predictions_path}")
    print(f"- Bobot model        : {model_path}")


if __name__ == "__main__":
    main()
