
import os
import re
import json
import math
import random
import argparse
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
# DEFAULT CONFIG
# =========================================================
DEFAULT_DATA_FILE_CANDIDATES = [
    "dataset5000ready.csv",
]
DEFAULT_OUTPUT_DIR = "ann_improved"

RANDOM_SEED = 124
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()

MAX_EPOCHS = 35
PATIENCE = 5
MIN_DELTA = 2e-4

DROP_EXACT_DUPLICATES = True
REBUILD_SPLIT = True
TARGET_MODE = "product_mean"      # product_mean / individual
GROUP_COL = "nama_produk_normalized"

MULTI_SEED_EVAL = True
DEFAULT_EVAL_SEEDS = [124, 3407, 2026, 777, 42]
SAMPLE_WEIGHT_MODE = "sqrt_review_count"   # sqrt_review_count / log1p_review_count / linear_review_count / none

BASE_CONFIG = {
    "batch_size": 64,
    "learning_rate": 3e-5,
    "weight_decay": 1e-4,
    "hidden_layers": [128, 64],
    "dropout": 0.12,
    "loss_name": "huber",
    "norm_name": "layernorm",
    "activation_name": "silu",
    "text_max_features": 3000,
    "text_min_df": 3,
    "text_svd_components": 32,
    "category_min_count": 12,
}

SEARCH_SPACE = {
    "batch_size": [64, 64, 96],
    "learning_rate": [2e-5, 3e-5, 4e-5, 5e-5],
    "weight_decay": [7e-5, 1e-4, 1.5e-4, 2e-4],
    "hidden_layers": [
        [96, 48],
        [128, 64],
        [128, 64, 32],
    ],
    "dropout": [0.10, 0.12, 0.15],
    "text_svd_components": [24, 32, 40],
    "category_min_count": [10, 12, 16],
}

TUNING_MAX_TRIALS = 40
TUNING_OVERFIT_PENALTY = 0.20
TUNING_GAP_TOLERANCE = 0.010

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

def make_criterion(loss_name: str, reduction: str = "mean"):
    if loss_name == "mse":
        return nn.MSELoss(reduction=reduction)
    if loss_name == "huber":
        return nn.SmoothL1Loss(beta=0.25, reduction=reduction)
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

def resolve_data_file(user_path=None):
    candidates = []
    if user_path is not None:
        candidates.append(user_path)
    candidates.extend(DEFAULT_DATA_FILE_CANDIDATES)
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(
        "File dataset tidak ditemukan. Pastikan salah satu file ini ada: "
        + ", ".join(DEFAULT_DATA_FILE_CANDIDATES)
        + " atau kirim path lewat --data-file"
    )

def config_to_hashable(config: dict):
    items = []
    for key in sorted(config.keys()):
        val = config[key]
        if isinstance(val, list):
            val = tuple(val)
        items.append((key, val))
    return tuple(items)

def sample_trial_configs(base_config, search_space, max_trials, seed=42):
    rng = random.Random(seed)
    trial_configs = [deepcopy(base_config)]
    seen = {config_to_hashable(base_config)}
    keys = list(search_space.keys())
    max_attempts = max_trials * 200
    attempts = 0
    while len(trial_configs) < max_trials and attempts < max_attempts:
        cfg = deepcopy(base_config)
        for key in keys:
            cfg[key] = deepcopy(rng.choice(search_space[key]))
        hashed = config_to_hashable(cfg)
        if hashed in seen:
            attempts += 1
            continue
        trial_configs.append(cfg)
        seen.add(hashed)
    return trial_configs

def compute_objective_score(val_rmse, train_rmse, penalty_weight=0.20, gap_tolerance=0.010):
    gap = max(0.0, float(val_rmse) - float(train_rmse) - float(gap_tolerance))
    return float(val_rmse) + float(penalty_weight) * gap

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

def safe_mode(series: pd.Series):
    series = series.dropna().astype(str)
    if series.empty:
        return "lainnya"
    return series.mode().iloc[0]

def make_rating_bin(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(series.mean())
    s = np.clip(np.round(s * 2.0) / 2.0, 1.0, 5.0)
    return s.astype(str)


def parse_seed_list(seed_text: str | None):
    if seed_text is None:
        return list(DEFAULT_EVAL_SEEDS)
    tokens = [tok.strip() for tok in str(seed_text).split(",") if tok.strip() != ""]
    if not tokens:
        return list(DEFAULT_EVAL_SEEDS)
    return [int(tok) for tok in tokens]


def compute_sample_weights_from_count(review_count, mode: str = SAMPLE_WEIGHT_MODE):
    counts = np.asarray(review_count, dtype=float)
    counts = np.where(np.isfinite(counts), counts, 1.0)
    counts = np.clip(counts, 1.0, None)

    if mode == "none":
        weights = np.ones_like(counts, dtype=np.float32)
    elif mode == "sqrt_review_count":
        weights = np.sqrt(counts)
    elif mode == "log1p_review_count":
        weights = np.log1p(counts)
    elif mode == "linear_review_count":
        weights = counts
    else:
        raise ValueError(f"mode sample weight tidak dikenali: {mode}")

    weights = weights.astype(np.float32)
    mean_weight = float(np.mean(weights)) if len(weights) > 0 else 1.0
    if mean_weight <= 0:
        return np.ones_like(weights, dtype=np.float32)
    return (weights / mean_weight).astype(np.float32)

# =========================================================
# DATASET
# =========================================================
def load_processed_dataset(path: str):
    df = pd.read_csv(path)
    df.columns = [standardize_colname(c) for c in df.columns]
    required_cols = [
        "kategori_produk", "review", "family_template", "jenis_produk", "merek",
        "warna", "material", "bentuk_kemasan", "log_harga", "is_non_brand",
        "berat_kg", "tebal_mm", "jumlah_lembar", "ukuran_cm", "panjang_meter",
        "yard", "box_panjang_cm", "box_lebar_cm", "box_tinggi_cm", "log_volume_box_cm3",
        "lebar_cm_roll", "panjang_meter_roll", "lebar_pair_yard", "panjang_yard_pair",
        "is_bflute", "is_cflute", "is_foam", "is_kraft", "jumlah_dimensi_x",
        "panjang_nama", "jumlah_token", "nama_produk_normalized",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Kolom wajib tidak ditemukan pada dataset: {missing_cols}")
    df["review"] = pd.to_numeric(df["review"], errors="coerce")
    df = df.loc[df["review"].between(1.0, 5.0, inclusive="both")].copy()
    if "split_grouped" in df.columns:
        df["split_grouped"] = df["split_grouped"].astype(str).str.strip().str.lower()
    else:
        df["split_grouped"] = "unknown"
    df[GROUP_COL] = df[GROUP_COL].fillna("").astype(str).str.strip()
    df = df.loc[df[GROUP_COL] != ""].reset_index(drop=True)
    return df

def build_noise_audit(df: pd.DataFrame, group_col: str, feature_cols: list[str]):
    audit = {}
    audit["n_rows"] = int(len(df))
    audit["n_groups"] = int(df[group_col].nunique())
    audit["duplicated_rows_full"] = int(df.duplicated().sum())
    audit["duplicated_group_names"] = int(df[group_col].duplicated().sum())

    same_input_cols = feature_cols.copy()
    if "review" in df.columns:
        grp = (
            df.groupby(same_input_cols, dropna=False)["review"]
            .agg(["nunique", "count", "mean", "std"])
            .reset_index()
        )
        conflict = grp.loc[grp["nunique"] > 1].copy()
        audit["n_conflicting_input_patterns"] = int(len(conflict))
        audit["ratio_conflicting_input_patterns"] = float(len(conflict) / max(1, len(grp)))
        audit["n_rows_in_conflicting_input_patterns"] = int(conflict["count"].sum()) if len(conflict) else 0

    per_group = df.groupby(group_col)["review"].agg(["size", "mean", "std", "nunique"]).reset_index()
    noisy_groups = per_group.loc[per_group["nunique"] > 1].copy()
    audit["groups_with_multiple_ratings"] = int(len(noisy_groups))
    audit["ratio_groups_with_multiple_ratings"] = float(len(noisy_groups) / max(1, len(per_group)))
    audit["top_noisy_groups"] = (
        noisy_groups.sort_values(["nunique", "size"], ascending=[False, False])
        .head(10)
        .to_dict(orient="records")
    )
    return audit

def aggregate_to_product_level(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "log_harga", "is_non_brand", "berat_kg", "tebal_mm", "jumlah_lembar",
        "ukuran_cm", "panjang_meter", "yard", "box_panjang_cm", "box_lebar_cm",
        "box_tinggi_cm", "log_volume_box_cm3", "lebar_cm_roll", "panjang_meter_roll",
        "lebar_pair_yard", "panjang_yard_pair", "jumlah_dimensi_x", "panjang_nama",
        "jumlah_token", "is_bflute", "is_cflute", "is_foam", "is_kraft",
    ]
    categorical_cols = [
        "kategori_produk", "family_template", "jenis_produk", "merek",
        "warna", "material", "bentuk_kemasan"
    ]

    grouped = df.groupby(GROUP_COL, dropna=False)

    agg_num = grouped[numeric_cols].median(numeric_only=True)
    agg_cat = grouped[categorical_cols].agg(safe_mode)
    agg_target = grouped["review"].agg(
        review="mean",
        review_count="size",
        review_std=lambda s: float(pd.to_numeric(s, errors="coerce").std(ddof=0) if len(s) > 1 else 0.0),
        review_nunique="nunique",
    )
    agg_text = grouped[[GROUP_COL]].first()

    result = pd.concat([agg_num, agg_cat, agg_target, agg_text], axis=1).reset_index(drop=True)
    result["review_std"] = result["review_std"].fillna(0.0)
    result["review_count"] = result["review_count"].astype(int)
    result["review_nunique"] = result["review_nunique"].astype(int)
    result["split_grouped"] = "unknown"
    return result

def create_balanced_group_splits(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    seed: int = 124,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
):
    """
    Split group-aware yang tahan terhadap kelas/bin langka.

    Kenapa tidak lagi memakai StratifiedShuffleSplit dua tahap?
    Karena untuk bin yang sangat jarang, subset sementara hasil split tahap-1
    bisa hanya menyisakan 1 grup pada suatu kelas. Saat itu split tahap-2 gagal.

    Strategi di sini:
    1. Bentuk tabel level-group.
    2. Bin target ke kelipatan 0.5.
    3. Untuk setiap bin, acak grup lalu alokasikan langsung ke
       train / validation / test.
    4. Paksa minimal 1 grup di train. Untuk bin dengan >= 3 grup,
       usahakan validation dan test masing-masing minimal 1.
    """
    if df[group_col].duplicated().any():
        group_df = (
            df.groupby(group_col, dropna=False)[target_col]
            .agg(target_for_split="mean")
            .reset_index()
        )
    else:
        group_df = df[[group_col, target_col]].rename(columns={target_col: "target_for_split"}).copy()

    group_df["target_bin"] = make_rating_bin(group_df["target_for_split"])

    rng = np.random.default_rng(seed)
    split_map = {}

    for _, bin_df in group_df.groupby("target_bin", sort=True):
        groups = bin_df[group_col].tolist()
        rng.shuffle(groups)
        n = len(groups)

        if n == 1:
            n_train, n_val, n_test = 1, 0, 0
        elif n == 2:
            n_train, n_val, n_test = 1, 1, 0
        else:
            n_val = max(1, int(round(n * val_ratio)))
            n_test = max(1, int(round(n * test_ratio)))
            n_train = n - n_val - n_test

            while n_train < 1:
                if n_val >= n_test and n_val > 1:
                    n_val -= 1
                elif n_test > 1:
                    n_test -= 1
                else:
                    break
                n_train = n - n_val - n_test

            if n >= 3:
                if n_val < 1:
                    n_val = 1
                if n_test < 1:
                    n_test = 1
                n_train = n - n_val - n_test

                while n_train < 1:
                    if n_val >= n_test and n_val > 1:
                        n_val -= 1
                    elif n_test > 1:
                        n_test -= 1
                    else:
                        break
                    n_train = n - n_val - n_test

        train_groups = groups[:n_train]
        val_groups = groups[n_train:n_train + n_val]
        test_groups = groups[n_train + n_val:n_train + n_val + n_test]

        for g in train_groups:
            split_map[g] = "train"
        for g in val_groups:
            split_map[g] = "validation"
        for g in test_groups:
            split_map[g] = "test"

    out = df.copy()
    out["split_grouped"] = out[group_col].map(split_map).fillna("train")
    return out

def split_by_split_grouped(df: pd.DataFrame):
    train_df = df.loc[df["split_grouped"] == "train"].copy()
    val_df = df.loc[df["split_grouped"] == "validation"].copy()
    test_df = df.loc[df["split_grouped"] == "test"].copy()
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("Kolom split_grouped harus berisi train, validation, dan test.")
    return train_df, val_df, test_df

def add_indicator_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["box_panjang_cm", "box_lebar_cm", "box_tinggi_cm", "lebar_cm_roll", "panjang_meter_roll", "yard", "lebar_pair_yard", "panjang_yard_pair"]:
        if col not in df.columns:
            df[col] = 0.0
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

def apply_train_based_rare_category_mapping(train_df, val_df, test_df, cat_cols, min_count=12):
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

    if "review_count" not in data.columns:
        data["review_count"] = 1
    data["review_count"] = pd.to_numeric(data["review_count"], errors="coerce").fillna(1).clip(lower=1)

    y = pd.to_numeric(data["review"], errors="coerce").astype(float).to_numpy()
    valid_mask = np.isfinite(y)
    data = data.loc[valid_mask].reset_index(drop=True)
    y = y[valid_mask]

    sample_weights = compute_sample_weights_from_count(
        data["review_count"].to_numpy() if "review_count" in data.columns else np.ones(len(data)),
        mode=SAMPLE_WEIGHT_MODE,
    )

    X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TEXT_FEATURE]].copy()
    return X, y, data, sample_weights

# =========================================================
# EDA
# =========================================================
def build_split_summary(df, train_df, val_df, test_df):
    summary = {
        "shape_full": [int(df.shape[0]), int(df.shape[1])],
        "shape_train": [int(train_df.shape[0]), int(train_df.shape[1])],
        "shape_val": [int(val_df.shape[0]), int(val_df.shape[1])],
        "shape_test": [int(test_df.shape[0]), int(test_df.shape[1])],
        "split_counts": df["split_grouped"].value_counts().to_dict(),
        "review_distribution_full": df["review"].value_counts().sort_index().to_dict(),
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
        "group_count_by_split": {
            "train": int(train_df[GROUP_COL].nunique()),
            "validation": int(val_df[GROUP_COL].nunique()),
            "test": int(test_df[GROUP_COL].nunique()),
        },
    }

    if "review_count" in df.columns:
        summary["review_count_mean_by_split"] = {
            "train": float(train_df["review_count"].mean()),
            "validation": float(val_df["review_count"].mean()),
            "test": float(test_df["review_count"].mean()),
        }
        summary["review_count_median_by_split"] = {
            "train": float(train_df["review_count"].median()),
            "validation": float(val_df["review_count"].median()),
            "test": float(test_df["review_count"].median()),
        }

    return summary

# =========================================================
# TEXT TRANSFORMER
# =========================================================
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
            return np.asarray(self.svd_.transform(X_tfidf), dtype=np.float32)
        return np.asarray(X_tfidf.toarray(), dtype=np.float32)

# =========================================================
# PREPROCESSOR
# =========================================================
def build_preprocessor(text_max_features, text_min_df, text_svd_components, random_seed):
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
            max_features=text_max_features,
            min_df=text_min_df,
            ngram_range=(1, 2),
            n_components=text_svd_components,
            random_state=random_seed,
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
def build_train_loader(X_train_processed, y_train_scaled, sample_weights, batch_size, seed):
    dataset = TensorDataset(
        torch.from_numpy(X_train_processed),
        torch.from_numpy(y_train_scaled),
        torch.from_numpy(np.asarray(sample_weights, dtype=np.float32)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    return loader

def train_model(model, train_loader, train_eval_loader, val_loader, y_scaler, config, max_epochs, patience, min_delta, verbose_epoch=False):
    train_criterion = make_criterion(config["loss_name"], reduction="none")
    eval_criterion = make_criterion(config["loss_name"], reduction="mean")
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
        "train_loss": [], "val_loss": [],
        "train_clipped_rmse": [], "val_clipped_rmse": [],
        "train_clipped_mae": [], "val_clipped_mae": [],
        "learning_rate": [],
    }

    model.to(DEVICE)

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_weight_sum = 0.0

        for batch in train_loader:
            if len(batch) == 3:
                xb, yb, wb = batch
                wb = wb.to(DEVICE).view(-1)
            else:
                xb, yb = batch
                wb = None

            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            pred = model(xb)
            loss_values = train_criterion(pred, yb).view(-1)

            if wb is not None:
                loss = (loss_values * wb).sum() / torch.clamp(wb.sum(), min=1e-8)
                train_loss_sum += float((loss_values.detach() * wb).sum().item())
                train_weight_sum += float(wb.detach().sum().item())
            else:
                loss = loss_values.mean()
                train_loss_sum += float(loss_values.detach().sum().item())
                train_weight_sum += float(len(loss_values))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        train_loss = train_loss_sum / max(train_weight_sum, 1e-8)
        fair_train_loss, fair_train_metrics, _, _ = evaluate_loader(model, train_eval_loader, eval_criterion, y_scaler)
        val_loss, val_metrics, _, _ = evaluate_loader(model, val_loader, eval_criterion, y_scaler)

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_clipped_rmse"].append(float(fair_train_metrics["clipped_rmse"]))
        history["val_clipped_rmse"].append(float(val_metrics["clipped_rmse"]))
        history["train_clipped_mae"].append(float(fair_train_metrics["clipped_mae"]))
        history["val_clipped_mae"].append(float(val_metrics["clipped_mae"]))
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))

        scheduler.step(val_metrics["clipped_rmse"])

        improved = False
        if val_metrics["clipped_rmse"] < best_val_rmse - min_delta:
            improved = True
        elif abs(val_metrics["clipped_rmse"] - best_val_rmse) <= min_delta and val_metrics["clipped_mae"] < best_val_mae:
            improved = True

        if improved:
            best_val_rmse = val_metrics["clipped_rmse"]
            best_val_mae = val_metrics["clipped_mae"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose_epoch:
            print(
                f"Epoch {epoch:03d} | "
                f"Train RMSE: {fair_train_metrics['clipped_rmse']:.4f} | "
                f"Val RMSE: {val_metrics['clipped_rmse']:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

        if patience_counter >= patience:
            break

    model.load_state_dict(best_state)
    best_train_rmse = float(history["train_clipped_rmse"][best_epoch - 1]) if best_epoch > 0 else math.inf
    best_train_mae = float(history["train_clipped_mae"][best_epoch - 1]) if best_epoch > 0 else math.inf
    return {
        "model": model,
        "history": history,
        "best_val_rmse": float(best_val_rmse),
        "best_val_mae": float(best_val_mae),
        "best_epoch": int(best_epoch),
        "best_train_rmse": best_train_rmse,
        "best_train_mae": best_train_mae,
    }

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

def plot_tuning_results(results_df, output_dir):
    if results_df.empty:
        return None
    plt.figure(figsize=(8, 5))
    plt.scatter(results_df["trial_id"], results_df["best_val_rmse"], label="Val RMSE")
    plt.scatter(results_df["trial_id"], results_df["objective_score"], label="Objective Score")
    plt.title("Ringkasan Hasil Hyperparameter Tuning")
    plt.xlabel("Trial ID")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "tuning_summary.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    return plot_path

# =========================================================
# DATA PREP CACHE
# =========================================================
def make_preprocessing_key(config):
    return (
        config["text_max_features"],
        config["text_min_df"],
        config["text_svd_components"],
        config["category_min_count"],
    )

def prepare_data_bundle(train_df_raw, val_df_raw, test_df_raw, config, random_seed):
    train_df, val_df, test_df = apply_train_based_rare_category_mapping(
        train_df=train_df_raw,
        val_df=val_df_raw,
        test_df=test_df_raw,
        cat_cols=CATEGORICAL_FEATURES,
        min_count=config["category_min_count"],
    )
    X_train, y_train, clean_train_df, train_sample_weights = build_feature_table(train_df)
    X_val, y_val, clean_val_df, val_sample_weights = build_feature_table(val_df)
    X_test, y_test, clean_test_df, test_sample_weights = build_feature_table(test_df)

    preprocessor = build_preprocessor(
        text_max_features=config["text_max_features"],
        text_min_df=config["text_min_df"],
        text_svd_components=config["text_svd_components"],
        random_seed=random_seed,
    )
    X_train_processed = preprocessor.fit_transform(X_train).astype(np.float32)
    X_val_processed = preprocessor.transform(X_val).astype(np.float32)
    X_test_processed = preprocessor.transform(X_test).astype(np.float32)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).astype(np.float32)
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).astype(np.float32)
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).astype(np.float32)

    return {
        "X_train_processed": X_train_processed,
        "X_val_processed": X_val_processed,
        "X_test_processed": X_test_processed,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "y_train_scaled": y_train_scaled,
        "y_val_scaled": y_val_scaled,
        "y_test_scaled": y_test_scaled,
        "train_sample_weights": train_sample_weights,
        "val_sample_weights": val_sample_weights,
        "test_sample_weights": test_sample_weights,
        "y_scaler": y_scaler,
        "input_dim": int(X_train_processed.shape[1]),
        "clean_train_df": clean_train_df,
        "clean_val_df": clean_val_df,
        "clean_test_df": clean_test_df,
    }

def get_cached_data_bundle(cache, train_df_raw, val_df_raw, test_df_raw, config, random_seed):
    key = make_preprocessing_key(config)
    if key not in cache:
        cache[key] = prepare_data_bundle(
            train_df_raw=train_df_raw,
            val_df_raw=val_df_raw,
            test_df_raw=test_df_raw,
            config=config,
            random_seed=random_seed,
        )
    return cache[key]

def compute_baseline_metrics(train_df, val_df, test_df):
    baseline_value = float(train_df["review"].mean())
    val_pred = np.full(len(val_df), baseline_value)
    test_pred = np.full(len(test_df), baseline_value)
    return {
        "baseline_constant_prediction": baseline_value,
        "validation": evaluate_predictions(val_df["review"].to_numpy(), val_pred),
        "test": evaluate_predictions(test_df["review"].to_numpy(), test_pred),
    }

# =========================================================
# TUNING LOOP
# =========================================================
def run_single_trial(trial_id, data_bundle, config, max_epochs, patience, min_delta, seed):
    train_loader = build_train_loader(
        X_train_processed=data_bundle["X_train_processed"],
        y_train_scaled=data_bundle["y_train_scaled"],
        sample_weights=data_bundle["train_sample_weights"],
        batch_size=config["batch_size"],
        seed=seed,
    )
    train_eval_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(data_bundle["X_train_processed"]),
            torch.from_numpy(data_bundle["y_train_scaled"]),
        ),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(data_bundle["X_val_processed"]),
            torch.from_numpy(data_bundle["y_val_scaled"]),
        ),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    model = ANNRegressor(
        input_dim=data_bundle["input_dim"],
        hidden_layers=config["hidden_layers"],
        dropout=config["dropout"],
        norm_name=config["norm_name"],
        activation_name=config["activation_name"],
    )
    result = train_model(
        model=model,
        train_loader=train_loader,
        train_eval_loader=train_eval_loader,
        val_loader=val_loader,
        y_scaler=data_bundle["y_scaler"],
        config=config,
        max_epochs=max_epochs,
        patience=patience,
        min_delta=min_delta,
        verbose_epoch=False,
    )
    generalization_gap = float(result["best_val_rmse"] - result["best_train_rmse"])
    objective_score = compute_objective_score(
        val_rmse=result["best_val_rmse"],
        train_rmse=result["best_train_rmse"],
        penalty_weight=TUNING_OVERFIT_PENALTY,
        gap_tolerance=TUNING_GAP_TOLERANCE,
    )
    summary = {
        "trial_id": int(trial_id),
        "objective_score": float(objective_score),
        "best_val_rmse": float(result["best_val_rmse"]),
        "best_val_mae": float(result["best_val_mae"]),
        "best_train_rmse": float(result["best_train_rmse"]),
        "best_train_mae": float(result["best_train_mae"]),
        "generalization_gap": float(generalization_gap),
        "best_epoch": int(result["best_epoch"]),
        **sanitize_for_json(config),
    }
    return result, summary

def run_hyperparameter_tuning(train_df_raw, val_df_raw, test_df_raw, output_dir, base_config, search_space, max_trials, random_seed, max_epochs, patience, min_delta):
    trial_configs = sample_trial_configs(
        base_config=base_config,
        search_space=search_space,
        max_trials=max_trials,
        seed=random_seed,
    )
    cache = {}
    best_trial = None
    all_summaries = []

    print("=" * 80)
    print(f"Mulai hyperparameter tuning | jumlah trial: {len(trial_configs)}")
    print("=" * 80)

    for trial_id, config in enumerate(trial_configs, start=1):
        data_bundle = get_cached_data_bundle(
            cache=cache,
            train_df_raw=train_df_raw,
            val_df_raw=val_df_raw,
            test_df_raw=test_df_raw,
            config=config,
            random_seed=random_seed,
        )
        result, summary = run_single_trial(
            trial_id=trial_id,
            data_bundle=data_bundle,
            config=config,
            max_epochs=max_epochs,
            patience=patience,
            min_delta=min_delta,
            seed=random_seed,
        )
        all_summaries.append(summary)

        print(
            f"Trial {trial_id:02d} | "
            f"Obj: {summary['objective_score']:.4f} | "
            f"Val RMSE: {summary['best_val_rmse']:.4f} | "
            f"Gap: {summary['generalization_gap']:.4f} | "
            f"Config: bs={config['batch_size']}, lr={config['learning_rate']}, wd={config['weight_decay']}, "
            f"hidden={config['hidden_layers']}, drop={config['dropout']}, svd={config['text_svd_components']}, "
            f"catmin={config['category_min_count']}"
        )

        if best_trial is None:
            best_trial = {
                "config": deepcopy(config),
                "summary": deepcopy(summary),
                "history": deepcopy(result["history"]),
                "model_state_dict": deepcopy(result["model"].state_dict()),
                "input_dim": data_bundle["input_dim"],
                "data_key": make_preprocessing_key(config),
            }
        else:
            current_tuple = (
                summary["objective_score"],
                summary["best_val_rmse"],
                summary["best_val_mae"],
                summary["generalization_gap"],
            )
            best_tuple = (
                best_trial["summary"]["objective_score"],
                best_trial["summary"]["best_val_rmse"],
                best_trial["summary"]["best_val_mae"],
                best_trial["summary"]["generalization_gap"],
            )
            if current_tuple < best_tuple:
                best_trial = {
                    "config": deepcopy(config),
                    "summary": deepcopy(summary),
                    "history": deepcopy(result["history"]),
                    "model_state_dict": deepcopy(result["model"].state_dict()),
                    "input_dim": data_bundle["input_dim"],
                    "data_key": make_preprocessing_key(config),
                }

    results_df = pd.DataFrame(all_summaries).sort_values(
        by=["objective_score", "best_val_rmse", "best_val_mae", "generalization_gap"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)

    tuning_csv = os.path.join(output_dir, "tuning_results.csv")
    results_df.to_csv(tuning_csv, index=False)
    write_json(results_df.to_dict(orient="records"), os.path.join(output_dir, "tuning_results.json"))
    write_json(best_trial["config"], os.path.join(output_dir, "best_config.json"))
    plot_tuning_results(results_df, output_dir)

    print("=" * 80)
    print("Tuning selesai")
    print(f"Best objective score : {best_trial['summary']['objective_score']:.4f}")
    print(f"Best val RMSE        : {best_trial['summary']['best_val_rmse']:.4f}")
    print(f"Best val MAE         : {best_trial['summary']['best_val_mae']:.4f}")
    print(f"Best gap             : {best_trial['summary']['generalization_gap']:.4f}")
    print(f"Best config          : {best_trial['config']}")
    print("=" * 80)

    best_data_bundle = cache[best_trial["data_key"]]
    return best_trial, best_data_bundle, results_df

# =========================================================
# MULTI-SEED EVALUATION
# =========================================================
def fit_config_once(data_bundle, config, max_epochs, patience, min_delta, seed, verbose_epoch=False):
    train_loader = build_train_loader(
        X_train_processed=data_bundle["X_train_processed"],
        y_train_scaled=data_bundle["y_train_scaled"],
        sample_weights=data_bundle["train_sample_weights"],
        batch_size=config["batch_size"],
        seed=seed,
    )
    train_eval_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(data_bundle["X_train_processed"]),
            torch.from_numpy(data_bundle["y_train_scaled"]),
        ),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(data_bundle["X_val_processed"]),
            torch.from_numpy(data_bundle["y_val_scaled"]),
        ),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    model = ANNRegressor(
        input_dim=data_bundle["input_dim"],
        hidden_layers=config["hidden_layers"],
        dropout=config["dropout"],
        norm_name=config["norm_name"],
        activation_name=config["activation_name"],
    )
    result = train_model(
        model=model,
        train_loader=train_loader,
        train_eval_loader=train_eval_loader,
        val_loader=val_loader,
        y_scaler=data_bundle["y_scaler"],
        config=config,
        max_epochs=max_epochs,
        patience=patience,
        min_delta=min_delta,
        verbose_epoch=verbose_epoch,
    )
    return result


def evaluate_config_on_test(data_bundle, config, trained_result):
    test_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(data_bundle["X_test_processed"]),
            torch.from_numpy(data_bundle["y_test_scaled"]),
        ),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    criterion = make_criterion(config["loss_name"], reduction="mean")
    test_loss, test_metrics, test_preds, test_targets = evaluate_loader(
        model=trained_result["model"],
        loader=test_loader,
        criterion=criterion,
        y_scaler=data_bundle["y_scaler"],
    )
    return test_loss, test_metrics, test_preds, test_targets


def summarize_metric_series(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def run_multi_seed_evaluation(train_df_raw, val_df_raw, test_df_raw, config, eval_seeds, output_dir, max_epochs, patience, min_delta):
    rows = []
    best_seed_artifact = None

    for seed in eval_seeds:
        seed_everything(seed)
        data_bundle = prepare_data_bundle(
            train_df_raw=train_df_raw,
            val_df_raw=val_df_raw,
            test_df_raw=test_df_raw,
            config=config,
            random_seed=seed,
        )
        trained_result = fit_config_once(
            data_bundle=data_bundle,
            config=config,
            max_epochs=max_epochs,
            patience=patience,
            min_delta=min_delta,
            seed=seed,
            verbose_epoch=False,
        )
        test_loss, test_metrics, _, _ = evaluate_config_on_test(
            data_bundle=data_bundle,
            config=config,
            trained_result=trained_result,
        )

        row = {
            "seed": int(seed),
            "best_epoch": int(trained_result["best_epoch"]),
            "best_train_rmse": float(trained_result["best_train_rmse"]),
            "best_val_rmse": float(trained_result["best_val_rmse"]),
            "best_val_mae": float(trained_result["best_val_mae"]),
            "generalization_gap": float(trained_result["best_val_rmse"] - trained_result["best_train_rmse"]),
            "test_loss": float(test_loss),
            "test_rmse": float(test_metrics["clipped_rmse"]),
            "test_mae": float(test_metrics["clipped_mae"]),
            "test_within_half_point": float(test_metrics["within_half_point"]),
            "test_halfstep_accuracy": float(test_metrics["halfstep_accuracy"]),
        }
        rows.append(row)

        current_tuple = (row["best_val_rmse"], row["test_rmse"], row["test_mae"])
        if best_seed_artifact is None or current_tuple < best_seed_artifact["ranking_tuple"]:
            best_seed_artifact = {
                "ranking_tuple": current_tuple,
                "seed": int(seed),
                "history": deepcopy(trained_result["history"]),
                "model_state_dict": deepcopy(trained_result["model"].state_dict()),
                "input_dim": int(data_bundle["input_dim"]),
            }

    results_df = pd.DataFrame(rows).sort_values(["best_val_rmse", "test_rmse", "test_mae"], ascending=[True, True, True]).reset_index(drop=True)
    results_csv = os.path.join(output_dir, "multiseed_results.csv")
    results_json = os.path.join(output_dir, "multiseed_results.json")
    results_df.to_csv(results_csv, index=False)
    write_json(results_df.to_dict(orient="records"), results_json)

    summary = {
        "eval_seeds": [int(s) for s in eval_seeds],
        "n_seeds": int(len(eval_seeds)),
        "best_seed_by_validation": int(results_df.iloc[0]["seed"]),
        "metrics": {
            "best_val_rmse": summarize_metric_series(results_df["best_val_rmse"].to_numpy()),
            "best_val_mae": summarize_metric_series(results_df["best_val_mae"].to_numpy()),
            "test_rmse": summarize_metric_series(results_df["test_rmse"].to_numpy()),
            "test_mae": summarize_metric_series(results_df["test_mae"].to_numpy()),
            "test_within_half_point": summarize_metric_series(results_df["test_within_half_point"].to_numpy()),
            "generalization_gap": summarize_metric_series(results_df["generalization_gap"].to_numpy()),
            "best_epoch": summarize_metric_series(results_df["best_epoch"].to_numpy()),
        },
        "sample_weight_mode": SAMPLE_WEIGHT_MODE,
    }
    summary_path = os.path.join(output_dir, "multiseed_summary.json")
    write_json(summary, summary_path)

    if best_seed_artifact is not None:
        plot_training_history(best_seed_artifact["history"], output_dir, prefix=f"multiseed_best_seed_{best_seed_artifact['seed']}")
        best_seed_model_path = os.path.join(output_dir, "best_ann_model_multiseed_best_seed.pth")
        torch.save(
            {
                "model_state_dict": best_seed_artifact["model_state_dict"],
                "input_dim": best_seed_artifact["input_dim"],
                "model_config": config,
                "seed": best_seed_artifact["seed"],
                "sample_weight_mode": SAMPLE_WEIGHT_MODE,
            },
            best_seed_model_path,
        )
        summary["best_seed_model_path"] = best_seed_model_path
        write_json(summary, summary_path)

    return summary, results_df


# =========================================================
# FINAL EVAL + SAVE ARTIFACTS
# =========================================================
def evaluate_best_trial_on_test(best_trial, best_data_bundle, output_dir, data_file, meta_info):
    best_config = best_trial["config"]
    test_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(best_data_bundle["X_test_processed"]),
            torch.from_numpy(best_data_bundle["y_test_scaled"]),
        ),
        batch_size=best_config["batch_size"],
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    model = ANNRegressor(
        input_dim=best_trial["input_dim"],
        hidden_layers=best_config["hidden_layers"],
        dropout=best_config["dropout"],
        norm_name=best_config["norm_name"],
        activation_name=best_config["activation_name"],
    )
    model.load_state_dict(best_trial["model_state_dict"])
    model.to(DEVICE)

    criterion = make_criterion(best_config["loss_name"], reduction="mean")
    test_loss, test_metrics, test_preds, test_targets = evaluate_loader(
        model=model,
        loader=test_loader,
        criterion=criterion,
        y_scaler=best_data_bundle["y_scaler"],
    )

    best_history = best_trial["history"]
    plot_path = plot_training_history(best_history, output_dir, prefix="best_tuned")

    best_data_bundle["clean_train_df"].to_csv(os.path.join(output_dir, "clean_train_best_config.csv"), index=False)
    best_data_bundle["clean_val_df"].to_csv(os.path.join(output_dir, "clean_val_best_config.csv"), index=False)
    best_data_bundle["clean_test_df"].to_csv(os.path.join(output_dir, "clean_test_best_config.csv"), index=False)

    predictions_df = pd.DataFrame({
        GROUP_COL: best_data_bundle["clean_test_df"][GROUP_COL].values if GROUP_COL in best_data_bundle["clean_test_df"].columns else [""] * len(test_targets),
        "review_count": best_data_bundle["clean_test_df"]["review_count"].values if "review_count" in best_data_bundle["clean_test_df"].columns else np.ones(len(test_targets)),
        "actual_review": test_targets,
        "predicted_review_raw": test_preds,
        "predicted_review_clipped": clip_rating_predictions(test_preds),
        "predicted_review_halfstep": round_to_half(test_preds),
        "abs_error_clipped": np.abs(test_targets - clip_rating_predictions(test_preds)),
        "abs_error_halfstep": np.abs(test_targets - round_to_half(test_preds)),
    })
    predictions_path = os.path.join(output_dir, "test_predictions_best_tuned.csv")
    predictions_df.to_csv(predictions_path, index=False)

    history_path = os.path.join(output_dir, "best_history.json")
    write_json(best_history, history_path)

    model_path = os.path.join(output_dir, "best_ann_model_tuned.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": best_trial["input_dim"],
            "model_config": best_config,
            "best_epoch": best_trial["summary"]["best_epoch"],
            "best_val_rmse": best_trial["summary"]["best_val_rmse"],
            "best_val_mae": best_trial["summary"]["best_val_mae"],
            "objective_score": best_trial["summary"]["objective_score"],
            "sample_weight_mode": SAMPLE_WEIGHT_MODE,
        },
        model_path,
    )

    metrics = {
        "data_file": data_file,
        "device": DEVICE,
        "input_dim_after_preprocessing": int(best_trial["input_dim"]),
        "best_trial_summary": best_trial["summary"],
        "best_config": best_config,
        "test_loss": float(test_loss),
        "test_metrics": test_metrics,
        "sample_weight_mode": SAMPLE_WEIGHT_MODE,
        **meta_info,
    }
    metrics_path = os.path.join(output_dir, "final_metrics.json")
    write_json(metrics, metrics_path)

    print("\n" + "=" * 80)
    print("HASIL FINAL SETELAH PERBAIKAN SISTEM")
    print("=" * 80)
    print(f"Best Validation RMSE : {best_trial['summary']['best_val_rmse']:.4f}")
    print(f"Best Validation MAE  : {best_trial['summary']['best_val_mae']:.4f}")
    print(f"Best Epoch           : {best_trial['summary']['best_epoch']}")
    print(f"Test RMSE clipped    : {test_metrics['clipped_rmse']:.4f}")
    print(f"Test MAE clipped     : {test_metrics['clipped_mae']:.4f}")
    print(f"Within ±0.5          : {test_metrics['within_half_point']:.4f}")
    print(f"Sample weight mode   : {SAMPLE_WEIGHT_MODE}")
    print("=" * 80)

    print("\nFile output tersimpan:")
    print(f"- Best config        : {os.path.join(output_dir, 'best_config.json')}")
    print(f"- Final metrics      : {metrics_path}")
    print(f"- History terbaik    : {history_path}")
    print(f"- Plot history       : {plot_path}")
    print(f"- Prediksi test      : {predictions_path}")
    print(f"- Bobot model        : {model_path}")
    return metrics

# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="ANN rating prediction - versi perbaikan")
    parser.add_argument("--data-file", type=str, default=None, help="Path dataset CSV")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Folder output")
    parser.add_argument("--max-trials", type=int, default=TUNING_MAX_TRIALS, help="Jumlah trial tuning")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS, help="Maks epoch per trial")
    parser.add_argument("--patience", type=int, default=PATIENCE, help="Early stopping patience")
    parser.add_argument("--min-delta", type=float, default=MIN_DELTA, help="Minimum delta improvement")
    parser.add_argument("--target-mode", type=str, default=TARGET_MODE, choices=["individual", "product_mean"])
    parser.add_argument("--rebuild-split", action="store_true")
    parser.add_argument("--eval-seeds", type=str, default=",".join(str(s) for s in DEFAULT_EVAL_SEEDS), help="Daftar seed dipisah koma untuk evaluasi multi-seed")
    parser.add_argument("--disable-multi-seed", action="store_true", help="Matikan evaluasi multi-seed")
    args = parser.parse_args()

    seed_everything(RANDOM_SEED)
    os.makedirs(args.output_dir, exist_ok=True)

    data_file = resolve_data_file(args.data_file)
    raw_df = load_processed_dataset(data_file)

    audit = build_noise_audit(
        raw_df,
        group_col=GROUP_COL,
        feature_cols=[c for c in raw_df.columns if c not in ["review", "split_grouped"]],
    )
    write_json(audit, os.path.join(args.output_dir, "noise_audit.json"))

    modeling_df = raw_df.copy()
    if args.target_mode == "product_mean":
        modeling_df = aggregate_to_product_level(modeling_df)

    if args.rebuild_split or REBUILD_SPLIT:
        modeling_df = create_balanced_group_splits(
            modeling_df,
            group_col=GROUP_COL,
            target_col="review",
            seed=RANDOM_SEED,
        )

    train_df, val_df, test_df = split_by_split_grouped(modeling_df)
    split_summary = build_split_summary(modeling_df, train_df, val_df, test_df)
    write_json(split_summary, os.path.join(args.output_dir, "split_summary.json"))

    baseline = compute_baseline_metrics(train_df, val_df, test_df)
    write_json(baseline, os.path.join(args.output_dir, "baseline_metrics.json"))

    eval_seeds = parse_seed_list(args.eval_seeds)

    print("=" * 80)
    print("Ringkasan Dataset")
    print("=" * 80)
    print(f"Data file               : {data_file}")
    print(f"Target mode             : {args.target_mode}")
    print(f"Train / Val / Test      : {len(train_df)} / {len(val_df)} / {len(test_df)}")
    print(f"Review mean train       : {train_df['review'].mean():.4f}")
    print(f"Review mean val         : {val_df['review'].mean():.4f}")
    print(f"Review mean test        : {test_df['review'].mean():.4f}")
    print(f"Baseline val RMSE       : {baseline['validation']['clipped_rmse']:.4f}")
    print(f"Baseline test RMSE      : {baseline['test']['clipped_rmse']:.4f}")
    print(f"Sample weight mode      : {SAMPLE_WEIGHT_MODE}")
    print(f"Eval seeds              : {eval_seeds}")
    print(f"Base config             : {BASE_CONFIG}")
    print("=" * 80)

    best_trial, best_data_bundle, results_df = run_hyperparameter_tuning(
        train_df_raw=train_df,
        val_df_raw=val_df,
        test_df_raw=test_df,
        output_dir=args.output_dir,
        base_config=BASE_CONFIG,
        search_space=SEARCH_SPACE,
        max_trials=args.max_trials,
        random_seed=RANDOM_SEED,
        max_epochs=args.max_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
    )

    multiseed_summary = None
    if MULTI_SEED_EVAL and not args.disable_multi_seed:
        print("\n" + "=" * 80)
        print("Mulai evaluasi multi-seed dengan best config")
        print("=" * 80)
        multiseed_summary, _ = run_multi_seed_evaluation(
            train_df_raw=train_df,
            val_df_raw=val_df,
            test_df_raw=test_df,
            config=best_trial["config"],
            eval_seeds=eval_seeds,
            output_dir=args.output_dir,
            max_epochs=args.max_epochs,
            patience=args.patience,
            min_delta=args.min_delta,
        )

    meta_info = {
        "target_mode": args.target_mode,
        "rebuild_split": True,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "drop_exact_duplicates": DROP_EXACT_DUPLICATES,
        "baseline_metrics": baseline,
        "noise_audit": audit,
        "split_summary": split_summary,
        "multi_seed_eval_enabled": bool(MULTI_SEED_EVAL and not args.disable_multi_seed),
        "eval_seeds": eval_seeds,
        "multi_seed_summary": multiseed_summary,
    }

    evaluate_best_trial_on_test(
        best_trial=best_trial,
        best_data_bundle=best_data_bundle,
        output_dir=args.output_dir,
        data_file=data_file,
        meta_info=meta_info,
    )

if __name__ == "__main__":
    main()
