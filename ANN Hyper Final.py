import os
import random
import re
import json
from copy import deepcopy

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# =========================================================
# CONFIG
# =========================================================
DATA_FILE = "dataset5000ready.csv"
OUTPUT_DIR = "ann_hyper_skripsi"

RANDOM_SEED = 124
MAX_EPOCHS = 200
PATIENCE = 20
MIN_DELTA = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()

# INI yang sebelumnya belum ada:
N_TRIALS = 5

SAFE_HPARAMETER_RANGE = {
    "batch_size": [32, 48, 64],
    "learning_rate": [3e-4, 5e-4, 7e-4, 1e-3],
    "weight_decay": [1e-6, 5e-6, 1e-5, 2e-5, 5e-5],
    "hidden_layers": [
        [256, 128],
        [320, 160],
        [384, 192, 96],
        [448, 224, 112],
        [512, 256, 128],
    ],
    "dropout": [0.05, 0.10, 0.15, 0.20],
    "loss_name": ["huber", "mse"],
    "use_batchnorm": [True],
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


def standardize_colname(col: str) -> str:
    col = str(col).strip().lower()
    col = re.sub(r"[^a-z0-9]+", "_", col)
    col = re.sub(r"_+", "_", col).strip("_")
    return col


def to_dense_numpy(X):
    if hasattr(X, "toarray"):
        return X.toarray()
    return X


def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def clip_rating_predictions(preds: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(preds, dtype=float), 1.0, 5.0)


def compute_rmse(y_true, y_pred):
    y_pred = clip_rating_predictions(y_pred)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_mae(y_true, y_pred):
    y_pred = clip_rating_predictions(y_pred)
    return float(mean_absolute_error(y_true, y_pred))


def inverse_transform_target(y_scaled, y_scaler):
    return y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()


def write_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def make_criterion(loss_name: str):
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "huber":
        return nn.SmoothL1Loss(beta=0.25)
    raise ValueError(f"loss_name tidak dikenali: {loss_name}")


def sample_random_config(rng):
    hidden_layer_options = SAFE_HPARAMETER_RANGE["hidden_layers"]
    hidden_layers = list(
        hidden_layer_options[int(rng.integers(0, len(hidden_layer_options)))]
    )

    return {
        "batch_size": int(rng.choice(SAFE_HPARAMETER_RANGE["batch_size"])),
        "learning_rate": float(rng.choice(SAFE_HPARAMETER_RANGE["learning_rate"])),
        "weight_decay": float(rng.choice(SAFE_HPARAMETER_RANGE["weight_decay"])),
        "hidden_layers": hidden_layers,
        "dropout": float(rng.choice(SAFE_HPARAMETER_RANGE["dropout"])),
        "loss_name": str(rng.choice(SAFE_HPARAMETER_RANGE["loss_name"])),
        "use_batchnorm": bool(rng.choice(SAFE_HPARAMETER_RANGE["use_batchnorm"])),
    }


# =========================================================
# DATASET PREP SESUAI DATASET 5000 READY
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

    return df


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


def build_feature_table_from_processed(df: pd.DataFrame):
    data = add_indicator_features(df.copy())
    data = data.drop_duplicates().reset_index(drop=True)

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

    categorical_features = [
        "kategori_produk",
        "jenis_produk",
        "merek",
        "warna",
        "material",
        "bentuk_kemasan",
        "family_template",
    ]

    for col in numeric_features:
        if col not in data.columns:
            data[col] = 0.0

    for col in categorical_features:
        if col not in data.columns:
            data[col] = "lainnya"

    X = data[numeric_features + categorical_features].copy()
    y = pd.to_numeric(data["review"], errors="coerce").astype(float).to_numpy()

    valid_mask = np.isfinite(y)
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y[valid_mask]
    data = data.loc[valid_mask].reset_index(drop=True)

    return X, y, data


def split_by_split_grouped(df: pd.DataFrame):
    train_df = df.loc[df["split_grouped"] == "train"].copy()
    val_df = df.loc[df["split_grouped"] == "validation"].copy()
    test_df = df.loc[df["split_grouped"] == "test"].copy()

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("Kolom split_grouped harus berisi train, validation, dan test.")

    return train_df, val_df, test_df


# =========================================================
# PREPROCESSOR
# =========================================================
def build_preprocessor(X_train: pd.DataFrame):
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_onehot_encoder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )

    return preprocessor


# =========================================================
# ANN MODEL
# =========================================================
class ANNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout, use_batchnorm=True):
        super().__init__()

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

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


# =========================================================
# EVALUASI
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
            preds_scaled.append(pred.cpu().numpy().ravel())
            targets_scaled.append(yb.cpu().numpy().ravel())

    avg_loss = loss_sum / len(loader.dataset)

    preds_scaled = np.concatenate(preds_scaled)
    targets_scaled = np.concatenate(targets_scaled)

    preds = inverse_transform_target(preds_scaled, y_scaler)
    targets = inverse_transform_target(targets_scaled, y_scaler)

    preds = clip_rating_predictions(preds)
    rmse = compute_rmse(targets, preds)
    mae = compute_mae(targets, preds)

    return avg_loss, rmse, mae, preds, targets


# =========================================================
# TRAINING
# =========================================================
def train_model(model, train_loader, val_loader, y_scaler, learning_rate, weight_decay, loss_name):
    criterion = make_criterion(loss_name)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    best_state = deepcopy(model.state_dict())
    best_val_rmse = float("inf")
    best_val_mae = float("inf")
    best_epoch = 0
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_rmse": [],
        "val_rmse": [],
        "train_mae": [],
        "val_mae": [],
        "learning_rate": [],
    }

    model.to(DEVICE)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
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

            train_loss_sum += loss.item() * len(xb)
            train_preds_scaled.append(pred.detach().cpu().numpy().ravel())
            train_targets_scaled.append(yb.detach().cpu().numpy().ravel())

        train_loss = train_loss_sum / len(train_loader.dataset)

        train_preds_scaled = np.concatenate(train_preds_scaled)
        train_targets_scaled = np.concatenate(train_targets_scaled)

        train_preds = inverse_transform_target(train_preds_scaled, y_scaler)
        train_targets = inverse_transform_target(train_targets_scaled, y_scaler)
        train_preds = clip_rating_predictions(train_preds)

        train_rmse = compute_rmse(train_targets, train_preds)
        train_mae = compute_mae(train_targets, train_preds)

        val_loss, val_rmse, val_mae, _, _ = evaluate_loader(model, val_loader, criterion, y_scaler)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_rmse"].append(train_rmse)
        history["val_rmse"].append(val_rmse)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])

        scheduler.step(val_rmse)

        improved = False
        if val_rmse < best_val_rmse - MIN_DELTA:
            improved = True
        elif abs(val_rmse - best_val_rmse) <= MIN_DELTA and val_mae < best_val_mae:
            improved = True

        if improved:
            best_val_rmse = val_rmse
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d} | "
            f"Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} | "
            f"Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f} | "
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
    plt.plot(history["train_rmse"], label="Train RMSE")
    plt.plot(history["val_rmse"], label="Validation RMSE")
    plt.title("RMSE over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history["train_mae"], label="Train MAE")
    plt.plot(history["val_mae"], label="Validation MAE")
    plt.title("MAE over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
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

    X_train, y_train, clean_train = build_feature_table_from_processed(train_df)
    X_val, y_val, clean_val = build_feature_table_from_processed(val_df)
    X_test, y_test, clean_test = build_feature_table_from_processed(test_df)

    print("=" * 70)
    print("Ringkasan Dataset")
    print("=" * 70)
    print(f"Data file               : {DATA_FILE}")
    print(f"Train / Val / Test      : {len(X_train)} / {len(X_val)} / {len(X_test)}")
    print(f"Jumlah trial            : {N_TRIALS}")
    print("=" * 70)

    # preprocessing satu kali saja
    preprocessor = build_preprocessor(X_train)

    X_train_processed = to_dense_numpy(preprocessor.fit_transform(X_train)).astype(np.float32)
    X_val_processed = to_dense_numpy(preprocessor.transform(X_val)).astype(np.float32)
    X_test_processed = to_dense_numpy(preprocessor.transform(X_test)).astype(np.float32)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).astype(np.float32)
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).astype(np.float32)
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).astype(np.float32)

    input_dim = X_train_processed.shape[1]
    print(f"Jumlah fitur setelah preprocessing: {input_dim}")

    rng = np.random.default_rng(RANDOM_SEED)
    trial_rows = []

    best_model = None
    best_config = None
    best_history = None
    best_epoch = 0
    best_val_rmse = float("inf")
    best_val_mae = float("inf")

    for trial in range(1, N_TRIALS + 1):
        config = sample_random_config(rng)

        print("\n" + "-" * 70)
        print(f"TRIAL {trial}/{N_TRIALS}")
        print(config)
        print("-" * 70)

        batch_size = config["batch_size"]

        train_dataset = TensorDataset(
            torch.from_numpy(X_train_processed),
            torch.from_numpy(y_train_scaled),
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_val_processed),
            torch.from_numpy(y_val_scaled),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )

        model = ANNRegressor(
            input_dim=input_dim,
            hidden_layers=config["hidden_layers"],
            dropout=config["dropout"],
            use_batchnorm=config["use_batchnorm"],
        )

        model, history, trial_val_rmse, trial_val_mae, trial_best_epoch = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            y_scaler=y_scaler,
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            loss_name=config["loss_name"],
        )

        trial_rows.append({
            "trial": trial,
            "batch_size": config["batch_size"],
            "learning_rate": config["learning_rate"],
            "weight_decay": config["weight_decay"],
            "hidden_layers": str(config["hidden_layers"]),
            "dropout": config["dropout"],
            "loss_name": config["loss_name"],
            "use_batchnorm": config["use_batchnorm"],
            "best_epoch": trial_best_epoch,
            "val_rmse": trial_val_rmse,
            "val_mae": trial_val_mae,
        })

        improved = False
        if trial_val_rmse < best_val_rmse - MIN_DELTA:
            improved = True
        elif abs(trial_val_rmse - best_val_rmse) <= MIN_DELTA and trial_val_mae < best_val_mae:
            improved = True

        if improved:
            best_model = deepcopy(model)
            best_config = config
            best_history = history
            best_epoch = trial_best_epoch
            best_val_rmse = trial_val_rmse
            best_val_mae = trial_val_mae

    results_df = pd.DataFrame(trial_rows).sort_values(
        by=["val_rmse", "val_mae"], ascending=[True, True]
    ).reset_index(drop=True)
    results_path = os.path.join(OUTPUT_DIR, "trial_results.csv")
    results_df.to_csv(results_path, index=False)

    print("\n" + "=" * 70)
    print("BEST CONFIG")
    print("=" * 70)
    print(best_config)
    print(f"Best Validation RMSE: {best_val_rmse:.4f}")
    print(f"Best Validation MAE : {best_val_mae:.4f}")
    print(f"Best Epoch          : {best_epoch}")

    # test pakai model terbaik
    test_dataset = TensorDataset(
        torch.from_numpy(X_test_processed),
        torch.from_numpy(y_test_scaled),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=best_config["batch_size"],
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    criterion = make_criterion(best_config["loss_name"])
    test_loss, test_rmse, test_mae, test_preds, test_targets = evaluate_loader(
        best_model,
        test_loader,
        criterion,
        y_scaler,
    )

    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE : {test_mae:.4f}")

    plot_path = plot_training_history(best_history, OUTPUT_DIR, prefix="best_trial")

    metrics = {
        "data_file": DATA_FILE,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "input_dim_after_preprocessing": int(input_dim),
        "n_trials": int(N_TRIALS),
        "best_epoch": int(best_epoch),
        "best_val_rmse": float(best_val_rmse),
        "best_val_mae": float(best_val_mae),
        "test_loss": float(test_loss),
        "test_rmse": float(test_rmse),
        "test_mae": float(test_mae),
        "best_model_config": best_config,
        "safe_hparameter_range": SAFE_HPARAMETER_RANGE,
    }

    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    write_json(metrics, metrics_path)

    predictions_df = pd.DataFrame({
        "actual_review": test_targets,
        "predicted_review": clip_rating_predictions(test_preds),
        "abs_error": np.abs(test_targets - clip_rating_predictions(test_preds)),
    })
    predictions_path = os.path.join(OUTPUT_DIR, "test_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)

    model_path = os.path.join(OUTPUT_DIR, "best_ann_model.pth")
    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "input_dim": input_dim,
            "best_model_config": best_config,
            "best_epoch": best_epoch,
            "best_val_rmse": best_val_rmse,
            "best_val_mae": best_val_mae,
        },
        model_path,
    )

    print("\nFile output tersimpan:")
    print(f"- Hasil semua trial  : {results_path}")
    print(f"- Plot history       : {plot_path}")
    print(f"- Metrics            : {metrics_path}")
    print(f"- Prediksi test      : {predictions_path}")
    print(f"- Bobot model        : {model_path}")


if __name__ == "__main__":
    main()
