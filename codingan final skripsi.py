import os
import random
import re
import json
from copy import deepcopy

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# Config

DATA_FILE = "dataset2000.csv"
OUTPUT_DIR = "ann_skripsi"

RANDOM_SEED = 124
TEST_SIZE = 0.15
VAL_SIZE = 0.15
MAX_EPOCHS = 200
PATIENCE = 18
MIN_DELTA = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0 
PIN_MEMORY = torch.cuda.is_available()


MANUAL_MODEL_CONFIG ={
"batch_size": 64,
"learning_rate": 5e-4,
"weight_decay": 1e-4,
"hidden_layers": [128, 64],
"dropout": 0.15,
}
MANUAL_MODEL_CONFIG = None

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function helper

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean_price(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.lower().str.strip()
    s = s.str.replace(r"rp\.?\s*", "", regex=True)
    s = s.str.replace(r"[^0-9,.\-]", "", regex=True)
    s = s.str.replace(".", "", regex=False)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def clean_review(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"[^0-9,.\-]", "", regex=True)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = text.replace("×", " x ")
    text = re.sub(r"[^a-z0-9\s.,/+x-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def first_keyword(text: str, keywords, default="lainnya") -> str:
    for kw in keywords:
        if kw in text:
            return kw
    return default


def extract_number_before_unit(text_series: pd.Series, unit_regex: str) -> pd.Series:
    values = text_series.str.extract(
        rf"(\d+(?:[.,]\d+)?)\s*(?:{unit_regex})\b",
        expand=False,
    )
    values = values.str.replace(",", ".", regex=False)
    return pd.to_numeric(values, errors="coerce").fillna(0)


def extract_box_dimensions_cm(text_series: pd.Series):
    s = text_series.fillna("").astype(str)
    s = s.str.replace(",", ".", regex=False)
    s = s.str.replace("×", " x ", regex=False)
    s = s.str.replace(r"\s+", " ", regex=True)

    dims = s.str.extract(
        r"(\d+(?:\.\d+)?)\s*(?:cm)?\s*x\s*(\d+(?:\.\d+)?)\s*(?:cm)?\s*x\s*(\d+(?:\.\d+)?)\s*cm\b",
        expand=True,
    )

    for col in dims.columns:
        dims[col] = pd.to_numeric(dims[col], errors="coerce")

    panjang_cm = dims[0].fillna(0)
    lebar_cm = dims[1].fillna(0)
    tinggi_cm = dims[2].fillna(0)

    return panjang_cm, lebar_cm, tinggi_cm


def extract_roll_dimensions(text_series: pd.Series):
    s = text_series.fillna("").astype(str)
    s = s.str.replace(",", ".", regex=False)
    s = s.str.replace("×", " x ", regex=False)
    s = s.str.replace(r"\s+", " ", regex=True)

    dims = s.str.extract(
        r"(\d+(?:\.\d+)?)\s*cm\s*x\s*(\d+(?:\.\d+)?)\s*(?:meter|m)\b",
        expand=True,
    )

    for col in dims.columns:
        dims[col] = pd.to_numeric(dims[col], errors="coerce")

    lebar_cm_roll = dims[0].fillna(0)
    panjang_meter_roll = dims[1].fillna(0)

    return lebar_cm_roll, panjang_meter_roll


def extract_yard_dimensions(text_series: pd.Series):
    s = text_series.fillna("").astype(str)
    s = s.str.replace(",", ".", regex=False)
    s = s.str.replace("×", " x ", regex=False)
    s = s.str.replace(r"\s+", " ", regex=True)

    dims = s.str.extract(
        r"(\d+(?:\.\d+)?)\s*(?:mm|cm)?\s*x\s*(\d+(?:\.\d+)?)\s*yard\b",
        expand=True,
    )

    for col in dims.columns:
        dims[col] = pd.to_numeric(dims[col], errors="coerce")

    lebar_pair = dims[0].fillna(0)
    panjang_yard_pair = dims[1].fillna(0)

    return lebar_pair, panjang_yard_pair


def extract_material(text: str) -> str:
    text = str(text).lower()
    keywords = [
        "bflute", "cflute", "kraft", "foam", "thermal",
        "hd", "pe", "opp", "pp"
    ]
    for kw in keywords:
        if re.search(rf"\b{re.escape(kw)}\b", text):
            return kw
    return "lainnya"


def extract_shape_type(text: str) -> str:
    text = str(text).lower()
    keywords = ["roll", "lembar", "sheet", "pack", "set", "pcs", "pc"]
    for kw in keywords:
        if re.search(rf"\b{re.escape(kw)}\b", text):
            return kw
    return "lainnya"


def get_min_category_count(n_rows: int) -> int:
    """
    Untuk data lebih besar, kategori langka digabung lebih agresif
    agar one-hot tidak terlalu meledak.
    """
    if n_rows < 2000:
        return 5
    if n_rows < 3500:
        return 10
    return 15


def build_feature_table(df: pd.DataFrame):
    data = df.copy()

    # Clean column names
    data.columns = data.columns.str.strip().str.replace(" ", "_", regex=False)

    required_cols = ["Nama_Produk", "Kategori_Produk", "Harga_Produk", "Review"]
    missing_cols = [c for c in required_cols if c not in data.columns]
    if missing_cols:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing_cols}")

    # Clean price and review
    data["Harga_Produk"] = clean_price(data["Harga_Produk"])
    data["Review"] = clean_review(data["Review"])

    # Drop invalid rows
    data = data.dropna(subset=["Harga_Produk", "Review"]).copy()

    # Text cleaning
    name_text = data["Nama_Produk"].fillna("").map(normalize_text)
    category_raw = data["Kategori_Produk"].fillna("Tidak diketahui").astype(str).str.strip()

    # Gabungkan kategori yang sangat jarang
    min_category_count = get_min_category_count(len(data))
    category_counts = category_raw.value_counts()
    category_grouped = category_raw.where(
        category_raw.map(category_counts) >= min_category_count,
        "Lainnya"
    )

    # Keyword
    product_keywords = [
        "bubble wrap", "lakban", "kardus", "plastik hd", "plastik pe", "plastik",
        "wrapping", "pe foam", "karung", "thermal", "label", "stiker",
        "cutter", "lem tembak", "lem", "strapping", "dispenser", "staples",
        "kertas", "tas"
    ]

    brand_keywords = [
        "smilepack", "sanpack", "upack", "global", "yatta", "hiro",
        "joyko", "deli", "gmp", "dkm", "non brand", "nonbrand"
    ]

    color_keywords = [
        "bening", "hitam", "coklat", "putih", "biru", "kuning", "merah", "hijau"
    ]

    X = pd.DataFrame(index=data.index)

    
    # Kategorikal
    
    X["kategori_grouped"] = category_grouped
    X["jenis_produk"] = name_text.map(lambda x: first_keyword(x, product_keywords))
    X["merek"] = name_text.map(lambda x: first_keyword(x, brand_keywords))
    X["warna"] = name_text.map(lambda x: first_keyword(x, color_keywords))
    X["material"] = name_text.map(extract_material)
    X["bentuk_kemasan"] = name_text.map(extract_shape_type)

    # Numerik
    
    X["log_harga"] = np.log1p(data["Harga_Produk"])
    X["is_non_brand"] = name_text.str.contains(r"non\s*brand").astype(int)
    X["berat_kg"] = extract_number_before_unit(name_text, "kg")
    X["tebal_mm"] = extract_number_before_unit(name_text, "mm")
    X["tebal_micron"] = extract_number_before_unit(name_text, "micron|mikron")
    X["jumlah_pcs"] = extract_number_before_unit(name_text, "pcs|pc")
    X["jumlah_lembar"] = extract_number_before_unit(name_text, "lembar")
    X["jumlah_roll"] = extract_number_before_unit(name_text, "roll")
    X["jumlah_set"] = extract_number_before_unit(name_text, "set")
    X["ukuran_cm"] = extract_number_before_unit(name_text, "cm")
    X["panjang_meter"] = extract_number_before_unit(name_text, "meter|m")
    X["yard"] = extract_number_before_unit(name_text, "yard")

    # Detail dimensi
    
    box_panjang_cm, box_lebar_cm, box_tinggi_cm = extract_box_dimensions_cm(name_text)
    X["box_panjang_cm"] = box_panjang_cm
    X["box_lebar_cm"] = box_lebar_cm
    X["box_tinggi_cm"] = box_tinggi_cm
    X["log_volume_box_cm3"] = np.log1p(
        X["box_panjang_cm"] * X["box_lebar_cm"] * X["box_tinggi_cm"]
    )

    lebar_cm_roll, panjang_meter_roll = extract_roll_dimensions(name_text)
    X["lebar_cm_roll"] = lebar_cm_roll
    X["panjang_meter_roll"] = panjang_meter_roll

    lebar_pair_yard, panjang_yard_pair = extract_yard_dimensions(name_text)
    X["lebar_pair_yard"] = lebar_pair_yard
    X["panjang_yard_pair"] = panjang_yard_pair

    # Flag teks

    X["is_bflute"] = name_text.str.contains(r"\bbflute\b").astype(int)
    X["is_cflute"] = name_text.str.contains(r"\bcflute\b").astype(int)
    X["is_foam"] = name_text.str.contains(r"\bfoam\b").astype(int)
    X["is_kraft"] = name_text.str.contains(r"\bkraft\b").astype(int)
    X["is_roll"] = name_text.str.contains(r"\broll\b").astype(int)


    # Teks tambahan

    X["jumlah_dimensi_x"] = name_text.str.count(r"\bx\b")
    X["panjang_nama"] = name_text.str.len()
    X["jumlah_token"] = name_text.str.split().str.len().fillna(0)

    y = data["Review"].astype(float).values

    return X, y, data



# Preprocessing function

def to_dense_numpy(X):
    if hasattr(X, "toarray"):
        return X.toarray()
    return X


def build_preprocessor(X_train: pd.DataFrame):
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )

    return preprocessor


def make_stratify_labels(y: np.ndarray):
    """
    Stratify berdasarkan rating bulat terdekat agar distribusi train/val/test lebih stabil.
    Jika jumlah per kelas terlalu sedikit, fallback ke None.
    """
    labels = np.clip(np.rint(y), 1, 5).astype(int).astype(str)
    value_counts = pd.Series(labels).value_counts()
    if len(value_counts) < 2 or value_counts.min() < 2:
        return None
    return labels


def choose_model_config(n_train_rows: int):
    """
    Konfigurasi otomatis untuk data 2.000 - 5.000 baris.
    Anda masih bisa override lewat MANUAL_MODEL_CONFIG.
    """
    if MANUAL_MODEL_CONFIG is not None:
        return MANUAL_MODEL_CONFIG

    if n_train_rows < 2500:
        return {
            "batch_size": 32,
            "learning_rate": 5e-4,
            "weight_decay": 1e-4,
            "hidden_layers": [128, 64],
            "dropout": 0.15,
        }
    elif n_train_rows < 4000:
        return {
            "batch_size": 64,
            "learning_rate": 4e-4,
            "weight_decay": 1e-4,
            "hidden_layers": [256, 128, 64],
            "dropout": 0.20,
        }
    else:
        return {
            "batch_size": 64,
            "learning_rate": 3e-4,
            "weight_decay": 2e-4,
            "hidden_layers": [256, 128, 64],
            "dropout": 0.20,
        }


# Ann

class ANNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout):
        super().__init__()

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
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



# Metrics helper

def inverse_transform_target(y_scaled, y_scaler):
    return y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()


def compute_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


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

    preds = np.clip(preds, 1.0, 5.0)
    rmse = compute_rmse(targets, preds)

    return avg_loss, rmse, preds, targets



# Training

def train_model(model, train_loader, val_loader, y_scaler, learning_rate, weight_decay):
    criterion = nn.MSELoss()
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
    best_epoch = 0
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_rmse": [],
        "val_rmse": [],
        "learning_rate": [],
    }

    model.to(DEVICE)

    for epoch in range(1, MAX_EPOCHS + 1):
        
        # Train
        
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
        train_preds = np.clip(train_preds, 1.0, 5.0)

        train_rmse = compute_rmse(train_targets, train_preds)

        
        # Validation
        
        val_loss, val_rmse, _, _ = evaluate_loader(model, val_loader, criterion, y_scaler)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_rmse"].append(train_rmse)
        history["val_rmse"].append(val_rmse)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])

        scheduler.step(val_rmse)

        if val_rmse < best_val_rmse - MIN_DELTA:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:03d} | "
                f"Train RMSE: {train_rmse:.4f} | "
                f"Val RMSE: {val_rmse:.4f} | "
                f"LR: {current_lr:.6f}"
            )

        if patience_counter >= PATIENCE:
            print(f"Early stopping pada epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model, history, best_val_rmse, best_epoch

# Plotting

def plot_training_history(history, output_dir):
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
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_history.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    return plot_path


# =========================================================
# MAIN
# =========================================================
def main():
    seed_everything(RANDOM_SEED)

    # Load and build features
    df = pd.read_csv(DATA_FILE)
    X, y, clean_data = build_feature_table(df)

    stratify_labels = make_stratify_labels(y)

    # Split data: train_full / test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=stratify_labels if stratify_labels is not None else None,
    )

    stratify_train = make_stratify_labels(y_train_full)

    # Split data: train / val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=VAL_SIZE / (1 - TEST_SIZE),
        random_state=RANDOM_SEED,
        stratify=stratify_train if stratify_train is not None else None,
    )

    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    # Konfigurasi model adaptif terhadap ukuran data train
    model_config = choose_model_config(len(X_train))
    batch_size = model_config["batch_size"]
    learning_rate = model_config["learning_rate"]
    weight_decay = model_config["weight_decay"]
    hidden_layers = model_config["hidden_layers"]
    dropout = model_config["dropout"]

    print("=" * 60)
    print("Ringkasan Konfigurasi Training")
    print("=" * 60)
    print(f"Total data valid       : {len(clean_data)}")
    print(f"Train / Val / Test     : {len(X_train)} / {len(X_val)} / {len(X_test)}")
    print(f"Input feature mentah   : {X.shape[1]}")
    print(f"Hidden layers          : {hidden_layers}")
    print(f"Dropout                : {dropout}")
    print(f"Batch size             : {batch_size}")
    print(f"Learning rate          : {learning_rate}")
    print(f"Weight decay           : {weight_decay}")
    print(f"Max epochs             : {MAX_EPOCHS}")
    print(f"Patience               : {PATIENCE}")
    print(f"Device                 : {DEVICE}")
    print("=" * 60)

    # Preprocessing
    preprocessor = build_preprocessor(X_train)

    X_train_processed = to_dense_numpy(preprocessor.fit_transform(X_train)).astype(np.float32)
    X_val_processed = to_dense_numpy(preprocessor.transform(X_val)).astype(np.float32)
    X_test_processed = to_dense_numpy(preprocessor.transform(X_test)).astype(np.float32)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).astype(np.float32)
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).astype(np.float32)
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).astype(np.float32)

    # Create DataLoaders
    train_dataset = TensorDataset(
        torch.from_numpy(X_train_processed),
        torch.from_numpy(y_train_scaled),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val_processed),
        torch.from_numpy(y_val_scaled),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test_processed),
        torch.from_numpy(y_test_scaled),
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
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # Initialize and train model
    input_dim = X_train_processed.shape[1]
    print(f"Jumlah fitur setelah preprocessing: {input_dim}")

    model = ANNRegressor(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        dropout=dropout,
    )

    model, history, best_val_rmse, best_epoch = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        y_scaler=y_scaler,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    print(f"\nBest Validation RMSE: {best_val_rmse:.4f}")
    print(f"Best Epoch          : {best_epoch}\n")

    # Evaluate on test set
    criterion = nn.MSELoss()
    test_loss, test_rmse, test_preds, test_targets = evaluate_loader(
        model,
        test_loader,
        criterion,
        y_scaler,
    )

    test_mae = mean_absolute_error(test_targets, test_preds)
    test_r2 = r2_score(test_targets, test_preds)

    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE : {test_mae:.4f}")
    print(f"Test R2  : {test_r2:.4f}")

    # Save outputs
    plot_path = plot_training_history(history, OUTPUT_DIR)

    metrics = {
        "n_total": int(len(clean_data)),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "input_dim_after_preprocessing": int(input_dim),
        "best_epoch": int(best_epoch),
        "best_val_rmse": float(best_val_rmse),
        "test_loss": float(test_loss),
        "test_rmse": float(test_rmse),
        "test_mae": float(test_mae),
        "test_r2": float(test_r2),
        "model_config": model_config,
    }

    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    predictions_df = pd.DataFrame({
        "actual_review": test_targets,
        "predicted_review": test_preds,
        "abs_error": np.abs(test_targets - test_preds),
    })
    predictions_path = os.path.join(OUTPUT_DIR, "test_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)

    model_path = os.path.join(OUTPUT_DIR, "best_ann_model.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "model_config": model_config,
            "best_epoch": best_epoch,
            "best_val_rmse": best_val_rmse,
        },
        model_path,
    )

    print("\nFile output tersimpan:")
    print(f"- Plot history      : {plot_path}")
    print(f"- Metrics           : {metrics_path}")
    print(f"- Prediksi test     : {predictions_path}")
    print(f"- Bobot model       : {model_path}")


if __name__ == "__main__":
    main()