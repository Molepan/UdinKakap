import os
import random
import re
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
import json
import pickle

# =========================================================
# CONFIGURATION
# =========================================================
DATA_FILE = "dataset2000.csv"
OUTPUT_DIR = "runs_ann_produk"

RANDOM_SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
BATCH_SIZE = 32
EPOCHS = 300
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 30
HIDDEN_LAYERS = [64, 32, 16]  # Optimized hidden layers
DROPOUT = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clean_price(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(".", "", regex=False)   # Format: 120.000
    s = s.str.replace(",", ".", regex=False)  # Format: 120,5
    return pd.to_numeric(s, errors="coerce")

def clean_review(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s.,/x-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def first_keyword(text: str, keywords, default="lainnya") -> str:
    for kw in keywords:
        if kw in text:
            return kw
    return default

def extract_number_before_unit(text_series: pd.Series, unit_regex: str) -> pd.Series:
    values = text_series.str.extract(rf"(\d+(?:[.,]\d+)?)\s*{unit_regex}", expand=False)
    values = values.str.replace(",", ".", regex=False)
    return pd.to_numeric(values, errors="coerce").fillna(0)

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

    # Drop rows with invalid data
    data = data.dropna(subset=["Harga_Produk", "Review"]).copy()

    # Clean text fields
    name_text = data["Nama_Produk"].fillna("").map(normalize_text)
    category_raw = data["Kategori_Produk"].fillna("Tidak diketahui").astype(str).str.strip()

    # Gabungkan kategori yang sangat jarang
    category_counts = category_raw.value_counts()
    category_grouped = category_raw.where(
        category_raw.map(category_counts) >= 5,
        "Lainnya"
    )

    # Keyword sederhana agar tidak overfitting seperti TF-IDF penuh
    product_keywords = [
        "bubble wrap", "lakban", "kardus", "plastik hd", "plastik pe", "plastik",
        "wrapping", "pe foam", "karung", "thermal", "label", "stiker",
        "cutter", "lem tembak", "lem", "strapping", "dispenser", "staples",
        "kertas", "tas"
    ]
    brand_keywords = [
        "smilepack", "sanpack", "upack", "global", "yatta", "hiro",
        "joyko", "deli", "non brand", "nonbrand"
    ]
    color_keywords = [
        "bening", "hitam", "coklat", "putih", "biru", "kuning", "merah", "hijau"
    ]

    X = pd.DataFrame(index=data.index)

    # Fitur kategorikal
    X["kategori_grouped"] = category_grouped
    X["jenis_produk"] = name_text.map(lambda x: first_keyword(x, product_keywords))
    X["merek"] = name_text.map(lambda x: first_keyword(x, brand_keywords))
    X["warna"] = name_text.map(lambda x: first_keyword(x, color_keywords))

    # Fitur numerik
    X["log_harga"] = np.log1p(data["Harga_Produk"])
    X["is_non_brand"] = name_text.str.contains(r"non\s*brand").astype(int)
    X["berat_kg"] = extract_number_before_unit(name_text, "kg")
    X["ukuran_cm"] = extract_number_before_unit(name_text, "cm")
    X["panjang_meter"] = extract_number_before_unit(name_text, "meter")
    X["tebal_mm"] = extract_number_before_unit(name_text, "mm")
    X["yard"] = extract_number_before_unit(name_text, "yard")
    X["jumlah_pcs"] = extract_number_before_unit(name_text, "pcs|pc")
    X["jumlah_lembar"] = extract_number_before_unit(name_text, "lembar")
    X["jumlah_dimensi_x"] = name_text.str.count(r"\bx\b")
    X["panjang_nama"] = name_text.str.len()
    X["jumlah_token"] = name_text.str.split().str.len().fillna(0)

    y = data["Review"].astype(float).values

    return X, y, data

# =========================================================
# TO DENSE NUMPY FUNCTION
# =========================================================
def to_dense_numpy(X):
    """Mengonversi hasil transformasi sparse ke dalam format NumPy dense."""
    if hasattr(X, "toarray"):
        return X.toarray()
    return X

# =========================================================
# ANN MODEL
# =========================================================
class ANNRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        layers = []
        current_dim = input_dim

        for hidden_dim in HIDDEN_LAYERS:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))  # Dropout
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# =========================================================
# TRAINING
# =========================================================
def train_model(model, train_loader, val_loader, y_scaler):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8
    )

    best_state = deepcopy(model.state_dict())
    best_val_rmse = float("inf")
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_rmse": [],
        "val_rmse": []
    }

    model.to(DEVICE)

    for epoch in range(1, EPOCHS + 1):
        # -------------------------
        # TRAIN
        # -------------------------
        model.train()
        train_loss_sum = 0.0
        train_preds_scaled = []
        train_targets_scaled = []

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(xb)
            train_preds_scaled.append(pred.detach().cpu().numpy().ravel())
            train_targets_scaled.append(yb.detach().cpu().numpy().ravel())

        train_loss = train_loss_sum / len(train_loader.dataset)

        train_preds_scaled = np.concatenate(train_preds_scaled)
        train_targets_scaled = np.concatenate(train_targets_scaled)

        train_preds = y_scaler.inverse_transform(train_preds_scaled.reshape(-1, 1)).ravel()
        train_targets = y_scaler.inverse_transform(train_targets_scaled.reshape(-1, 1)).ravel()

        train_rmse = np.sqrt(mean_squared_error(train_targets, np.clip(train_preds, 1.0, 5.0)))

        # -------------------------
        # VALIDATION
        # -------------------------
        model.eval()
        val_loss_sum = 0.0
        val_preds_scaled = []
        val_targets_scaled = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)

                pred = model(xb)
                loss = criterion(pred, yb)

                val_loss_sum += loss.item() * len(xb)
                val_preds_scaled.append(pred.cpu().numpy().ravel())
                val_targets_scaled.append(yb.cpu().numpy().ravel())

        val_loss = val_loss_sum / len(val_loader.dataset)

        val_preds_scaled = np.concatenate(val_preds_scaled)
        val_targets_scaled = np.concatenate(val_targets_scaled)

        val_preds = y_scaler.inverse_transform(val_preds_scaled.reshape(-1, 1)).ravel()
        val_targets = y_scaler.inverse_transform(val_targets_scaled.reshape(-1, 1)).ravel()

        val_rmse = np.sqrt(mean_squared_error(val_targets, np.clip(val_preds, 1.0, 5.0)))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_rmse"].append(train_rmse)
        history["val_rmse"].append(val_rmse)

        scheduler.step(val_rmse)

        if val_rmse < best_val_rmse - 1e-5:
            best_val_rmse = val_rmse
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 20 == 0 or epoch == 1:
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
    return model, history, best_val_rmse

# =========================================================
# PREDICT MODEL
# =========================================================
def predict_model(model, X_np, y_scaler):
    """Fungsi untuk melakukan prediksi pada data input (X_np)"""
    model.eval()
    with torch.no_grad():  # Menonaktifkan perhitungan gradien untuk efisiensi
        X_tensor = torch.tensor(X_np, dtype=torch.float32).to(DEVICE)
        pred_scaled = model(X_tensor).cpu().numpy().ravel()

    # Menggunakan scaler untuk mengembalikan hasil prediksi ke skala asli
    pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    pred = np.clip(pred, 1.0, 5.0)  # Membatasi hasil prediksi pada rentang tertentu
    return pred

# =========================================================
# SAVE BEST MODEL
# =========================================================
def save_best_model(model, output_dir="models"):
    """Menyimpan model terbaik ke file .pth"""
    os.makedirs(output_dir, exist_ok=True)
    model_save_path = os.path.join(output_dir, "best_model_ann.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model disimpan di {model_save_path}")

# =========================================================
# SAVE CONFIGS AND RESULTS
# =========================================================
def save_configs(best_params, history, output_dir="models"):
    """Menyimpan hasil hyperparameter tuning dan evaluasi"""
    best_config_path = os.path.join(output_dir, "best_tuning_config.json")
    with open(best_config_path, 'w') as f:
        json.dump(best_params, f)

    # Save training and validation history to JSON
    history_path = os.path.join(output_dir, "metric.json")
    with open(history_path, 'w') as f:
        json.dump(history, f)

    print(f"Configs and metrics saved to {output_dir}")

# =========================================================
# MAIN EXECUTION
# =========================================================
def main():
    seed_everything(RANDOM_SEED)

    # 1. Load and process data
    df = pd.read_csv(DATA_FILE)

    # 2. Build features and targets
    X_df, y, cleaned_df = build_feature_table(df)

    # 3. Split the data into train/val/test
    X_train_full_df, X_test_df, y_train_full, y_test = train_test_split(
        X_df, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    relative_val_size = VAL_SIZE / (1 - TEST_SIZE)
    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_train_full_df, y_train_full, test_size=relative_val_size, random_state=RANDOM_SEED
    )

    # 4. Preprocessing
    numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_df.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols)
        ]
    )

    # 5. Feature scaling and processing
    X_train = to_dense_numpy(preprocessor.fit_transform(X_train_df))
    X_val = to_dense_numpy(preprocessor.transform(X_val_df))
    X_test = to_dense_numpy(preprocessor.transform(X_test_df))

    # Scale the target
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).ravel()

    # 6. Create DataLoader
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train_scaled, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val_scaled, dtype=torch.float32)), batch_size=BATCH_SIZE, shuffle=False)

    # 7. Build the ANN model
    input_dim = X_train.shape[1]
    model = ANNRegressor(input_dim)

    # 8. Train the model
    model, history, best_val_rmse = train_model(model, train_loader, val_loader, y_scaler)

    # 9. Save Best Model and Configurations
    save_best_model(model, OUTPUT_DIR)
    save_configs(best_params={}, history=history, output_dir=OUTPUT_DIR)

    # 10. Evaluate and save results
    y_pred_test = predict_model(model, X_test, y_scaler)
    ann_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    metrics_df = pd.DataFrame([{
        "model": "ANN",
        "best_val_rmse": best_val_rmse,
        "test_rmse": ann_rmse
    }])
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "evaluation.csv"), index=False)

    # 11. Visualizations
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_rmse"], label="Train RMSE")
    plt.plot(history["val_rmse"], label="Validation RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_test, alpha=0.7)
    plt.plot([1, 5], [1, 5], color="red", linestyle="--")
    plt.xlabel("Actual Review")
    plt.ylabel("Predicted Review")
    plt.title("Actual vs Predicted (Test Set)")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "refit_training.png"), dpi=200)
    plt.show()

if __name__ == "__main__":
    main()