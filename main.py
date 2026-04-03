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

# =========================================================
# CONFIGURATION
# =========================================================
DATA_FILE = "dataset2.csv"
OUTPUT_DIR = "runs_ann_produk"

RANDOM_SEED = 124
TEST_SIZE = 0.15
VAL_SIZE = 0.15
BATCH_SIZE = 8
EPOCHS = 105
LEARNING_RATE = 9e-5
WEIGHT_DECAY = 1e-4
PATIENCE = 15
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
    s = series.astype(str).str.lower().str.strip()
    s = s.str.replace(r"rp\.?\s*", "", regex=True)
    s = s.str.replace(r"[^0-9,.\-]", "", regex=True)
    s = s.str.replace(".", "", regex=False)   # contoh: 120.000
    s = s.str.replace(",", ".", regex=False)  # contoh: 120,5
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
        expand=False
    )
    values = values.str.replace(",", ".", regex=False)
    return pd.to_numeric(values, errors="coerce").fillna(0)


def extract_box_dimensions_cm(text_series: pd.Series):
    """
    Mengekstrak dimensi pola kardus seperti:
    34 cm x 19 cm x 9 cm
    34 x 19 x 9 cm
    """
    s = text_series.fillna("").astype(str)
    s = s.str.replace(",", ".", regex=False)
    s = s.str.replace("×", " x ", regex=False)
    s = s.str.replace(r"\s+", " ", regex=True)

    dims = s.str.extract(
        r"(\d+(?:\.\d+)?)\s*(?:cm)?\s*x\s*(\d+(?:\.\d+)?)\s*(?:cm)?\s*x\s*(\d+(?:\.\d+)?)\s*cm\b",
        expand=True
    )

    for col in dims.columns:
        dims[col] = pd.to_numeric(dims[col], errors="coerce")

    panjang_cm = dims[0].fillna(0)
    lebar_cm = dims[1].fillna(0)
    tinggi_cm = dims[2].fillna(0)

    return panjang_cm, lebar_cm, tinggi_cm


def extract_roll_dimensions(text_series: pd.Series):
    """
    Mengekstrak pola roll seperti:
    125 cm x 50 meter
    """
    s = text_series.fillna("").astype(str)
    s = s.str.replace(",", ".", regex=False)
    s = s.str.replace("×", " x ", regex=False)
    s = s.str.replace(r"\s+", " ", regex=True)

    dims = s.str.extract(
        r"(\d+(?:\.\d+)?)\s*cm\s*x\s*(\d+(?:\.\d+)?)\s*(?:meter|m)\b",
        expand=True
    )

    for col in dims.columns:
        dims[col] = pd.to_numeric(dims[col], errors="coerce")

    lebar_cm_roll = dims[0].fillna(0)
    panjang_meter_roll = dims[1].fillna(0)

    return lebar_cm_roll, panjang_meter_roll


def extract_yard_dimensions(text_series: pd.Series):
    """
    Mengekstrak pola lakban seperti:
    48 x 85 yard
    48 mm x 85 yard
    """
    s = text_series.fillna("").astype(str)
    s = s.str.replace(",", ".", regex=False)
    s = s.str.replace("×", " x ", regex=False)
    s = s.str.replace(r"\s+", " ", regex=True)

    dims = s.str.extract(
        r"(\d+(?:\.\d+)?)\s*(?:mm|cm)?\s*x\s*(\d+(?:\.\d+)?)\s*yard\b",
        expand=True
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
    category_counts = category_raw.value_counts()
    category_grouped = category_raw.where(
        category_raw.map(category_counts) >= 5,
        "Lainnya"
    )

    # Keyword dasar
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

    # -------------------------
    # Fitur kategorikal
    # -------------------------
    X["kategori_grouped"] = category_grouped
    X["jenis_produk"] = name_text.map(lambda x: first_keyword(x, product_keywords))
    X["merek"] = name_text.map(lambda x: first_keyword(x, brand_keywords))
    X["warna"] = name_text.map(lambda x: first_keyword(x, color_keywords))
    X["material"] = name_text.map(extract_material)
    X["bentuk_kemasan"] = name_text.map(extract_shape_type)

    # -------------------------
    # Fitur numerik dasar
    # -------------------------
    X["log_harga"] = np.log1p(data["Harga_Produk"])
    X["is_non_brand"] = name_text.str.contains(r"non\s*brand").astype(int)
    X["berat_kg"] = extract_number_before_unit(name_text, "kg")
    X["tebal_mm"] = extract_number_before_unit(name_text, "mm")
    X["tebal_micron"] = extract_number_before_unit(name_text, "micron|mikron")
    X["jumlah_pcs"] = extract_number_before_unit(name_text, "pcs|pc")
    X["jumlah_lembar"] = extract_number_before_unit(name_text, "lembar")
    X["jumlah_roll"] = extract_number_before_unit(name_text, "roll")
    X["jumlah_set"] = extract_number_before_unit(name_text, "set")

    # Tetap dipertahankan agar kompatibel dengan struktur lama
    X["ukuran_cm"] = extract_number_before_unit(name_text, "cm")
    X["panjang_meter"] = extract_number_before_unit(name_text, "meter|m")
    X["yard"] = extract_number_before_unit(name_text, "yard")

    # -------------------------
    # Fitur dimensi yang lebih detail
    # -------------------------
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

    # -------------------------
    # Flag fitur teks
    # -------------------------
    X["is_bflute"] = name_text.str.contains(r"\bbflute\b").astype(int)
    X["is_cflute"] = name_text.str.contains(r"\bcflute\b").astype(int)
    X["is_foam"] = name_text.str.contains(r"\bfoam\b").astype(int)
    X["is_kraft"] = name_text.str.contains(r"\bkraft\b").astype(int)
    X["is_roll"] = name_text.str.contains(r"\broll\b").astype(int)

    # -------------------------
    # Fitur teks tambahan
    # -------------------------
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
            layers.append(nn.Dropout(DROPOUT))
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


def main():
    seed_everything(RANDOM_SEED)

    # Load and build features
    df = pd.read_csv(DATA_FILE)
    X, y, _ = build_feature_table(df)

    # Split data
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=VAL_SIZE / (1 - TEST_SIZE),
        random_state=RANDOM_SEED
    )

    # Preprocessing
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

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
        remainder="passthrough"
    )

    # Transform data
    X_train_processed = to_dense_numpy(preprocessor.fit_transform(X_train))
    X_val_processed = to_dense_numpy(preprocessor.transform(X_val))
    X_test_processed = to_dense_numpy(preprocessor.transform(X_test))

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))

    # Create DataLoaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_processed),
        torch.FloatTensor(y_train_scaled)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_processed),
        torch.FloatTensor(y_val_scaled)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_processed),
        torch.FloatTensor(y_test_scaled)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Initialize and train model
    input_dim = X_train_processed.shape[1]
    model = ANNRegressor(input_dim)

    model, history, best_val_rmse = train_model(model, train_loader, val_loader, y_scaler)
    print(f"\nBest Validation RMSE: {best_val_rmse:.4f}\n")

    # Evaluate on test set
    model.eval()
    test_preds_scaled = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(DEVICE)
            pred = model(xb)
            test_preds_scaled.append(pred.cpu().numpy().ravel())

    test_preds_scaled = np.concatenate(test_preds_scaled)
    test_preds = y_scaler.inverse_transform(test_preds_scaled.reshape(-1, 1)).ravel()
    test_rmse = np.sqrt(mean_squared_error(y_test, np.clip(test_preds, 1.0, 5.0)))
    print(f"Test RMSE: {test_rmse:.4f}")

    # Plot training history
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
    plt.savefig(os.path.join(OUTPUT_DIR, "training_history.png"))
    plt.show()


if __name__ == "__main__":
    main()