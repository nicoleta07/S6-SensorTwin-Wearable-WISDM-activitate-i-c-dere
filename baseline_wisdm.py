
# ============================================================
# S6 - SensorTwin Wearable (WISDM)
# Neural Network (MLP) cu antrenare pe epoci
# Sursa: phone/accel
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

INPUT_DIR = "/kaggle/input/datasets/nghhhhn/wisdm-dataset/wisdm-dataset/wisdm-dataset/raw/phone/accel"
OUTPUT_DIR = "/kaggle/working"

WINDOW_SIZE = 200       # 10 secunde la 20 Hz
STEP_SIZE = 200         # fara suprapunere
EPOCHS = 50
BATCH_SIZE = 256

# ------------------------------------------------------------
# 1. Gaseste toate fisierele phone/accel
# ------------------------------------------------------------

candidate_files = []

for root, dirs, files in os.walk(INPUT_DIR):
    normalized = root.replace("\\", "/").lower()

    if "/phone/accel" not in normalized:
        continue

    for filename in files:
        if filename.lower().endswith((".txt", ".csv", ".data")):
            if any(word in filename.lower() for word in
                   ["readme", "license", "metadata"]):
                continue
            candidate_files.append(os.path.join(root, filename))

if not candidate_files:
    raise ValueError(
        "Nu am gasit fisiere in phone/accel. "
        "Verifica structura datasetului din Kaggle."
    )

candidate_files = sorted(set(candidate_files))

# ------------------------------------------------------------
# 2. Citeste toate fisierele si extrage caracteristicile
# ------------------------------------------------------------

feature_rows = []
total_valid_records = 0
files_read = 0

for file_number, path in enumerate(candidate_files, start=1):
    print(f"Fisier {file_number}/{len(candidate_files)}")

    try:
        df = pd.read_csv(
            path,
            header=None,
            sep=",",
            engine="python",
            on_bad_lines="skip"
        )

        if df.shape[1] < 6:
            print("  Sarit: formatul nu are minimum 6 coloane.")
            continue

        df = df.iloc[:, :6].copy()
        df.columns = [
            "subject", "activity", "timestamp", "x", "y", "z"
        ]

        df["subject"] = df["subject"].astype(str).str.strip()
        df["activity"] = (
            df["activity"].astype(str).str.strip().str.rstrip(";")
        )

        for col in ["timestamp", "x", "y", "z"]:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.strip().str.rstrip(";"),
                errors="coerce"
            )

        df = df.dropna(
            subset=["timestamp", "x", "y", "z"]
        )

        df = df[
            (df["subject"] != "") &
            (df["activity"] != "") &
            (df["activity"].str.lower() != "activity") &
            (df["activity"].str.lower() != "nan")
        ]

        df = df.drop_duplicates(
            subset=["subject", "activity", "timestamp", "x", "y", "z"]
        )

        if df.empty:
            continue

        total_valid_records += len(df)
        files_read += 1
        df = df.sort_values(["subject", "timestamp"])

        # Ferestre separate pe participant si activitate.
        for (subject, activity), group in df.groupby(
            ["subject", "activity"], sort=False
        ):
            values = group[["x", "y", "z"]].to_numpy(dtype=np.float32)

            for start in range(
                0, len(values) - WINDOW_SIZE + 1, STEP_SIZE
            ):
                window = values[start:start + WINDOW_SIZE]

                row = {
                    "subject": subject,
                    "activity": activity
                }

                for i, axis in enumerate(["x", "y", "z"]):
                    row[f"{axis}_mean"] = float(np.mean(window[:, i]))
                    row[f"{axis}_std"] = float(np.std(window[:, i]))
                    row[f"{axis}_min"] = float(np.min(window[:, i]))
                    row[f"{axis}_max"] = float(np.max(window[:, i]))

                feature_rows.append(row)

        del df

    except Exception as error:
        print("  Fisier sarit:", str(error)[:150])

if not feature_rows:
    raise ValueError(
        "Nu s-au creat ferestre. Verifica formatul datelor."
    )

data = pd.DataFrame(feature_rows)

# Pastram activitatile cu minimum doua ferestre.
counts = data["activity"].value_counts()
valid_classes = counts[counts >= 2].index
data = data[data["activity"].isin(valid_classes)].copy()

if data["activity"].nunique() < 2:
    raise ValueError("Sunt necesare cel putin doua clase.")

feature_cols = [
    col for col in data.columns
    if col not in ["subject", "activity"]
]

X = data[feature_cols].to_numpy(dtype=np.float32)
groups = data["subject"].to_numpy()
encoder = LabelEncoder()
y = encoder.fit_transform(data["activity"].to_numpy())
num_classes = len(encoder.classes_)

print("\nRezumatul datelor")
print("Fisiere citite cu succes:", files_read)
print("Inregistrari valide:", f"{total_valid_records:,}")
print("Participanti:", data["subject"].nunique())
print("Activitati:", num_classes)
print("Ferestre:", f"{len(data):,}")

# ------------------------------------------------------------
# 3. Impartire pe participanti: train / validation / test
# ------------------------------------------------------------
print("\n3. Impart datele pe participanti...")

all_classes = set(np.unique(y))
all_indices = np.arange(len(y))

# Incercam impartiri diferite pana gasim un set de train
# care contine toate clasele.
test_splitter = GroupShuffleSplit(
    n_splits=40, test_size=0.20, random_state=SEED
)

chosen = None

for trainval_idx, test_idx in test_splitter.split(X, y, groups):
    if set(np.unique(y[trainval_idx])) != all_classes:
        continue

    X_tv, y_tv, g_tv = X[trainval_idx], y[trainval_idx], groups[trainval_idx]

    val_splitter = GroupShuffleSplit(
        n_splits=30, test_size=0.20, random_state=SEED + 1
    )

    for train_rel, val_rel in val_splitter.split(X_tv, y_tv, g_tv):
        if set(np.unique(y_tv[train_rel])) == all_classes:
            chosen = (
                trainval_idx[train_rel],
                trainval_idx[val_rel],
                test_idx
            )
            break

    if chosen is not None:
        break

if chosen is None:
    raise ValueError(
        "Nu am gasit o impartire pe participanti in care setul "
        "de antrenare sa contina toate clasele. Verifica daca "
        "fiecare activitate este prezenta la suficienti participanti."
    )

train_idx, val_idx, test_idx = chosen

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

print("Participanti train:", len(np.unique(groups[train_idx])))
print("Participanti validation:", len(np.unique(groups[val_idx])))
print("Participanti test:", len(np.unique(groups[test_idx])))

# ------------------------------------------------------------
# 4. Normalizeaza datele fara a invata din validation/test
# ------------------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val = scaler.transform(X_val).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

# Ponderi calculate numai din setul de antrenare.
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=y_train
)
class_weights = {
    i: float(weight)
    for i, weight in enumerate(class_weights_array)
}

# ------------------------------------------------------------
# 5. Construieste reteaua neuronala MLP
# ------------------------------------------------------------
print("\n4. Construiesc reteaua neuronala...")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.30),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.20),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Opreste antrenarea daca validarea nu se mai imbunatateste.
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )
]

# ------------------------------------------------------------
# 6. Antreneaza pe epoci
# ------------------------------------------------------------
print("\n5. Incep antrenarea...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# ------------------------------------------------------------
# 7. Evaluare finala pe setul de test
# ------------------------------------------------------------
print("\n6. Evaluez modelul pe setul de test...")

test_loss, test_accuracy = model.evaluate(
    X_test, y_test, verbose=0
)

probabilities = model.predict(X_test, verbose=0)
predictions = np.argmax(probabilities, axis=1)

macro_f1 = f1_score(
    y_test, predictions,
    labels=np.arange(num_classes),
    average="macro",
    zero_division=0
)

print("\n" + "=" * 48)
print("REZULTATE SENSOR TWIN - MLP")
print("=" * 48)
print(f"Accuracy test: {test_accuracy * 100:.2f}%")
print(f"Macro F1-score: {macro_f1 * 100:.2f}%")
print(f"Test loss: {test_loss:.4f}")
print(f"Epoci efectuate: {len(history.history['loss'])}")
print(f"Fisiere citite: {files_read}")
print(f"Inregistrari valide: {total_valid_records:,}")
print(f"Participanti: {data['subject'].nunique()}")
print(f"Activitati: {num_classes}")
print(f"Ferestre totale: {len(data):,}")
print("=" * 48)

print("\nRaport pe activitati:")
print(classification_report(
    y_test,
    predictions,
    labels=np.arange(num_classes),
    target_names=encoder.classes_,
    zero_division=0
))

# ------------------------------------------------------------
# 8. Grafice de antrenare si matrice de confuzie
# ------------------------------------------------------------
epochs_done = range(1, len(history.history["loss"]) + 1)

plt.figure(figsize=(8, 4))
plt.plot(epochs_done, history.history["accuracy"], label="Train accuracy")
plt.plot(epochs_done, history.history["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoca")
plt.ylabel("Acuratete")
plt.title("Acuratetea pe epoci")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(epochs_done, history.history["loss"], label="Train loss")
plt.plot(epochs_done, history.history["val_loss"], label="Validation loss")
plt.xlabel("Epoca")
plt.ylabel("Loss")
plt.title("Loss pe epoci")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    labels=np.arange(num_classes),
    display_labels=encoder.classes_,
    xticks_rotation=90,
    cmap="Blues",
    ax=ax,
    colorbar=False
)
plt.title("Matrice de confuzie - setul de test")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 9. Salveaza modelul si rezumatul rezultatelor
# ------------------------------------------------------------
metrics = pd.DataFrame([{
    "model": "MLP Neural Network",
    "accuracy_percent": round(test_accuracy * 100, 2),
    "macro_f1_percent": round(macro_f1 * 100, 2),
    "test_loss": round(float(test_loss), 4),
    "epochs_completed": len(history.history["loss"]),
    "files_read": files_read,
    "valid_records": total_valid_records,
    "participants": int(data["subject"].nunique()),
    "activities": num_classes,
    "total_windows": len(data),
    "train_windows": len(train_idx),
    "validation_windows": len(val_idx),
    "test_windows": len(test_idx)
}])

metrics.to_csv(
    os.path.join(OUTPUT_DIR, "sensortwin_metrics.csv"),
    index=False
)

model.save(os.path.join(OUTPUT_DIR, "sensortwin_mlp.keras"))

print("\nRezumat salvat: /kaggle/working/sensortwin_metrics.csv")
print("Model salvat: /kaggle/working/sensortwin_mlp.keras")
print("\nNota: modelul recunoaste activitati, nu detecteaza direct")
print("caderile daca datasetul nu contine etichete pentru caderi.")
