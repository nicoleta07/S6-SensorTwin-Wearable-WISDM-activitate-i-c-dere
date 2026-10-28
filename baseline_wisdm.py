"""
SensorTwin / WISDM — baseline compact pentru recunoașterea activităților.
Input: CSV/TXT cu coloanele subject-id, activity-code, timestamp, x, y, z.
Pentru structura oficială pe fișiere per subiect, unește/exportă datele într-un CSV
cu aceste coloane înainte de rulare. Nu detectează căderi dacă acestea nu există în etichete.
"""
from pathlib import Path
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import GroupShuffleSplit

ALIASES = {
    "subject-id": ["subject-id", "subject", "user", "user_id"],
    "activity-code": ["activity-code", "activity", "label"],
    "timestamp": ["timestamp", "time"],
    "x": ["x", "accel_x"], "y": ["y", "accel_y"], "z": ["z", "accel_z"],
}

def load_data(path):
    # Încercăm CSV standard și format WISDM separat prin virgulă cu punctuație finală.
    df = pd.read_csv(path, sep=None, engine="python", on_bad_lines="skip")
    df.columns = [str(c).strip().lower().replace("_", "-") for c in df.columns]
    rename = {}
    for target, candidates in ALIASES.items():
        for c in df.columns:
            if c in [x.replace("_", "-") for x in candidates]:
                rename[c] = target
                break
    df = df.rename(columns=rename)
    needed = ["subject-id", "activity-code", "timestamp", "x", "y", "z"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Lipsesc coloanele {missing}. Header găsit: {list(df.columns)}")
    for c in ["timestamp", "x", "y", "z"]:
        df[c] = pd.to_numeric(df[c].astype(str).str.rstrip(";"), errors="coerce")
    df["subject-id"] = df["subject-id"].astype(str)
    df["activity-code"] = df["activity-code"].astype(str).str.strip().str.rstrip(";")
    df = df.dropna(subset=needed).sort_values(["subject-id", "timestamp"])
    return df

def make_windows(df, window=200):
    rows = []
    for (subject, activity), g in df.groupby(["subject-id", "activity-code"]):
        vals = g[["x", "y", "z"]].to_numpy()
        # ferestre complete de 10 s la 20 Hz; nu traversăm limitele activităților
        for start in range(0, len(vals) - window + 1, window):
            chunk = vals[start:start + window]
            features = {}
            for i, axis in enumerate(["x", "y", "z"]):
                features.update({
                    f"{axis}_mean": chunk[:, i].mean(),
                    f"{axis}_std": chunk[:, i].std(),
                    f"{axis}_min": chunk[:, i].min(),
                    f"{axis}_max": chunk[:, i].max(),
                })
            rows.append({"subject": subject, "label": activity, **features})
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="CSV consolidat cu subject-id, activity-code, timestamp, x, y, z")
    args = parser.parse_args()
    df = load_data(args.csv)
    print(f"Rânduri valide: {len(df):,}; subiecți: {df['subject-id'].nunique()}; etichete: {df['activity-code'].nunique()}")
    print("Valori lipsă după curățare:", int(df[["x", "y", "z"]].isna().sum().sum()))
    data = make_windows(df)
    if data.empty or data["subject"].nunique() < 2:
        raise ValueError("Nu există suficiente ferestre/subiecți; verifică fișierul și frecvența.")
    X = data.drop(columns=["subject", "label"])
    y = data["label"]
    groups = data["subject"]
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(split.split(X, y, groups))
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = model.predict(X.iloc[test_idx])
    print(f"Ferestre: {len(data):,} | train: {len(train_idx):,} | test: {len(test_idx):,}")
    print(f"Accuracy: {accuracy_score(y.iloc[test_idx], pred):.3f}")
    print(f"Macro-F1: {f1_score(y.iloc[test_idx], pred, average='macro', zero_division=0):.3f}")
    print(classification_report(y.iloc[test_idx], pred, zero_division=0))

if __name__ == "__main__":
    main()
