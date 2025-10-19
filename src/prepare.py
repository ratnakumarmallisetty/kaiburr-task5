from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from config import CFG
from labels import product_to_label
from text_clean import clean_text


def find_data_file() -> Path:
    """Search for the complaint(s).csv or .json file automatically."""
    base = Path(__file__).resolve().parent.parent / "data"  # always go to ../data
    if not base.exists():
        raise SystemExit(f"❌ Data folder not found at: {base}")

    # list possible names
    candidates = [
        "complaint.csv", "complaints.csv",
        "complaint.json", "complaints.json",
    ]

    for name in candidates:
        f = base / name
        if f.exists():
            print(f"[prepare] ✅ Found data file → {f}")
            return f

    # nothing found
    print("Files present in data folder:", list(base.iterdir()))
    raise SystemExit(
        "❌ Could not find any complaint*.csv or complaint*.json in data folder.\n"
        "👉 Please check the file name exactly."
    )


def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        print("[prepare] Loading first 200,000 rows (sample) ... this is faster ⚡")
        return pd.read_csv(path, nrows=200000)   # ← limits to 200k rows

    elif path.suffix.lower() == ".json":
        print("[prepare] Loading JSON ... this may take a while ⏳")
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            return pd.read_json(path)
    else:
        raise SystemExit("Unsupported format; only CSV or JSON allowed.")


def main():
    data_path = find_data_file()
    df = load_dataset(data_path)

    # 2) Check columns
    need = {CFG.text_col, CFG.product_col}
    missing = need - set(df.columns)
    if missing:
        print("Existing columns:", list(df.columns)[:10])
        raise SystemExit(f"❌ Missing required columns: {missing}")

    # 3) Map label
    df["label"] = df[CFG.product_col].map(product_to_label)
    before = len(df)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    after = len(df)
    print(f"[prepare] kept {after}/{before} rows for the 4 classes.")

    # 4) Clean text
    df["text"] = df[CFG.text_col].apply(clean_text)
    df = df[df["text"].str.len() > 3].copy()

    # 5) Split
    X = df["text"].to_numpy()
    y = df["label"].to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=CFG.random_state, stratify=y
    )

    # 6) Save outputs
    CFG.artifacts_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"text": Xtr, "label": ytr}).to_csv(CFG.artifacts_dir / "train.csv", index=False)
    pd.DataFrame({"text": Xte, "label": yte}).to_csv(CFG.artifacts_dir / "test.csv", index=False)

    summary = {
        "n_total_kept": int(len(df)),
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "class_counts": df["label"].value_counts().sort_index().to_dict(),
        "label_mapping": {
            "0": "Credit reporting, repair, or other",
            "1": "Debt collection",
            "2": "Consumer Loan",
            "3": "Mortgage"
        }
    }
    with open(CFG.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[prepare] ✅ Wrote train/test & summary under: {CFG.artifacts_dir}/")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
