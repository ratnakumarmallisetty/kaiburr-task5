from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from config import CFG

def load_split():
    tr = pd.read_csv(CFG.artifacts_dir / "train.csv")
    te = pd.read_csv(CFG.artifacts_dir / "test.csv")
    return tr["text"].to_numpy(), tr["label"].to_numpy(), te["text"].to_numpy(), te["label"].to_numpy()

def build_logreg():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])

def build_linsvm():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2)),
        ("clf", LinearSVC()),
    ])

def evaluate(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(y_true, y_pred, digits=4)
    }

def main():
    Xtr, ytr, Xte, yte = load_split()

    candidates = {
        "logreg": build_logreg(),
        "linear_svm": build_linsvm(),
    }

    results = {}
    best_name = None
    best_f1 = -1.0

    for name, model in candidates.items():
        print(f"[train] training {name} ...")
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        metrics = evaluate(yte, pred)
        results[name] = metrics
        print(f"[train] {name} acc={metrics['accuracy']:.4f}  f1_macro={metrics['f1_macro']:.4f}")

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_name = name
            dump(model, CFG.model_path)
            print(f"[train] saved best model so far → {CFG.model_path}")

    # write report
    lines = []
    for name, m in results.items():
        lines.append(f"=== {name} ===")
        lines.append(f"accuracy: {m['accuracy']:.4f}")
        lines.append(f"f1_macro: {m['f1_macro']:.4f}")
        lines.append("confusion_matrix:")
        lines.append(json.dumps(m["confusion_matrix"]))
        lines.append("classification_report:")
        lines.append(m["report"])
        lines.append("")
    lines.append(f"BEST_MODEL: {best_name} (f1_macro={best_f1:.4f})")

    CFG.report_txt.write_text("\n".join(lines), encoding="utf-8")
    print(f"[train] wrote report → {CFG.report_txt}")
    print(f"[train] BEST_MODEL: {best_name}  f1_macro={best_f1:.4f}")

if __name__ == "__main__":
    main()
