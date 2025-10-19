from __future__ import annotations
import argparse, sys
from pathlib import Path
from joblib import load
from config import CFG

LABEL_NAMES = {
    0: "Credit reporting, repair, or other",
    1: "Debt collection",
    2: "Consumer Loan",
    3: "Mortgage",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", help="Classify a single text")
    ap.add_argument("--file", type=Path, help="File with one text per line")
    args = ap.parse_args()

    if not CFG.model_path.exists():
        raise SystemExit(f"Model not found: {CFG.model_path}. Run train.py first.")

    model = load(CFG.model_path)

    texts = []
    if args.text:
        texts.append(args.text)
    if args.file:
        lines = args.file.read_text(encoding="utf-8").splitlines()
        texts += [ln.strip() for ln in lines if ln.strip()]

    if not texts:
        print("Provide --text or --file")
        sys.exit(1)

    preds = model.predict(texts)
    for t, y in zip(texts, preds):
        print(f"[predict] {y} :: {LABEL_NAMES[int(y)]} :: {t[:80]}{'...' if len(t)>80 else ''}")

if __name__ == "__main__":
    main()
