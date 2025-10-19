"""
Central config: column names and default paths.
Change text_col/product_col if your CSV headers differ.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    csv_path: Path = Path("data/complaints.csv")
    text_col: str = "Consumer complaint narrative"
    product_col: str = "Product"
    artifacts_dir: Path = Path("artifacts")
    summary_json: Path = artifacts_dir / "X_y_summary.json"
    report_txt: Path = artifacts_dir / "train_report.txt"
    model_path: Path = artifacts_dir / "best_model.joblib"
    random_state: int = 42

CFG = Config()
