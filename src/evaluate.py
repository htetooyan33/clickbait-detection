"""
evaluate.py
─────────────────────────────────────────────────────────────────────────────
Standalone evaluation module.

Loads the saved best_model.pkl, runs inference on an evaluation set (or the
full test partition) and prints a detailed classification report along with
per-class metrics.

Usage
-----
    python src/evaluate.py
"""

import os
import sys
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess_corpus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
DATA_PATH  = os.path.join(BASE_DIR, "data",   "clickbait_data.csv")


def evaluate():
    # ── Load model artefact ──────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        log.error("Model not found at %s — run src/train.py first.", MODEL_PATH)
        sys.exit(1)

    payload    = joblib.load(MODEL_PATH)
    model      = payload["model"]
    vectorizer = payload["vectorizer"]
    model_name = payload["model_name"]
    log.info("Loaded model: %s", model_name)

    # ── Load & preprocess data ───────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH).dropna(subset=["headline", "clickbait"])
    X_clean = preprocess_corpus(df["headline"].tolist())
    y_true  = df["clickbait"].values

    X_features = vectorizer.transform(X_clean)
    y_pred     = model.predict(X_features)
    y_proba    = model.predict_proba(X_features)[:, 1]

    # ── Print report ─────────────────────────────────────────────────────────
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  EVALUATION REPORT — {model_name}")
    print(sep)
    print(f"\n  Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  ROC AUC  : {roc_auc_score(y_true, y_proba):.4f}")
    print()
    print(classification_report(
        y_true, y_pred,
        target_names=["Non-Clickbait", "Clickbait"],
        digits=4,
    ))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"  Confusion Matrix")
    print(f"    True Negatives  : {tn}")
    print(f"    False Positives : {fp}")
    print(f"    False Negatives : {fn}")
    print(f"    True Positives  : {tp}")
    print(sep + "\n")


if __name__ == "__main__":
    evaluate()