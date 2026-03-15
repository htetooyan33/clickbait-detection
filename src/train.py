"""
train.py
─────────────────────────────────────────────────────────────────────────────
End-to-end training pipeline for the Clickbait Detection project.

Pipeline
--------
1.  Load & inspect dataset
2.  Handle missing values
3.  Preprocess text (src/preprocess.py)
4.  Feature extraction with CountVectorizer (unigrams + bigrams)
5.  Train/test split  (80 / 20, seed = 42)
6.  Train three classifiers
    - Multinomial Naïve Bayes
    - Logistic Regression
    - Gradient Boosting (scikit-learn; equivalent to XGBoost for portfolio use)
7.  Evaluate all models and print comparison table
8.  Save the best model (F1 score) to models/best_model.pkl
9.  Produce visualisation artefacts (confusion matrices, ROC curves, bar chart)

Usage
-----
    python src/train.py
"""

import os
import sys
import logging
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless back-end — no display required
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
)

# Allow running from the project root or from src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess_corpus

warnings.filterwarnings("ignore")

# ── Reproducibility seed ──────────────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "data",    "clickbait_data.csv")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(OUTPUT_DIR, "training.log")),
    ],
)
log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING & INSPECTION
# ═════════════════════════════════════════════════════════════════════════════
def load_data(path: str) -> pd.DataFrame:
    log.info("Loading dataset from: %s", path)
    df = pd.read_csv(path)

    log.info("Dataset shape          : %s", df.shape)
    log.info("Columns                : %s", list(df.columns))
    log.info("Class distribution:\n%s", df["clickbait"].value_counts().to_string())
    log.info("Missing values:\n%s",      df.isnull().sum().to_string())
    log.info("Sample rows:\n%s",         df.head(3).to_string())

    # ── Handle missing values ─────────────────────────────────────────────────
    before = len(df)
    df.dropna(subset=["headline", "clickbait"], inplace=True)
    df["headline"] = df["headline"].astype(str).str.strip()
    df = df[df["headline"] != ""]
    after = len(df)
    if before != after:
        log.warning("Dropped %d rows with missing / empty headlines.", before - after)

    return df.reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════
def build_features(X_train_text, X_test_text):
    """
    CountVectorizer with unigrams + bigrams (ngram_range=(1,2)).

    Why CountVectorizer?
    --------------------
    * Unigrams capture individual discriminative words
      ("you", "won't", "believe").
    * Bigrams capture short phrases that are strong clickbait signals
      ("won't believe", "you need", "will shock").
    * Multinomial Naïve Bayes requires non-negative integer features —
      CountVectorizer raw counts satisfy this constraint directly.
    * TF-IDF is an alternative but raw counts work well for short-headline
      classification where term frequency differences matter.
    """
    log.info("Building CountVectorizer features (ngram_range=(1,2), max_features=50000) ...")
    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        max_features=50_000,
        stop_words="english",   # second stop-word filter as a safety net
        min_df=2,               # ignore terms appearing in fewer than 2 docs
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test  = vectorizer.transform(X_test_text)
    log.info("Feature matrix (train) : %s", X_train.shape)
    log.info("Feature matrix (test)  : %s", X_test.shape)
    return vectorizer, X_train, X_test


# ═════════════════════════════════════════════════════════════════════════════
# 3.  MODEL DEFINITIONS
# ═════════════════════════════════════════════════════════════════════════════
def get_models():
    """
    Return a dict of {name: estimator} for the three classifiers.

    Model rationale
    ---------------
    MultinomialNB
        Strong baseline for text — fast, interpretable, works well on
        count-based sparse features.  Assumes feature independence (naïve)
        which is surprisingly effective for NLP tasks.

    LogisticRegression
        Linear classifier with L2 regularisation.  Learns feature weights
        directly; highly interpretable via coefficient inspection.  Often
        the best performer on linearly separable text tasks.

    GradientBoostingClassifier
        Ensemble of shallow decision trees trained sequentially (gradient
        boosting).  Captures non-linear feature interactions that linear
        models miss.  Equivalent role to XGBoost; uses scikit-learn's
        native implementation for portability.
    """
    return {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression":     LogisticRegression(
                                       max_iter=1000,
                                       C=1.0,
                                       random_state=RANDOM_STATE,
                                   ),
        "Gradient Boosting":       GradientBoostingClassifier(
                                       n_estimators=200,
                                       max_depth=4,
                                       learning_rate=0.1,
                                       random_state=RANDOM_STATE,
                                   ),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 4.  TRAINING & EVALUATION
# ═════════════════════════════════════════════════════════════════════════════
def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    """
    Train every model, compute evaluation metrics, return results dict.
    """
    results   = {}
    trained   = {}

    for name, model in models.items():
        log.info("Training: %s ...", name)
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "Accuracy":  accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall":    recall_score(y_test, y_pred, zero_division=0),
            "F1 Score":  f1_score(y_test, y_pred, zero_division=0),
            "ROC AUC":   roc_auc_score(y_test, y_proba),
        }
        results[name] = metrics
        trained[name] = (model, y_pred, y_proba)

        log.info(
            "%s  →  Acc=%.4f  Prec=%.4f  Rec=%.4f  F1=%.4f  AUC=%.4f",
            name,
            metrics["Accuracy"], metrics["Precision"],
            metrics["Recall"],   metrics["F1 Score"], metrics["ROC AUC"],
        )

    return results, trained


def print_comparison_table(results: dict):
    df = pd.DataFrame(results).T
    df = df[["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]]
    df = df.round(4)
    sep = "─" * 78
    log.info("\n%s\n  MODEL PERFORMANCE COMPARISON\n%s\n%s\n%s",
             sep, sep, df.to_string(), sep)
    print("\n" + sep)
    print("  MODEL PERFORMANCE COMPARISON")
    print(sep)
    print(df.to_string())
    print(sep + "\n")


# ═════════════════════════════════════════════════════════════════════════════
# 5.  VISUALISATIONS
# ═════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrices(trained, y_test, out_dir):
    n = len(trained)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    fig.suptitle("Confusion Matrices", fontsize=16, fontweight="bold", y=1.02)

    for ax, (name, (_, y_pred, _)) in zip(axes, trained.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-Clickbait", "Clickbait"],
            yticklabels=["Non-Clickbait", "Clickbait"],
            ax=ax, linewidths=0.5,
        )
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    plt.tight_layout()
    path = os.path.join(out_dir, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", path)
    return path


def plot_roc_curves(trained, y_test, out_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    for (name, (_, _, y_proba)), color in zip(trained.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f"{name}  (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curves — Clickbait Detection", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

    path = os.path.join(out_dir, "roc_curves.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", path)
    return path


def plot_performance_comparison(results: dict, out_dir: str):
    df      = pd.DataFrame(results).T
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
    df      = df[metrics]

    x      = np.arange(len(metrics))
    width  = 0.22
    colors = ["#2196F3", "#FF5722", "#4CAF50"]
    models = list(df.index)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (model, color) in enumerate(zip(models, colors)):
        offset = (i - len(models) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, df.loc[model, metrics], width,
                        label=model, color=color, edgecolor="white", linewidth=0.7)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0.0, 1.12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(out_dir, "model_comparison.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", path)
    return path


def plot_class_distribution(df: pd.DataFrame, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    counts = df["clickbait"].value_counts()
    axes[0].bar(["Non-Clickbait (0)", "Clickbait (1)"],
                [counts.get(0, 0), counts.get(1, 0)],
                color=["#4CAF50", "#FF5722"], edgecolor="white", linewidth=0.7)
    for i, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
        axes[0].text(i, v + 50, str(v), ha="center", fontweight="bold")
    axes[0].set_title("Class Distribution", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].grid(axis="y", alpha=0.3)

    # Headline length histogram
    df["headline_len"] = df["headline"].apply(lambda x: len(str(x).split()))
    for label, color, lname in [(0, "#4CAF50", "Non-Clickbait"),
                                  (1, "#FF5722", "Clickbait")]:
        subset = df[df["clickbait"] == label]["headline_len"]
        axes[1].hist(subset, bins=30, alpha=0.6, color=color, label=lname, edgecolor="white")
    axes[1].set_title("Headline Length Distribution", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Word Count")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    path = os.path.join(out_dir, "eda.png")
    plt.suptitle("Exploratory Data Analysis", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: %s", path)
    return path


# ═════════════════════════════════════════════════════════════════════════════
# 6.  SAVE BEST MODEL
# ═════════════════════════════════════════════════════════════════════════════
def save_best_model(trained, results, vectorizer, model_dir):
    best_name = max(results, key=lambda n: results[n]["F1 Score"])
    best_model = trained[best_name][0]
    log.info("Best model: %s  (F1=%.4f)", best_name, results[best_name]["F1 Score"])

    payload = {
        "model":       best_model,
        "vectorizer":  vectorizer,
        "model_name":  best_name,
        "metrics":     results[best_name],
    }
    path = os.path.join(model_dir, "best_model.pkl")
    joblib.dump(payload, path)
    log.info("Model saved to: %s", path)
    return best_name, path


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 78)
    log.info("  CLICKBAIT DETECTION — TRAINING PIPELINE")
    log.info("=" * 78)

    # 1. Load
    df = load_data(DATA_PATH)

    # 2. EDA visualisation
    plot_class_distribution(df, OUTPUT_DIR)

    # 3. Preprocess
    log.info("Step 3: Text preprocessing ...")
    X_clean = preprocess_corpus(df["headline"].tolist())
    y       = df["clickbait"].values

    # 4. Split
    log.info("Step 4: Train/test split (80/20, seed=%d) ...", RANDOM_STATE)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_clean, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    log.info("Train samples: %d  |  Test samples: %d", len(y_train), len(y_test))

    # 5. Feature extraction
    vectorizer, X_train, X_test = build_features(X_train_text, X_test_text)

    # 6. Train & evaluate
    log.info("Step 6: Training and evaluating models ...")
    models            = get_models()
    results, trained  = train_and_evaluate(models, X_train, X_test, y_train, y_test)

    # 7. Print comparison table
    print_comparison_table(results)

    # 8. Visualisations
    log.info("Step 8: Generating visualisations ...")
    plot_confusion_matrices(trained, y_test, OUTPUT_DIR)
    plot_roc_curves(trained, y_test, OUTPUT_DIR)
    plot_performance_comparison(results, OUTPUT_DIR)

    # 9. Save best model
    log.info("Step 9: Saving best model ...")
    best_name, model_path = save_best_model(trained, results, vectorizer, MODEL_DIR)

    log.info("=" * 78)
    log.info("Training complete.  Best model: %s", best_name)
    log.info("Artefacts saved to: %s", OUTPUT_DIR)
    log.info("=" * 78)


if __name__ == "__main__":
    main()