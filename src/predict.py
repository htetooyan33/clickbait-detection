"""
predict.py
─────────────────────────────────────────────────────────────────────────────
Inference module — exposes a simple predict_clickbait(text) function.

Usage (as a script)
-------------------
    python src/predict.py "You won't believe what happened next!"

Usage (as a library)
--------------------
    from src.predict import predict_clickbait
    result = predict_clickbait("Scientists discover a new planet!")
    print(result)
"""

import os
import sys
import logging
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")

# ── Lazy-load model once ──────────────────────────────────────────────────────
_payload = None


def _load_model():
    global _payload
    if _payload is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run src/train.py first."
            )
        _payload = joblib.load(MODEL_PATH)
        log.info("Model loaded: %s", _payload["model_name"])
    return _payload


def predict_clickbait(text: str) -> dict:
    """
    Predict whether a given headline is clickbait.

    Parameters
    ----------
    text : str
        A headline string to classify.

    Returns
    -------
    dict with keys:
        label       : "Clickbait" | "Non-Clickbait"
        probability : float  probability of being clickbait (0–1)
        model       : str    name of the underlying model
    """
    payload    = _load_model()
    model      = payload["model"]
    vectorizer = payload["vectorizer"]

    cleaned    = preprocess_text(text)
    features   = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    label = "Clickbait" if prediction == 1 else "Non-Clickbait"

    return {
        "label":       label,
        "probability": round(float(probability), 4),
        "model":       payload["model_name"],
    }


# ── Demo examples ─────────────────────────────────────────────────────────────
DEMO_HEADLINES = [
    "You won't believe what happened next!",
    "Scientists Discover New Exoplanet in Habitable Zone",
    "This one weird trick will make you rich overnight",
    "Federal Reserve Raises Interest Rates by 25 Basis Points",
    "Which Disney princess are you based on your food choices?",
    "Study Links Ultra-Processed Food Consumption to Cognitive Decline",
    "This dog's reaction to seeing his owner is everything",
    "Parliament passes landmark climate legislation after three-day debate",
]


if __name__ == "__main__":
    import json

    if len(sys.argv) > 1:
        # Single headline from command-line argument
        headline = " ".join(sys.argv[1:])
        result   = predict_clickbait(headline)
        print(f"\nHeadline  : {headline}")
        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['probability']:.2%}")
        print(f"Model     : {result['model']}\n")
    else:
        # Run all demo headlines
        print("\n" + "=" * 65)
        print("  CLICKBAIT DETECTION — DEMO PREDICTIONS")
        print("=" * 65)
        for headline in DEMO_HEADLINES:
            result = predict_clickbait(headline)
            bar    = "█" * int(result["probability"] * 20)
            bar    = bar.ljust(20)
            print(f"\n  ❯ {headline}")
            print(f"    → {result['label']:15s}  [{bar}] {result['probability']:.2%}")
        print("\n" + "=" * 65 + "\n")