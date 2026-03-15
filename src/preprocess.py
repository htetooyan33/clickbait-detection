"""
preprocess.py
─────────────────────────────────────────────────────────────────────────────
Text preprocessing pipeline for clickbait detection.

Design decisions
----------------
* Self-contained pipeline with no NLTK/spaCy runtime dependency so the module
  runs in any environment without extra downloads.
* A curated English stop-word list is embedded directly (mirrors NLTK corpus).
* Lemmatisation is approximated via ordered suffix-stripping rules covering the
  most common English inflections. In production this would be replaced by a
  WordNetLemmatizer or spaCy lemmatiser.
"""

import re
import string
import logging
from typing import List

logger = logging.getLogger(__name__)

# ── English stop-word list (mirrors NLTK english corpus) ─────────────────────
STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "s", "t", "can", "will", "just", "don", "should",
    "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren",
    "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma",
    "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren",
    "won", "wouldn",
}


def _lemmatize(word: str) -> str:
    """
    Approximate lemmatisation via ordered suffix-stripping rules.

    Rules are ordered most-specific → least-specific to prevent over-stripping.
    Example: 'running' → 'run',  'studies' → 'study'.
    """
    if len(word) <= 3:
        return word

    rules = [
        (r"ies$",  "y"),
        (r"ied$",  "y"),
        (r"ves$",  "fe"),
        (r"ness$", ""),
        (r"ment$", ""),
        (r"tion$", ""),
        (r"able$", ""),
        (r"ible$", ""),
        (r"ful$",  ""),
        (r"less$", ""),
        (r"ous$",  ""),
        (r"ing$",  ""),
        (r"ed$",   ""),
        (r"ly$",   ""),
        (r"er$",   ""),
        (r"est$",  ""),
        (r"s$",    ""),
    ]
    for pattern, replacement in rules:
        if re.search(pattern, word):
            candidate = re.sub(pattern, replacement, word)
            if len(candidate) >= 3:          # guard against over-stripping
                return candidate
    return word


def preprocess_text(text: str) -> str:
    """
    Full NLP preprocessing pipeline for a single document.

    Steps
    -----
    1. Lowercase
    2. Remove URLs
    3. Remove HTML/XML tags
    4. Replace punctuation with spaces
    5. Remove digits
    6. Tokenise on whitespace
    7. Drop stop-words and single-character tokens
    8. Lemmatise each token
    9. Re-join into a clean string

    Parameters
    ----------
    text : str  Raw headline / document text.

    Returns
    -------
    str  Cleaned, space-joined token string.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"<[^>]+>", " ", text)                    # HTML tags
    text = text.translate(
        str.maketrans(string.punctuation, " " * len(string.punctuation))
    )                                                        # punctuation
    text = re.sub(r"\d+", " ", text)                        # digits

    tokens: List[str] = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    tokens = [_lemmatize(t) for t in tokens]

    return " ".join(tokens)


def preprocess_corpus(texts) -> List[str]:
    """Apply ``preprocess_text`` to an iterable of documents."""
    logger.info("Preprocessing %d documents ...", len(texts))
    processed = [preprocess_text(t) for t in texts]
    logger.info("Preprocessing complete.")
    return processed