"""Data validation, result shaping, and optional evaluation helpers."""
from __future__ import annotations

from collections.abc import Iterable
import re

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

SENTIMENTS = ("Negative", "Neutral", "Positive")
COMMON_TEXT_COLUMNS = ("headline", "title", "summary", "content", "text", "news", "article")


def validate_text(value: object) -> str:
    """Return a usable text string or raise a friendly validation error."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise ValueError("Please enter financial news text to analyze.")
    text = str(value).strip()
    if not text:
        raise ValueError("Please enter financial news text to analyze.")
    return text


def normalize_sentiment_label(label: object) -> str | None:
    """Normalize common three-class label spellings; unsupported values are None."""
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return None
    clean = re.sub(r"[^a-z]", "", str(label).lower().strip())
    mapping = {
        "positive": "Positive", "pos": "Positive", "bullish": "Positive", "1": "Positive",
        "negative": "Negative", "neg": "Negative", "bearish": "Negative", "1negative": "Negative",
        "neutral": "Neutral", "neu": "Neutral", "mixed": "Neutral", "0": "Neutral",
    }
    return mapping.get(clean)


def detect_text_column(columns: Iterable[object]) -> str | None:
    """Find a likely news-text column, case-insensitively."""
    lookup = {str(column).lower().strip(): str(column) for column in columns}
    return next((lookup[name] for name in COMMON_TEXT_COLUMNS if name in lookup), None)


def clean_batch_dataframe(data: pd.DataFrame, text_column: str, remove_duplicates: bool = True) -> tuple[pd.DataFrame, int, int]:
    """Drop unusable rows while retaining every original data column."""
    if text_column not in data.columns:
        raise ValueError(f"The selected text column '{text_column}' is not in the CSV.")
    result = data.copy()
    text = result[text_column]
    valid = text.notna() & text.astype(str).str.strip().ne("")
    skipped = int((~valid).sum())
    result = result.loc[valid].copy()
    result[text_column] = result[text_column].astype(str).str.strip()
    duplicates = 0
    if remove_duplicates:
        duplicates = int(result.duplicated(subset=[text_column]).sum())
        result = result.drop_duplicates(subset=[text_column]).copy()
    return result, skipped, duplicates


def probabilities_to_result(text: str, probabilities: dict[str, float]) -> dict[str, object]:
    """Validate and convert class probabilities into one standardized prediction."""
    text = validate_text(text)
    normalized = {name: float(probabilities.get(name, 0.0)) for name in SENTIMENTS}
    total = sum(normalized.values())
    if total <= 0:
        raise ValueError("Model output did not contain usable sentiment probabilities.")
    normalized = {name: value / total for name, value in normalized.items()}
    sentiment = max(normalized, key=normalized.get)
    return {
        "text": text,
        "sentiment": sentiment,
        "confidence": normalized[sentiment],
        "negative_probability": normalized["Negative"],
        "neutral_probability": normalized["Neutral"],
        "positive_probability": normalized["Positive"],
        "sentiment_score": normalized["Positive"] - normalized["Negative"],
    }


def results_dataframe(original: pd.DataFrame, predictions: list[dict[str, object]]) -> pd.DataFrame:
    """Append prediction fields to original CSV rows without losing source columns."""
    if len(original) != len(predictions):
        raise ValueError("The number of model predictions does not match the input rows.")
    return pd.concat([original.reset_index(drop=True), pd.DataFrame(predictions).drop(columns=["text"], errors="ignore")], axis=1)


def evaluate_predictions(data: pd.DataFrame, reference_column: str) -> dict[str, object] | None:
    """Evaluate predictions only where a compatible trusted reference label exists."""
    if reference_column not in data or "sentiment" not in data:
        return None
    evaluation = pd.DataFrame({
        "actual": data[reference_column].map(normalize_sentiment_label),
        "predicted": data["sentiment"].map(normalize_sentiment_label),
    }).dropna()
    if evaluation.empty:
        return None
    labels = list(SENTIMENTS)
    precision, recall, f1, _ = precision_recall_fscore_support(
        evaluation["actual"], evaluation["predicted"], labels=labels, average="weighted", zero_division=0
    )
    return {
        "records": len(evaluation), "accuracy": accuracy_score(evaluation["actual"], evaluation["predicted"]),
        "precision": precision, "recall": recall, "f1": f1,
        "matrix": confusion_matrix(evaluation["actual"], evaluation["predicted"], labels=labels), "labels": labels,
    }
