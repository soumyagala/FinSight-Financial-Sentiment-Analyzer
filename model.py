"""Lazy FinBERT loading and device-aware inference."""
from __future__ import annotations

from collections.abc import Callable

from src.analysis import SENTIMENTS, probabilities_to_result, validate_text

MODEL_NAME = "ProsusAI/finbert"


def get_device() -> int:
    # Delayed import keeps helper/unit-test imports lightweight; PyTorch is still
    # required at real inference time through the declared project dependency.
    import torch
    return 0 if torch.cuda.is_available() else -1


def load_finbert_pipeline() -> Callable:
    """Load the official FinBERT pipeline. Called only through Streamlit cache."""
    from transformers import pipeline
    return pipeline("text-classification", model=MODEL_NAME, tokenizer=MODEL_NAME, device=get_device(), top_k=None)


def label_to_sentiment(label: object) -> str | None:
    value = str(label).upper().strip()
    aliases = {"POSITIVE": "Positive", "NEGATIVE": "Negative", "NEUTRAL": "Neutral"}
    if value in aliases:
        return aliases[value]
    # FinBERT configurations occasionally expose LABEL_0 etc.; inspect standard ordering.
    indexed = {"LABEL_0": "Positive", "LABEL_1": "Negative", "LABEL_2": "Neutral"}
    return indexed.get(value)


def infer_texts(texts: list[str], classifier: Callable, batch_size: int = 16, max_length: int = 256) -> list[dict[str, object]]:
    """Run a classifier and return consistently named FinBERT prediction fields."""
    cleaned = [validate_text(text) for text in texts]
    raw_output = classifier(cleaned, truncation=True, max_length=max_length, batch_size=batch_size)
    if len(cleaned) == 1 and raw_output and isinstance(raw_output[0], dict):
        raw_output = [raw_output]
    results = []
    for text, rows in zip(cleaned, raw_output):
        probabilities = {sentiment: 0.0 for sentiment in SENTIMENTS}
        for row in rows:
            sentiment = label_to_sentiment(row.get("label"))
            if sentiment:
                probabilities[sentiment] = float(row.get("score", 0.0))
        results.append(probabilities_to_result(text, probabilities))
    return results
