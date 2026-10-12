import pandas as pd
import pytest
from src.analysis import clean_batch_dataframe, evaluate_predictions, probabilities_to_result, validate_text

def test_empty_text_is_rejected():
    with pytest.raises(ValueError): validate_text("   ")

def test_probabilities_make_prediction_and_score():
    item = probabilities_to_result("Results improved", {"Positive": .7, "Negative": .1, "Neutral": .2})
    assert item["sentiment"] == "Positive"
    assert item["confidence"] == pytest.approx(.7)
    assert item["sentiment_score"] == pytest.approx(.6)

def test_cleaning_preserves_columns_and_removes_empty_and_duplicate():
    data = pd.DataFrame({"headline": ["Good news", "", None, "Good news", "Bad news"], "url": [1,2,3,4,5]})
    cleaned, skipped, duplicates = clean_batch_dataframe(data, "headline")
    assert list(cleaned.columns) == ["headline", "url"]
    assert len(cleaned) == 2 and skipped == 2 and duplicates == 1

def test_evaluation_metrics_for_compatible_reference_labels():
    data = pd.DataFrame({"truth": ["positive", "negative", "neutral"], "sentiment": ["Positive", "Negative", "Neutral"]})
    result = evaluate_predictions(data, "truth")
    assert result["records"] == 3 and result["accuracy"] == 1
