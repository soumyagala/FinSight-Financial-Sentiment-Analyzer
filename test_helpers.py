from src.analysis import detect_text_column, normalize_sentiment_label
from src.model import infer_texts

def test_label_normalization():
    assert normalize_sentiment_label("BULLISH") == "Positive"
    assert normalize_sentiment_label("unknown") is None

def test_column_detection():
    assert detect_text_column(["URL", "Summary", "Date"]) == "Summary"

def test_mocked_model_label_mapping():
    def mock_pipeline(texts, **kwargs):
        return [[{"label": "positive", "score": .8}, {"label": "negative", "score": .1}, {"label": "neutral", "score": .1}] for _ in texts]
    result = infer_texts(["mock story"], mock_pipeline)[0]
    assert result["sentiment"] == "Positive"
