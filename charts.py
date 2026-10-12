"""Plotly chart factories used by the Streamlit interface."""
from __future__ import annotations
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.analysis import SENTIMENTS

COLORS = {"Positive": "#2dd4bf", "Negative": "#fb7185", "Neutral": "#60a5fa"}


def sentiment_counts(data: pd.DataFrame) -> pd.DataFrame:
    counts = data.get("sentiment", pd.Series(dtype=str)).value_counts().reindex(SENTIMENTS, fill_value=0)
    return counts.rename_axis("sentiment").reset_index(name="articles")


def donut_chart(data: pd.DataFrame) -> go.Figure:
    counts = sentiment_counts(data)
    figure = px.pie(counts, names="sentiment", values="articles", hole=.63, color="sentiment", color_discrete_map=COLORS)
    return figure.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=20, b=10), legend_title_text="Sentiment")


def bar_chart(data: pd.DataFrame) -> go.Figure:
    counts = sentiment_counts(data)
    figure = px.bar(counts, x="sentiment", y="articles", color="sentiment", color_discrete_map=COLORS, text="articles")
    return figure.update_layout(template="plotly_dark", showlegend=False, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="", yaxis_title="Articles")


def probability_chart(result: dict[str, object]) -> go.Figure:
    data = pd.DataFrame({"sentiment": list(SENTIMENTS), "probability": [result[f"{x.lower()}_probability"] for x in SENTIMENTS]})
    return px.bar(data, x="sentiment", y="probability", color="sentiment", color_discrete_map=COLORS, range_y=[0, 1]).update_layout(template="plotly_dark", showlegend=False, yaxis_tickformat=".0%")


def confidence_chart(data: pd.DataFrame) -> go.Figure:
    return px.histogram(data, x="confidence", color="sentiment", color_discrete_map=COLORS, nbins=15, barmode="overlay", range_x=[0, 1]).update_layout(template="plotly_dark", yaxis_title="Articles", xaxis_tickformat=".0%")


def confusion_chart(evaluation: dict[str, object]) -> go.Figure:
    return px.imshow(evaluation["matrix"], x=evaluation["labels"], y=evaluation["labels"], text_auto=True, color_continuous_scale="Blues", labels={"x": "Predicted", "y": "Reference"}).update_layout(template="plotly_dark")
