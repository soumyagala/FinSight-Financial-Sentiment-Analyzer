"""FinSight — Financial News Sentiment Analyzer for Indian Stock Markets."""
from __future__ import annotations

from io import StringIO
import pandas as pd
import streamlit as st

from src.analysis import clean_batch_dataframe, detect_text_column, evaluate_predictions, results_dataframe
from src.charts import bar_chart, confidence_chart, confusion_chart, donut_chart, probability_chart
from src.model import MODEL_NAME, infer_texts, load_finbert_pipeline

st.set_page_config(page_title="FinSight | Financial News Intelligence", page_icon="F", layout="wide")
st.markdown("""<style>
  .stApp { background: #07111f; color: #eaf2ff; }
  [data-testid='stSidebar'] { background: #0b1a2d; border-right: 1px solid #254563; }
  .hero { padding: .2rem 0 1rem; } .hero h1 { font-size: 2.35rem; margin-bottom: .15rem; color: #f5f9ff; }
  .subtle { color: #a9bed4; font-size: 1rem; }
  .metric-card { background: #102842; border: 1px solid #285579; border-radius: 12px; padding: 1rem 1.15rem; box-shadow: 0 8px 22px rgba(0, 0, 0, .15); }
  .metric-card p { color:#a9bed4; margin:0; font-size:.85rem; } .metric-card h2 { margin:.25rem 0 0; color:#67e8f9; }
  .result-card { padding: 1.35rem; background: #10314a; border-left: 4px solid #38bdf8; border-radius: 8px; }
  .small-note { color:#a9bed4; font-size:.85rem; }
  /* Form controls: Streamlit otherwise inherits a white browser input surface. */
  .stTextArea textarea, .stTextInput input, [data-baseweb='input'] input {
    background-color: #10263c !important; color: #f8fbff !important;
    -webkit-text-fill-color: #f8fbff !important; border: 1px solid #3a6689 !important;
    caret-color: #67e8f9 !important; border-radius: 8px !important;
  }
  .stTextArea textarea::placeholder, .stTextInput input::placeholder { color: #8ea8c2 !important; -webkit-text-fill-color: #8ea8c2 !important; }
  .stTextArea textarea:focus, .stTextInput input:focus { border-color: #38bdf8 !important; box-shadow: 0 0 0 1px #38bdf8 !important; }
  [data-baseweb='select'] > div, [data-baseweb='base-input'] > div { background-color: #10263c !important; color: #f8fbff !important; border-color: #3a6689 !important; }
  [data-baseweb='select'] * { color: #f8fbff !important; }
  [data-testid='stFileUploader'] section { background: #10263c; border-color: #3a6689; }
  .stButton > button { background: #0e7490; color: #f8fbff; border: 1px solid #38bdf8; }
  .stButton > button:hover { background: #155e75; border-color: #67e8f9; color: #ffffff; }
</style>""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

EXAMPLES = [
    "Example: A major Indian bank reports stronger quarterly profit and loan growth.",
    "Example: An infrastructure company wins a large renewable-energy contract.",
    "Example: A manufacturer reports weaker margins and cuts its annual outlook.",
    "Example: The company announces the date of its annual general meeting.",
]

@st.cache_resource(show_spinner="Loading ProsusAI/finbert (first use may download the model)...")
def get_classifier():
    return load_finbert_pipeline()

def analyze(texts: list[str], batch_size: int = 16):
    try:
        return infer_texts(texts, get_classifier(), batch_size=batch_size)
    except Exception as exc:
        raise RuntimeError(
            "FinBERT could not be loaded or used. The first run needs internet access to download "
            "ProsusAI/finbert; then it is cached locally. Check your connection and installed dependencies."
        ) from exc

def history_frame() -> pd.DataFrame:
    return pd.DataFrame(st.session_state.history)

def add_history(items: list[dict]):
    st.session_state.history.extend(items)

def metric_card(column, label: str, value: int):
    column.markdown(f"<div class='metric-card'><p>{label}</p><h2>{value}</h2></div>", unsafe_allow_html=True)

def dashboard():
    st.markdown("<div class='hero'><h1>Financial News Intelligence</h1><div class='subtle'>Analyze financial-news tone with a finance-specific transformer model. Results describe the text—not a price forecast.</div></div>", unsafe_allow_html=True)
    if st.button("Run sample analysis", type="primary", use_container_width=False):
        with st.spinner("Analyzing clearly labeled demonstration headlines..."):
            add_history(analyze(EXAMPLES))
        st.success("Sample analysis added to this browser session.")
    data = history_frame()
    counts = data["sentiment"].value_counts() if not data.empty else pd.Series(dtype=int)
    cols = st.columns(4)
    metric_card(cols[0], "TOTAL ARTICLES ANALYZED", len(data))
    metric_card(cols[1], "POSITIVE ARTICLES", int(counts.get("Positive", 0)))
    metric_card(cols[2], "NEGATIVE ARTICLES", int(counts.get("Negative", 0)))
    metric_card(cols[3], "NEUTRAL ARTICLES", int(counts.get("Neutral", 0)))
    if data.empty:
        st.info("No articles analyzed yet. Use **Run sample analysis**, analyze a headline, or upload a CSV.")
        return
    left, right = st.columns(2)
    left.plotly_chart(donut_chart(data), use_container_width=True)
    right.plotly_chart(bar_chart(data), use_container_width=True)
    st.subheader("Recently analyzed articles")
    st.dataframe(data[["text", "sentiment", "confidence"]].tail(10).iloc[::-1], use_container_width=True, hide_index=True, column_config={"confidence": st.column_config.ProgressColumn("Confidence", format="%.1f%%", min_value=0, max_value=1)})

def analyze_news():
    st.title("Analyze News")
    mode = st.radio("Input type", ["Single headline", "News passage"], horizontal=True)
    placeholder = "Paste a financial headline..." if mode == "Single headline" else "Paste a financial news summary or short article..."
    text = st.text_area("Financial news text", placeholder=placeholder, height=110 if mode == "Single headline" else 220)
    if st.button("Analyze Sentiment", type="primary"):
        if not text.strip():
            st.error("Enter a financial headline or news passage before analyzing.")
            return
        with st.spinner("Running FinBERT inference..."):
            try:
                result = analyze([text])[0]
            except RuntimeError as exc:
                st.error(str(exc)); return
        add_history([result])
        st.markdown(f"<div class='result-card'><span class='small-note'>MODEL CLASSIFICATION</span><h2>{result['sentiment']}</h2><p>Confidence: <b>{result['confidence']:.1%}</b> (a model probability, not a guarantee)</p></div>", unsafe_allow_html=True)
        st.caption("Input analyzed: " + result["text"])
        st.plotly_chart(probability_chart(result), use_container_width=True)
        tone = result["sentiment"].lower()
        st.info(f"Interpretation: The model classifies this text as **{tone}** financial sentiment. This is an NLP assessment of wording and is not investment advice or a stock-price prediction.")

def batch_page():
    st.title("Batch CSV Analysis")
    st.caption("Upload a CSV of financial-news records. Original columns remain in the downloaded enriched file.")
    upload = st.file_uploader("CSV file", type="csv")
    if upload is None: return
    try:
        raw = pd.read_csv(upload)
    except Exception as exc:
        st.error(f"Could not read this CSV: {exc}"); return
    if raw.empty:
        st.error("The CSV has no rows."); return
    st.write(f"**{len(raw):,} rows** · columns: {', '.join(map(str, raw.columns))}")
    st.dataframe(raw.head(), use_container_width=True)
    detected = detect_text_column(raw.columns)
    index = list(raw.columns).index(detected) if detected else 0
    text_col = st.selectbox("Text column to analyze", list(raw.columns), index=index, help="Detected common news text columns when available.")
    remove_duplicates = st.checkbox("Remove duplicate text rows", value=True)
    max_records = st.number_input("Maximum records to analyze", min_value=1, max_value=max(len(raw), 1), value=min(100, len(raw)), step=1)
    batch_size = st.select_slider("Inference batch size", options=[4, 8, 16, 32], value=16)
    if st.button("Analyze CSV", type="primary"):
        try:
            cleaned, skipped, duplicates = clean_batch_dataframe(raw, text_col, remove_duplicates)
        except ValueError as exc:
            st.error(str(exc)); return
        if cleaned.empty:
            st.error("No non-empty text rows remain after cleaning."); return
        selected = cleaned.head(int(max_records)).copy()
        st.info(f"Analyzing {len(selected):,} rows. Skipped {skipped:,} empty rows" + (f" and {duplicates:,} duplicate rows." if remove_duplicates else "."))
        progress = st.progress(0, text="Preparing FinBERT...")
        predictions = []
        try:
            classifier = get_classifier()
            for start in range(0, len(selected), batch_size):
                batch = selected[text_col].tolist()[start:start + batch_size]
                predictions.extend(infer_texts(batch, classifier, batch_size=batch_size))
                finished = min(start + batch_size, len(selected))
                progress.progress(finished / len(selected), text=f"Analyzed {finished:,} of {len(selected):,} rows")
        except Exception as exc:
            progress.empty(); st.error(f"FinBERT analysis failed: {exc}"); return
        progress.empty()
        st.session_state.batch_results = results_dataframe(selected, predictions)
        add_history(predictions)
        st.success("CSV analysis complete.")
    results = st.session_state.get("batch_results")
    if results is None: return
    counts = results["sentiment"].value_counts()
    cards = st.columns(4)
    metric_card(cards[0], "ANALYZED", len(results)); metric_card(cards[1], "POSITIVE", int(counts.get("Positive", 0))); metric_card(cards[2], "NEGATIVE", int(counts.get("Negative", 0))); metric_card(cards[3], "NEUTRAL", int(counts.get("Neutral", 0)))
    c1, c2 = st.columns(2); c1.plotly_chart(donut_chart(results), use_container_width=True); c2.plotly_chart(confidence_chart(results), use_container_width=True)
    st.subheader("Extremes by sentiment score")
    e1, e2 = st.columns(2); e1.dataframe(results.nlargest(3, "sentiment_score")[[text_col, "sentiment_score"]], use_container_width=True, hide_index=True); e2.dataframe(results.nsmallest(3, "sentiment_score")[[text_col, "sentiment_score"]], use_container_width=True, hide_index=True)
    filter_value = st.selectbox("Filter results", ["All", "Positive", "Negative", "Neutral"])
    query = st.text_input("Search text")
    display = results if filter_value == "All" else results[results.sentiment == filter_value]
    if query: display = display[display[text_col].astype(str).str.contains(query, case=False, na=False)]
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.download_button("Download enriched CSV", results.to_csv(index=False).encode("utf-8"), "finsight_enriched_results.csv", "text/csv")
    candidates = [c for c in results.columns if c not in {"sentiment", "confidence", "negative_probability", "neutral_probability", "positive_probability", "sentiment_score"}]
    reference = st.selectbox("Optional reference-label column for evaluation", ["None"] + candidates)
    if reference != "None":
        evaluation = evaluate_predictions(results, reference)
        if evaluation:
            st.caption("Metrics are meaningful only when reference labels are trustworthy and correctly mapped.")
            em = st.columns(4)
            for col, label, value in zip(em, ["Accuracy", "Precision", "Recall", "F1"], [evaluation["accuracy"], evaluation["precision"], evaluation["recall"], evaluation["f1"]]): metric_card(col, label, f"{value:.1%}")
            st.write(f"Evaluated records: {evaluation['records']}"); st.plotly_chart(confusion_chart(evaluation), use_container_width=True)
        else: st.warning("No compatible Positive, Negative, or Neutral reference labels were found in this column.")

def about():
    st.title("About FinSight")
    st.markdown("""**FinSight — Financial News Sentiment Analyzer for Indian Stock Markets** is a college NLP mini-project that performs pretrained-model inference on financial news. It does not train FinBERT, forecast prices, or provide investment advice.

### Problem and NLP pipeline
Financial news is abundant but difficult to scan quickly. FinSight turns news text into an interpretable three-class tone assessment:

`News text → input validation → FinBERT tokenization → transformer inference → sentiment probabilities → visualizations and insights`

Tokenization converts text into model tokens; contextual embeddings let the transformer interpret a word using surrounding language. A finance-specific model is useful because financial phrasing differs from general sentiment. FinBERT is generally more context-aware for this task than lexicon rules, while basic TF-IDF classifiers require labeled data and may miss contextual meaning.

### Data and evaluation
Use your own CSV with fields such as URL, Content, Summary, or Sentiment. `sample_news.csv` contains explicitly synthetic demonstration headlines only. Existing labels are preserved, but are never treated as ground truth unless you select them and know their provenance.

### Model references
- [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)
- [BERT paper](https://arxiv.org/abs/1810.04805)
- [FinBERT research paper](https://arxiv.org/abs/1908.10063)

### Limitations and ethics
Sentiment can miss sarcasm, mixed opinions, nuanced financial language, and Indian company-specific context. Confidence is not accuracy. Financial-news sentiment alone cannot reliably predict price movements; use it as an analytical aid, never investment advice. The model is pretrained and performs inference only.
""")

with st.sidebar:
    st.markdown("## FIN<span style='color:#2dd4bf'>SIGHT</span>", unsafe_allow_html=True)
    st.caption("Financial news sentiment intelligence")
    page = st.radio("Navigation", ["Dashboard", "Analyze News", "Batch CSV Analysis", "About the Project"])
    st.divider(); st.caption("Model: ProsusAI/finbert\n\nFirst inference may download model files.")

{"Dashboard": dashboard, "Analyze News": analyze_news, "Batch CSV Analysis": batch_page, "About the Project": about}[page]()
