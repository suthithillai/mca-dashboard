"""
MCA E-Consultation Dashboard
Checkbox Outputs + Pie Charts + Wordclouds + Bar Charts + Rough Scores + Summary
"""

import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import spacy
from nltk.corpus import stopwords
import nltk
from textblob import TextBlob
import plotly.express as px
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Ensure NLTK
nltk.download('punkt')
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

# spaCy
try:
    nlp = spacy.load('en_core_web_sm')
except:
    nlp = None

# ------------------ Utilities ------------------

def safe_read_csv(file):
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(file, sep=sep, engine="python", on_bad_lines="skip")
            if not df.empty:
                return df
        except Exception:
            continue
    return pd.DataFrame(columns=["comments", "topic"])

def preprocess_text(text: str) -> str:
    try:
        if not isinstance(text, str):
            text = str(text)
        if nlp:
            doc = nlp(text)
            tokens = []
            for token in doc:
                if token.is_punct or token.is_space:
                    continue
                if token.text.lower() in STOPWORDS:
                    continue
                lemma = token.lemma_.lower().strip()
                if len(lemma) <= 1:
                    continue
                tokens.append(lemma)
            return ' '.join(tokens)
        else:
            return text.lower()
    except Exception:
        return str(text).lower()

@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

@st.cache_resource
def load_summary_pipeline():
    try:
        return pipeline("summarization", model="facebook/bart-large-cnn")
    except:
        return None

def map_star_to_label(pred):
    label = pred[0]['label']
    if label in ["4 stars", "5 stars"]:
        return "positive"
    elif label == "3 stars":
        return "neutral"
    else:
        return "negative"

def generate_wordcloud_matplotlib(texts, sentiment):
    if not texts.strip():
        return None
    palette_map = {
        "positive": "Greens",
        "neutral": "Blues",
        "negative": "Reds"
    }
    wc = WordCloud(width=800, height=400,
                   background_color="white",
                   colormap=palette_map.get(sentiment, "viridis"),
                   stopwords=STOPWORDS).generate(texts)

    fig, ax = plt.subplots(figsize=(8,4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig

# ------------------ Streamlit Main ------------------

def main():
    st.set_page_config(page_title="MCA E-Consultation Dashboard", page_icon="📊", layout="wide")

    # ---------- Background Choice ----------
    bg_choice = st.sidebar.radio("🎨 Choose Background Style", ["Light Mode", "Indian Flag Gradient"])
    if bg_choice == "Light Mode":
        st.markdown("""<style>.stApp { background: white; font-family: "Segoe UI"; }</style>""", unsafe_allow_html=True)
    elif bg_choice == "Indian Flag Gradient":
        st.markdown("""<style>.stApp {
            background: linear-gradient(180deg, #FF9933 0%, #FFFFFF 50%, #138808 100%);
            font-family: "Segoe UI"; background-attachment: fixed;
        }</style>""", unsafe_allow_html=True)

    # ---------- Header ----------
    st.markdown("<h1 style='text-align:center;'>📊 MCA E-Consultation — AI Insights Dashboard</h1>", unsafe_allow_html=True)

    # Sidebar
    st.sidebar.header("⚙️ Settings")
    uploaded = st.sidebar.file_uploader("Upload Dataset (CSV with comments, topic)", type=['csv'])

    if uploaded is None:
        st.info("⬆️ Upload a dataset to start.")
        st.stop()

    df = safe_read_csv(uploaded)
    if "comments" not in df.columns:
        st.error("Dataset must contain a 'comments' column.")
        st.stop()
    if "topic" not in df.columns:
        df["topic"] = "General"

    # Preprocess
    df["clean"] = df["comments"].astype(str).apply(preprocess_text)

    # Load models
    sentiment_pipe = load_sentiment_pipeline()
    summary_pipe = load_summary_pipeline()

    # Sentiment Prediction
    preds = []
    for t in df["clean"]:
        try:
            res = sentiment_pipe(t[:512])
            lab = map_star_to_label(res)
        except:
            pol = TextBlob(t).sentiment.polarity
            lab = "negative" if pol < -0.05 else "positive" if pol > 0.05 else "neutral"
        preds.append(lab)
    df["predicted_sentiment"] = preds

    # Tabs
    tab1, tab2 = st.tabs(["📌 Topic-wise Analysis", "📈 Global Insights"])

    # ---------------- Topic-wise ----------------
    with tab1:
        topics = df["topic"].unique()
        selected_topic = st.selectbox("Select a Topic", topics)
        grp = df[df["topic"] == selected_topic]

        if grp.empty:
            st.warning("⚠️ No comments available for this topic.")
            st.stop()

        # KPI Cards
        col1, col2, col3 = st.columns(3)
        col1.metric("😊 Positive", (grp['predicted_sentiment']=='positive').sum())
        col2.metric("😐 Neutral", (grp['predicted_sentiment']=='neutral').sum())
        col3.metric("😞 Negative", (grp['predicted_sentiment']=='negative').sum())

        # Rough Score
        if st.checkbox("Show Rough Score"):
            score_map = {"positive": 1, "neutral": 0, "negative": -1}
            rough_score = grp["predicted_sentiment"].map(score_map).mean()
            st.subheader("📊 Rough Sentiment Score")
            st.info(f"{rough_score:.2f}")

        # Pie Chart for topic
        if st.checkbox("Show Pie Chart for Topic"):
            st.subheader("🥧 Sentiment Distribution for Topic")
            dist = grp["predicted_sentiment"].value_counts().reset_index()
            dist.columns = ["Sentiment", "Count"]
            fig_pie = px.pie(dist, names="Sentiment", values="Count", hole=0.3,
                             color="Sentiment",
                             color_discrete_map={"positive":"#138808","neutral":"#000080","negative":"#e74c3c"})
            st.plotly_chart(fig_pie, use_container_width=True)

        # Wordclouds
        if st.checkbox("Show Wordclouds"):
            st.subheader("☁️ Wordclouds by Sentiment")
            for sentiment in ["positive","neutral","negative"]:
                texts = " ".join(grp[grp["predicted_sentiment"]==sentiment]["clean"])
                if texts.strip():
                    st.markdown(f"**{sentiment.capitalize()} Feedback**")
                    fig_wc = generate_wordcloud_matplotlib(texts, sentiment)
                    if fig_wc:
                        st.pyplot(fig_wc)

        # Bar Charts
        if st.checkbox("Show Bar Chart for Sentiments"):
            st.subheader("📊 Top Keywords per Sentiment")
            for sentiment in ["positive", "neutral", "negative"]:
                texts = " ".join(grp[grp["predicted_sentiment"]==sentiment]["clean"])
                if texts.strip():
                    words = [w for w in texts.split() if w not in STOPWORDS]
                    freq = Counter(words).most_common(10)
                    if freq:
                        df_freq = pd.DataFrame(freq, columns=["Word", "Count"])
                        fig_bar = px.bar(df_freq, x="Word", y="Count",
                                         title=f"Top Words in {sentiment.capitalize()} Feedback",
                                         color="Count", color_continuous_scale="viridis")
                        st.plotly_chart(fig_bar, use_container_width=True)

        # Summary
        if st.checkbox("Show Summary"):
            st.subheader("📝 Abstractive Summary")
            try:
                if summary_pipe:
                    top_comments = " ".join(
                        grp.sort_values(by="comments", key=lambda x: x.str.len(), ascending=False)["comments"].head(5)
                    )
                    summary = summary_pipe(top_comments[:2000], max_length=120, min_length=40, do_sample=False)[0]['summary_text']
                    st.info(summary)
                else:
                    st.warning("⚠️ Summarization model not available.")
            except:
                st.warning("⚠️ Could not generate abstractive summary. Showing extractive fallback.")
                top_comments = grp["comments"].head(3).tolist()
                st.write(" ".join(top_comments))

    # ---------------- Global Insights ----------------
    with tab2:
        if st.checkbox("Show Global Pie Chart"):
            st.subheader("🥧 Global Sentiment Distribution")
            dist_global = df["predicted_sentiment"].value_counts().reset_index()
            dist_global.columns = ["Sentiment", "Count"]
            fig_global_pie = px.pie(dist_global, names="Sentiment", values="Count", hole=0.3,
                                    color="Sentiment",
                                    color_discrete_map={"positive":"#138808","neutral":"#000080","negative":"#e74c3c"})
            st.plotly_chart(fig_global_pie, use_container_width=True)

        if st.checkbox("Show Global Insights Text"):
            st.subheader("📖 Overall Global Insights")
            insights = []
            for topic in df["topic"].unique():
                grp = df[df["topic"] == topic]
                pos = (grp["predicted_sentiment"]=="positive").sum()
                neu = (grp["predicted_sentiment"]=="neutral").sum()
                neg = (grp["predicted_sentiment"]=="negative").sum()
                total = len(grp)
                if total == 0: 
                    continue
                pos_pct, neu_pct, neg_pct = (pos/total)*100, (neu/total)*100, (neg/total)*100
                if pos_pct > 50:
                    insights.append(f"✅ **{topic}** → Mostly Positive ({pos_pct:.1f}%).")
                elif neg_pct > 40:
                    insights.append(f"⚠️ **{topic}** → High Negative ({neg_pct:.1f}%).")
                elif neu_pct > 40:
                    insights.append(f"ℹ️ **{topic}** → Mostly Neutral ({neu_pct:.1f}%).")
                else:
                    insights.append(f"🔀 **{topic}** → Mixed ({pos_pct:.1f}% pos, {neg_pct:.1f}% neg, {neu_pct:.1f}% neu).")
            st.write("\n".join(insights))

if __name__ == "__main__":
    main()
