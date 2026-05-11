"""
Streamlit Frontend — Smart Feedback Intelligence System
Run: streamlit run frontend/app.py
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import time

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Smart Feedback Intelligence",
    page_icon="🧠",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Feedback Intelligence")
    st.markdown("---")
    mode = st.radio("Mode", ["Single Analysis", "Batch Analysis", "Dashboard"])
    st.markdown("---")
    include_absa = st.toggle("Aspect-Based Sentiment", True)
    include_entities = st.toggle("Named Entity Recognition", True)
    st.markdown("---")
    st.caption("Powered by BERT + XGBoost + FastAPI")

# ── Color mapping ─────────────────────────────────────────
SENTIMENT_COLORS = {"positive": "🟢", "neutral": "🟡", "negative": "🔴"}
URGENCY_COLORS = {"low": "🟢", "medium": "🟡", "high": "🔴"}


def call_analyze(text: str) -> dict:
    try:
        res = requests.post(
            f"{API_URL}/analyze",
            json={"text": text, "include_absa": include_absa, "include_entities": include_entities},
            timeout=15,
        )
        return res.json() if res.status_code == 200 else {"error": res.text}
    except requests.ConnectionError:
        return {"error": "Cannot connect to API. Make sure the FastAPI server is running."}


def render_confidence_bar(label: str, confidence: float, color: str = "#7F77DD"):
    pct = round((confidence or 0) * 100, 1)
    st.markdown(f"""
    <div style='margin-bottom:8px;'>
      <div style='font-size:12px;color:#888;margin-bottom:3px;'>{label} <b style='color:#fff;'>{pct}%</b></div>
      <div style='background:#2a2a2a;border-radius:4px;height:6px;'>
        <div style='width:{pct}%;background:{color};height:6px;border-radius:4px;'></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Single Analysis ───────────────────────────────────────
if mode == "Single Analysis":
    st.title("🔍 Single Feedback Analysis")
    examples = [
        "The delivery was extremely late and the package was damaged.",
        "Customer support was amazing, resolved my issue in minutes!",
        "Product quality is great but billing charged me twice.",
        "I've been waiting 3 weeks for my order with no response from support.",
    ]
    selected = st.selectbox("Try an example:", ["Custom input..."] + examples)
    text_input = st.text_area(
        "Customer Feedback",
        value="" if selected == "Custom input..." else selected,
        height=120,
        placeholder="Enter customer feedback here...",
    )

    if st.button("🚀 Analyze", type="primary", disabled=not text_input.strip()):
        with st.spinner("Analyzing..."):
            result = call_analyze(text_input)

        if "error" in result:
            st.error(result["error"])
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                emoji = SENTIMENT_COLORS.get(result["sentiment"], "⚪")
                st.metric("Sentiment", f"{emoji} {result['sentiment'].title()}")
            with col2:
                st.metric("Category", f"📁 {result['category'].title()}")
            with col3:
                emoji = URGENCY_COLORS.get(result["urgency"], "⚪")
                st.metric("Urgency", f"{emoji} {result['urgency'].title()}")
            with col4:
                churn = result["churn_risk"]
                st.metric("Churn Risk", f"{'⚠️' if churn == '1' else '✅'} {'High' if churn == '1' else 'Low'}")

            st.markdown("---")
            col_left, col_right = st.columns(2)

            with col_left:
                st.subheader("📊 Confidence Scores")
                render_confidence_bar("Sentiment", result.get("sentiment_confidence", 0), "#7F77DD")
                render_confidence_bar("Category", result.get("category_confidence", 0), "#1D9E75")
                render_confidence_bar("Urgency", result.get("urgency_confidence", 0), "#BA7517")
                render_confidence_bar("Churn", result.get("churn_confidence", 0), "#D85A30")

            with col_right:
                if result.get("aspect_sentiments"):
                    st.subheader("🎯 Aspect-Based Sentiment")
                    absa = result["aspect_sentiments"]
                    for aspect, sentiment in absa.items():
                        emoji = SENTIMENT_COLORS.get(sentiment, "⚪")
                        st.markdown(f"**{aspect.title()}**: {emoji} {sentiment.title()}")
                else:
                    st.info("No specific aspects detected in this review.")

            if result.get("entities"):
                st.subheader("🏷️ Named Entities")
                for entity_type, values in result["entities"].items():
                    st.markdown(f"**{entity_type}**: {', '.join(values)}")

            st.caption(f"⚡ Processing time: {result.get('processing_time_ms', 0):.1f}ms")

# ── Batch Analysis ────────────────────────────────────────
elif mode == "Batch Analysis":
    st.title("📦 Batch Feedback Analysis")
    st.info("Upload a CSV file with a 'text' column or paste texts below.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        if "text" not in df.columns:
            st.error("CSV must have a 'text' column.")
        else:
            st.dataframe(df.head(), use_container_width=True)
            if st.button("🚀 Analyze All", type="primary"):
                with st.spinner(f"Analyzing {len(df)} rows..."):
                    try:
                        res = requests.post(
                            f"{API_URL}/batch",
                            json={"texts": df["text"].tolist()[:100], "include_absa": False},
                            timeout=60,
                        )
                        data = res.json()
                        st.success(f"✅ Analyzed {data['total']} reviews")

                        summary = data["summary"]
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            fig = px.pie(
                                values=list(summary["sentiment_distribution"].values()),
                                names=list(summary["sentiment_distribution"].keys()),
                                title="Sentiment Distribution",
                                color_discrete_sequence=["#1D9E75", "#BA7517", "#D85A30"],
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        with col2:
                            fig = px.bar(
                                x=list(summary["category_distribution"].keys()),
                                y=list(summary["category_distribution"].values()),
                                title="Category Distribution",
                                color_discrete_sequence=["#7F77DD"],
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        with col3:
                            fig = px.pie(
                                values=list(summary["urgency_distribution"].values()),
                                names=list(summary["urgency_distribution"].keys()),
                                title="Urgency Distribution",
                                color_discrete_sequence=["#1D9E75", "#BA7517", "#D85A30"],
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        col1, col2 = st.columns(2)
                        col1.metric("High Urgency Cases", summary["high_urgency_count"])
                        col2.metric("Churn Risk %", f"{summary['churn_rate_pct']}%")

                        # Download results
                        results_df = pd.DataFrame([
                            {
                                "text": r["text"],
                                "sentiment": r["sentiment"],
                                "category": r["category"],
                                "urgency": r["urgency"],
                                "churn_risk": r["churn_risk"],
                            }
                            for r in data["results"]
                        ])
                        st.download_button(
                            "⬇️ Download Results CSV",
                            results_df.to_csv(index=False),
                            "feedback_analysis.csv",
                            "text/csv",
                        )
                    except Exception as e:
                        st.error(f"API error: {e}")

# ── Dashboard ─────────────────────────────────────────────
elif mode == "Dashboard":
    st.title("📊 Analytics Dashboard")
    st.info("Connect your database or upload historical analysis results.")

    # Demo charts
    demo_data = {
        "sentiment": {"positive": 420, "neutral": 180, "negative": 300},
        "category": {"product": 280, "delivery": 220, "support": 180, "billing": 140, "returns": 80},
        "urgency": {"low": 520, "medium": 280, "high": 100},
    }

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews", "900", "+12%")
    col2.metric("Avg Sentiment Score", "0.62", "+0.05")
    col3.metric("High Urgency", "100", "-8%")
    col4.metric("Churn Risk Rate", "28%", "-3%")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            x=list(demo_data["category"].keys()),
            y=list(demo_data["category"].values()),
            title="Issues by Category",
            color=list(demo_data["category"].values()),
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.pie(
            values=list(demo_data["sentiment"].values()),
            names=list(demo_data["sentiment"].keys()),
            title="Sentiment Breakdown",
            color_discrete_sequence=["#1D9E75", "#BA7517", "#D85A30"],
        )
        st.plotly_chart(fig, use_container_width=True)
