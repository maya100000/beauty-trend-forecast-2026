import streamlit as st
import pandas as pd
import altair as alt

st.title("Dashboard")

skin_path = "notebooks/data/processed/skincare_2026_forecast.csv"
makeup_path = "notebooks/data/processed/makeup_2026_forecast.csv"
metrics_path = "notebooks/data/processed/model_evaluation_metrics.csv"
prod_rank_path = "notebooks/data/processed/product_forecast_6m_rank.csv"

skin = pd.read_csv(skin_path)
makeup = pd.read_csv(makeup_path)
metrics = pd.read_csv(metrics_path)
prod_rank = pd.read_csv(prod_rank_path)

col1, col2 = st.columns(2, gap="large")  

with col1:
    st.subheader("Top Skincare Categories - 2026")

    score_col = "trend_2026_score_v2" if "trend_2026_score_v2" in skin.columns else skin.columns[-1]
    top_skin = skin.sort_values(score_col, ascending=False).head(12)

    chart_skin = (
        alt.Chart(top_skin)
        .mark_bar()
        .encode(
            y=alt.Y("category:N", sort="-x", title="Category"),
            x=alt.X(f"{score_col}:Q", title="Trend Score"),
            tooltip=[
                alt.Tooltip("category:N", title="Category"),
                alt.Tooltip(f"{score_col}:Q", title="Trend Score", format=".3f"),
            ],
        )
    )

    st.altair_chart(chart_skin, use_container_width=True)

   
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    st.caption(
        "Trend Score = weighted blend of: review volume growth + Google Trends growth/level + editorial mentions. "
        "Higher score ⇒ stronger momentum toward 2026."
    )

with col2:
    st.subheader("Makeup Themes - 2026")

    score_col2 = "makeup_2026_score" if "makeup_2026_score" in makeup.columns else makeup.columns[-1]
    top_makeup = makeup.sort_values(score_col2, ascending=False).head(12)

    chart_makeup = (
        alt.Chart(top_makeup)
        .mark_bar()
        .encode(
            y=alt.Y("theme:N", sort="-x", title="Theme"),
            x=alt.X(f"{score_col2}:Q", title="Score"),
            tooltip=[
                alt.Tooltip("theme:N", title="Theme"),
                alt.Tooltip(f"{score_col2}:Q", title="Score", format=".3f"),
            ],
        )
    )

    st.altair_chart(chart_makeup, use_container_width=True)

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    st.caption(
        "Makeup 2026 Score = 0.55×Luxxify trend + 0.30×Google Trends + 0.15×Editorial. "
        "Higher score ⇒ theme likely to be more in-demand in 2026."
    )

st.divider()

st.subheader("Model Evaluation (Product-level Forecast)")
st.dataframe(metrics)

st.subheader("Top Products by Forecasted 6M Uplift")
st.dataframe(prod_rank.head(15))