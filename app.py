import streamlit as st

st.set_page_config(page_title="Beauty Trends Forecast 2026", layout="wide")

st.title("Beauty Trends Forecast - 2026")
st.write(
    "A portfolio Data Science project combining consumer reviews, Google Trends, and editorial signals "
    "to forecast skincare and makeup trends for 2026, plus product-level demand forecasting."
)

st.markdown("### Pages")
st.markdown("- **Dashboard**: Skincare & Makeup top trends + evaluation metrics")
st.markdown("- **Product Forecast**: Actual vs Forecast (T+1..T+6) for a selected product")
st.markdown("- **Ask the Project (LLM)**: Ask questions about methodology and results (RAG)")