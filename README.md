# Makeup & Skincare Trend Forecast - 2026

**Applied Data Science | Forecasting | Streamlit | LLM + RAG**

---

## Project Overview

This project forecasts **skincare categories, makeup themes, and product-level demand trends toward 2026** by combining multiple real-world signals and validated machine learning models.

The goal is to demonstrate an **end-to-end applied data science project**:
from raw data and feature engineering,  
through forecasting and model validation,  
to an interactive dashboard and LLM-powered analytical interface.

---

## What This Project Does

### Trend Forecasting (2026)

Predicts which **skincare categories** and **makeup themes** are most likely to grow toward 2026.

Signals used:

- Review volume and growth (proxy for consumer demand)
- Google Trends (interest level + momentum)
- Editorial / content mentions

These signals are combined into a weighted **Trend Score**.

---

### Product-Level Demand Forecasting

- Forecasts short-term demand (T+1 to T+6 months) per product
- Uses time-series features (lags, rolling averages)
- Validated against a **naive baseline**
- Evaluation metrics:
  - MAE
  - MAPE

The model demonstrates improved accuracy compared to the baseline.

---

### Interactive Dashboard (Streamlit)

The app includes:

- Top skincare categories for 2026
- Top makeup themes for 2026
- Product-level forecast rankings
- Model evaluation transparency
- Forecast visualization (Actual vs Forecast)

---

### LLM + RAG: “Ask the Project”

An LLM-powered interface that allows users to ask natural-language questions such as:

- _Why are Face Serums trending above Moisturizers?_
- _Which makeup themes are expected to grow toward 2026?_
- _How reliable is the forecast model?_

The LLM:

- Reads project outputs (CSV-based RAG)
- Uses computation tools for numeric comparisons
- Grounds answers in real model results (no hallucination)

---

## 🧠 Technologies Used

**Data & Modeling**

- Python
- pandas, numpy
- scikit-learn
- Time-series feature engineering
- Baseline vs model validation

**Visualization & UI**

- Streamlit
- Altair
- Matplotlib

**LLM & RAG**

- OpenAI API (`gpt-4o-mini`)
- Retrieval from structured CSV outputs
- Tool-based numeric reasoning

---

## 📁 Project Structure

```text
makeup-trend-forecast/
│
├── app.py
├── pages/
│   ├── 1_Dashboard.py
│   ├── 2_Product_Forecast.py
│   └── 3_Ask_the_Project.py
│
├── data/
│   └── processed/
│       ├── skincare_2026_forecast.csv
│       ├── makeup_2026_forecast.csv
│       ├── product_forecast_6m_rank.csv
│       └── model_evaluation_metrics.csv
│
├── notebooks/
│   └── feature_engineering_and_modeling.ipynb
│
├── rag/
│   └── knowledge.md
│
├── requirements.txt
└── README.md
```
