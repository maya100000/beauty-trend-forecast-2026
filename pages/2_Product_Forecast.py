import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np

st.title("Product Forecast (T+6)")

pm_path = "notebooks/data/processed/product_monthly_table.csv"  # ניצור עוד רגע
model_path = "notebooks/models/product_demand_rf.pkl"

pm = pd.read_csv(pm_path)
model = joblib.load(model_path)

# month כ-period
pm["month"] = pm["month"].astype(str).apply(lambda x: pd.Period(x, freq="M"))

# בחירת מוצר
products = pm[["product_id", "product_name_x"]].drop_duplicates()
products["label"] = products["product_name_x"] + " (" + products["product_id"].astype(str) + ")"
label_to_id = dict(zip(products["label"], products["product_id"]))

choice = st.selectbox("Select a product", sorted(label_to_id.keys()))
pid = label_to_id[choice]

hist = pm[pm["product_id"] == pid].sort_values("month").copy()

# --- recursive forecast ---
def forecast_t_plus_6_for_product(hist_df, model, steps=6):
    hist_df = hist_df.sort_values("month").copy()
    y = hist_df["review_count"].tolist()
    if len(y) < 6:
        return None

    last_month = hist_df["month"].max()
    lag1, lag2, lag3 = y[-1], y[-2], y[-3]
    roll3 = float(np.mean([lag1, lag2, lag3]))

    preds = []
    cur_month = last_month

    for _ in range(steps):
        X_next = pd.DataFrame([{
            "lag_1": lag1,
            "lag_2": lag2,
            "lag_3": lag3,
            "rolling_3": roll3
        }])
        yhat = float(model.predict(X_next)[0])
        yhat = max(0.0, yhat)

        cur_month = cur_month + 1
        preds.append({"month": cur_month, "yhat": yhat})

        lag3, lag2, lag1 = lag2, lag1, yhat
        roll3 = float(np.mean([lag1, lag2, lag3]))

    return pd.DataFrame(preds)

fc = forecast_t_plus_6_for_product(hist, model, steps=6)


st.caption(
    "This chart demonstrates short-term demand forecasting (T+1 to T+6) "
    "used to validate the model's ability to learn product-level demand dynamics. "
    "These dynamics are later aggregated into long-term (2026) trend signals."
)
hist_tail = hist.tail(24).copy()

fig = plt.figure(figsize=(10,4))
plt.plot(hist_tail["month"].astype(str), hist_tail["review_count"], marker="o", label="Actual (last 24M)")
plt.plot(fc["month"].astype(str), fc["yhat"], marker="o", linestyle="--", label="Forecast (T+1..T+6)")
plt.xticks(rotation=45)
plt.ylabel("Monthly review count (proxy demand)")
plt.legend()
plt.tight_layout()
st.pyplot(fig)

st.subheader("Product Demand Dynamics - Short-Term Forecast (Validation)")
st.dataframe(fc.assign(month=fc["month"].astype(str)))
st.info(
    "Reliable long-term trend forecasts require models that capture short-term demand behavior. "
    "This validation step ensures the trend signals used for 2026 are grounded in real product dynamics."
)
