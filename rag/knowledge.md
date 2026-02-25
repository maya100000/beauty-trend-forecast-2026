# Project: Beauty Trends Forecast for 2026

## Goal

Forecast beauty trends for 2026 using:

1. Consumer review signals (Sephora)
2. Google Trends signals
3. Editorial signals (scraped articles)
4. Product-level demand forecasting (proxy: monthly review volume)

## Key Outputs

- Skincare: category-level 2026 trend score (trend_2026_score_v2)
- Makeup: theme-level 2026 score (makeup_2026_score)
- Product forecast: next 6 months review-volume forecast (proxy demand), per product
- Evaluation: model vs naive baseline

## Metrics

MAE = average absolute error in monthly review count (proxy demand).
MAPE = percentage error; less stable when monthly volume is small.
We primarily rely on MAE.

## Visualization note

Forecast plots focus on the last 24 months for interpretability,
while training retains historical outliers to preserve real demand shocks.
