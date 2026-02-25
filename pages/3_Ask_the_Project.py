import os
import re
import streamlit as st
import pandas as pd
from openai import OpenAI

st.set_page_config(page_title="Ask the Project", layout="wide")
st.title("Ask the Project (LLM + RAG + Tools)")

# --------- OpenAI client ----------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY is not set. Restart terminal / set env var and rerun Streamlit.")
    st.stop()

client = OpenAI(api_key=api_key)

# --------- Paths ----------
BASE = "notebooks/data/processed"
skin_path = f"{BASE}/skincare_2026_forecast.csv"
makeup_path = f"{BASE}/makeup_2026_forecast.csv"
metrics_path = f"{BASE}/model_evaluation_metrics.csv"
prod_rank_path = f"{BASE}/product_forecast_6m_rank.csv"

# --------- Load data ----------
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

skin = load_csv(skin_path)
makeup = load_csv(makeup_path)
metrics = load_csv(metrics_path)
prod_rank = load_csv(prod_rank_path)

skin_score_col = "trend_2026_score_v2" if "trend_2026_score_v2" in skin.columns else skin.columns[-1]
makeup_score_col = "makeup_2026_score" if "makeup_2026_score" in makeup.columns else makeup.columns[-1]

# --------- Build compact "project context" ----------
def df_top_as_text(df, sort_col=None, top_n=8, cols=None, title=""):
    d = df.copy()
    if sort_col and sort_col in d.columns:
        d = d.sort_values(sort_col, ascending=False)
    if cols:
        d = d[cols]
    d = d.head(top_n)
    return f"{title}\n{d.to_string(index=False)}\n"

context_parts = []
context_parts.append(
    "PROJECT SUMMARY:\n"
    "- Goal: Forecast skincare & makeup trends toward 2026 using a blend of signals.\n"
    "- Signals: review dynamics (proxy demand), Google Trends (level & growth), editorial mentions.\n"
    "- Plus: product-level short-term demand model validated vs naive baseline.\n"
)

context_parts.append(df_top_as_text(
    skin, sort_col=skin_score_col, top_n=10,
    cols=["category", skin_score_col],
    title="TOP SKINCARE CATEGORIES (2026 score):"
))

context_parts.append(df_top_as_text(
    makeup, sort_col=makeup_score_col, top_n=10,
    cols=["theme", makeup_score_col],
    title="TOP MAKEUP THEMES (2026 score):"
))

prod_cols = [c for c in ["product_id","product_name_x","predicted_next","uplift_6m","rank"] if c in prod_rank.columns]
if not prod_cols:
    prod_cols = prod_rank.columns[:5].tolist()

context_parts.append(df_top_as_text(
    prod_rank, sort_col=prod_cols[-1] if len(prod_cols) else None, top_n=10,
    cols=prod_cols,
    title="TOP PRODUCTS (forecast ranking / uplift):"
))

context_parts.append("MODEL EVALUATION METRICS:\n" + metrics.to_string(index=False) + "\n")
PROJECT_CONTEXT = "\n".join(context_parts)

with st.expander("Show project context sent to the LLM"):
    st.text(PROJECT_CONTEXT)

# =========================
# --------- TOOLS ---------
# =========================

def tool_model_results(metrics_df: pd.DataFrame) -> str:
    cols = set(metrics_df.columns)

    if {"metric", "value"}.issubset(cols):
        d = dict(zip(metrics_df["metric"].astype(str), metrics_df["value"]))
        return (
            "Model validation results:\n"
            f"- Baseline (naive) MAE: {d.get('baseline_mae','NA')}\n"
            f"- Model MAE: {d.get('model_mae','NA')}\n"
            f"- Baseline MAPE: {d.get('baseline_mape','NA')}\n"
            f"- Model MAPE: {d.get('model_mape','NA')}\n"
            "Interpretation: lower is better. Model should beat baseline."
        )

    if "baseline_mae" in cols and "model_mae" in cols:
        row = metrics_df.iloc[0]
        return (
            "Model validation results:\n"
            f"- Baseline (naive) MAE: {row['baseline_mae']}\n"
            f"- Model MAE: {row['model_mae']}\n"
            f"- Baseline MAPE: {row.get('baseline_mape','NA')}\n"
            f"- Model MAPE: {row.get('model_mape','NA')}\n"
            "Interpretation: lower is better. Model should beat baseline."
        )

    return "Model metrics format not recognized. (Check metrics columns.)"


def tool_top_trends(domain: str, n: int) -> str:
    domain = domain.lower().strip()
    if domain == "skincare":
        score_col = skin_score_col
        top = skin.sort_values(score_col, ascending=False).head(n)[["category", score_col]]
        return top.to_string(index=False)

    if domain == "makeup":
        score_col = makeup_score_col
        top = makeup.sort_values(score_col, ascending=False).head(n)[["theme", score_col]]
        return top.to_string(index=False)

    return "domain must be skincare or makeup"


def tool_compare_items(domain: str, a: str, b: str) -> str:
    domain = domain.lower().strip()

    if domain == "skincare":
        ra = skin[skin["category"] == a]
        rb = skin[skin["category"] == b]
        if ra.empty or rb.empty:
            return "One of the skincare categories was not found."
        sa, sb = float(ra.iloc[0][skin_score_col]), float(rb.iloc[0][skin_score_col])
        return f"{a}: {sa:.3f}\n{b}: {sb:.3f}\nDiff (A-B): {(sa-sb):.3f}"

    if domain == "makeup":
        ra = makeup[makeup["theme"] == a]
        rb = makeup[makeup["theme"] == b]
        if ra.empty or rb.empty:
            return "One of the makeup themes was not found."
        sa, sb = float(ra.iloc[0][makeup_score_col]), float(rb.iloc[0][makeup_score_col])
        return f"{a}: {sa:.3f}\n{b}: {sb:.3f}\nDiff (A-B): {(sa-sb):.3f}"

    return "domain must be skincare or makeup"


def tool_product_forecast(product_id: str, horizon: int = 6) -> str:
    pr = prod_rank[prod_rank["product_id"].astype(str) == str(product_id)]
    if pr.empty:
        return "Product not found in product_forecast_6m_rank.csv"
    cols = [c for c in ["product_id","product_name_x","predicted_next","uplift_6m"] if c in pr.columns]
    if not cols:
        cols = pr.columns[:6].tolist()
    return pr[cols].head(1).to_string(index=False)


def tool_explain_score(domain: str, item: str) -> str:
    domain = domain.lower().strip()

    if domain == "skincare":
        row = skin[skin["category"] == item]
        if row.empty:
            return "Category not found."
        row = row.iloc[0]
        candidate_cols = [c for c in skin.columns if any(k in c.lower() for k in ["google","editorial","volume","growth","sentiment","review"])]
        out = {"category": item, "score": row[skin_score_col]}
        for c in candidate_cols[:10]:
            out[c] = row.get(c, None)
        return "\n".join([f"{k}: {v}" for k, v in out.items()])

    if domain == "makeup":
        row = makeup[makeup["theme"] == item]
        if row.empty:
            return "Theme not found."
        row = row.iloc[0]
        candidate_cols = [c for c in makeup.columns if any(k in c.lower() for k in ["google","editorial","growth","level","lux","score"])]
        out = {"theme": item, "score": row[makeup_score_col]}
        for c in candidate_cols[:10]:
            out[c] = row.get(c, None)
        return "\n".join([f"{k}: {v}" for k, v in out.items()])

    return "domain must be skincare or makeup"


TOOLS_HELP = """
Available tools (write exactly in this format):
TOOL: model_results
TOOL: top_trends domain=skincare n=10
TOOL: top_trends domain=makeup n=10
TOOL: compare_items domain=skincare a="Face Serums" b="Moisturizers"
TOOL: compare_items domain=makeup a="Blush/Cheeks" b="Lips"
TOOL: product_forecast product_id="P309308" horizon=6
TOOL: explain_score domain=skincare item="Face Serums"
TOOL: explain_score domain=makeup item="Blush/Cheeks"
"""


def run_tool(tool_line: str) -> str:
    m = re.match(r"TOOL:\s*(\w+)\s*(.*)", tool_line.strip())
    if not m:
        return "Invalid tool format."
    name, args_str = m.group(1), m.group(2)

    def get_arg(key, default=None):
        mm = re.search(rf'{key}\s*=\s*(".*?"|\S+)', args_str)
        if not mm:
            return default
        val = mm.group(1)
        if val.startswith('"') and val.endswith('"'):
            val = val[1:-1]
        return val

    if name == "model_results":
        return tool_model_results(metrics)

    if name == "top_trends":
        domain = get_arg("domain", "skincare")
        n = int(get_arg("n", "10"))
        return tool_top_trends(domain, n)

    if name == "compare_items":
        domain = get_arg("domain", "skincare")
        a = get_arg("a", "")
        b = get_arg("b", "")
        return tool_compare_items(domain, a, b)

    if name == "product_forecast":
        pid = get_arg("product_id", "")
        horizon = int(get_arg("horizon", "6"))
        return tool_product_forecast(pid, horizon)

    if name == "explain_score":
        domain = get_arg("domain", "skincare")
        item = get_arg("item", "")
        return tool_explain_score(domain, item)

    return "Unknown tool."


# =========================
# --------- UI ------------
# =========================

st.divider()
st.subheader("Ask")

suggestions = [
    "Why are Face Serums ranked above Moisturizers?",
    "Which skincare categories have the strongest 2026 momentum and why?",
    "Which makeup themes are expected to grow toward 2026?",
    "How reliable is the product-level forecast model vs baseline?",
    "Give me a short executive summary of 2026 skincare & makeup trends."
]

question = st.text_area("Your question", value=suggestions[0], height=110)

st.caption("Optional: Compare 2 skincare categories numerically (extra facts for the answer)")
all_cats = sorted(skin["category"].dropna().astype(str).unique().tolist()) if "category" in skin.columns else []
c1, c2 = st.columns(2)
with c1:
    cat_a = st.selectbox("Category A", all_cats, index=0 if all_cats else 0)
with c2:
    cat_b = st.selectbox("Category B", all_cats, index=1 if len(all_cats) > 1 else 0)

use_compare = st.checkbox("Include numeric comparison in the answer", value=True)

if st.button("Ask LLM"):
    extra_facts = ""
    if use_compare and cat_a and cat_b and cat_a != cat_b:
        
        ra = skin[skin["category"] == cat_a]
        rb = skin[skin["category"] == cat_b]
        if not ra.empty and not rb.empty:
            sa = float(ra.iloc[0][skin_score_col])
            sb = float(rb.iloc[0][skin_score_col])
            extra_facts = (
                "\nNUMERIC FACTS (computed locally):\n"
                f"- {cat_a} score = {sa:.3f}\n"
                f"- {cat_b} score = {sb:.3f}\n"
                f"- difference (A-B) = {(sa-sb):.3f}\n"
            )

   
    prompt = f"""
You are an applied data science assistant.
If you need exact numbers, rankings, comparisons, or model evaluation results, you MUST request a tool.

{TOOLS_HELP}

Project context:
{PROJECT_CONTEXT}

{extra_facts}

User question:
{question}

RULES:
- If you need a tool, respond ONLY with exactly one line that starts with TOOL:
- Otherwise, answer normally in 5-10 bullet points.
"""

    with st.spinner("Thinking..."):
        first = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        assistant_text = first.choices[0].message.content.strip()

   
    if assistant_text.startswith("TOOL:"):
        tool_output = run_tool(assistant_text)

        with st.spinner("Using tool + writing final answer..."):
            followup = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": assistant_text},
                    {"role": "user", "content": f"Tool output:\n{tool_output}\n\nNow answer the user clearly in bullets."},
                ],
            )
        st.subheader("Answer")
        st.write(followup.choices[0].message.content)

    else:
        st.subheader("Answer")
        st.write(assistant_text)