import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.explainability.reason_codes import ReasonCodeExplainer
from src.config import TARGET_COL

# ── config ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Fraud Analyst Console", page_icon="🛡️", layout="wide")

ALERTS_PATH = "dashboard/data/alert_queue.parquet"
FEEDBACK_PATH = "dashboard/data/analyst_feedback.json"


# ── data loading (cached) ───────────────────────────────────────────
@st.cache_data
def load_alerts():
    return pd.read_parquet(ALERTS_PATH)


@st.cache_resource
def load_explainer():
    return ReasonCodeExplainer(
        model_path="models/lgbm_champion.txt",
        explainer_path="models/shap_explainer.pkl",
    )


@st.cache_data
def load_threshold():
    return json.loads(Path("models/operating_threshold.json").read_text())["chosen_threshold"]


def load_feedback() -> dict:
    if Path(FEEDBACK_PATH).exists():
        return json.loads(Path(FEEDBACK_PATH).read_text())
    return {}


def save_feedback(feedback: dict):
    Path(FEEDBACK_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(FEEDBACK_PATH).write_text(json.dumps(feedback, indent=2))


# ── feature columns for scoring ─────────────────────────────────────
@st.cache_data
def get_feature_cols():
    return json.loads(Path("models/feature_cols.json").read_text())


# ── main ────────────────────────────────────────────────────────────
alerts = load_alerts()
explainer = load_explainer()
threshold = load_threshold()
feature_cols = get_feature_cols()
feedback = load_feedback()

st.title("🛡️ Fraud Analyst Console")
st.caption(f"Operating threshold: {threshold:.3f}  ·  {len(alerts)} alerts in queue")

# ── sidebar: queue summary ──────────────────────────────────────────
with st.sidebar:
    st.header("Queue summary")
    n_alerts = len(alerts)
    n_reviewed = len(feedback)
    st.metric("Total alerts", n_alerts)
    st.metric("Reviewed", n_reviewed)
    st.metric("Remaining", n_alerts - n_reviewed)

    if n_reviewed > 0:
        tp = sum(1 for v in feedback.values() if v == "fraud")
        fp = sum(1 for v in feedback.values() if v == "legit")
        st.metric("Confirmed fraud", tp)
        st.metric("False alarms", fp)
        if (tp + fp) > 0:
            st.metric("Analyst precision", f"{tp/(tp+fp)*100:.1f}%")

    st.divider()
    show_reviewed = st.checkbox("Show reviewed alerts", value=False)

# ── main: alert queue ───────────────────────────────────────────────
tab1, tab2 = st.tabs(["📋 Alert Queue", "📊 Queue Analytics"])

with tab1:
    # filter
    display_alerts = alerts.copy()
    if not show_reviewed:
        reviewed_ids = set(int(k) for k in feedback.keys())
        display_alerts = display_alerts[~display_alerts["TransactionID"].isin(reviewed_ids)]

    if len(display_alerts) == 0:
        st.success("🎉 All alerts reviewed!")
    else:
        st.subheader(f"Alerts to review ({len(display_alerts)})")

        # select an alert to inspect
        for idx, row in display_alerts.head(20).iterrows():
            tid = int(row["TransactionID"])
            score = row["fraud_score"]

            # color by score
            score_color = "🔴" if score > 0.5 else "🟠" if score > 0.25 else "🟡"

            with st.expander(
                f"{score_color} Transaction {tid}  ·  Score {score:.3f}  ·  ${row['TransactionAmt']:,.0f}",
                expanded=False,
            ):
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.markdown("**Transaction details**")
                    st.write(f"Amount: ${row['TransactionAmt']:,.2f}")
                    st.write(f"Product: {row.get('ProductCD', 'N/A')}")
                    st.write(f"Card: {row.get('card4', 'N/A')} / {row.get('card6', 'N/A')}")
                    st.write(f"Email: {row.get('P_emaildomain', 'N/A')}")
                    st.write(f"Device: {row.get('DeviceType', 'N/A')} — {row.get('DeviceInfo', 'N/A')}")
                    st.write(f"Region: {row.get('addr1', 'N/A')}")

                with col2:
                    st.markdown("**Why flagged (reason codes)**")
                    # build feature row for SHAP
                    X = pd.DataFrame([{c: row.get(c, None) for c in feature_cols}])[feature_cols]
                    for c in X.columns:
                        if X[c].dtype == object:
                            X[c] = X[c].astype("category")
                    result = explainer.explain_one(X, top_k=3)
                    for i, rc in enumerate(result["reason_codes"], 1):
                        st.write(f"{i}. {rc['explanation']}")
                        st.caption(f"   SHAP impact: +{rc['shap_impact']}")

                # feedback buttons
                st.markdown("**Analyst decision**")
                fcol1, fcol2, fcol3 = st.columns([1, 1, 3])
                with fcol1:
                    if st.button("✅ Confirm Fraud", key=f"fraud_{tid}"):
                        feedback[str(tid)] = "fraud"
                        save_feedback(feedback)
                        st.rerun()
                with fcol2:
                    if st.button("❌ False Alarm", key=f"legit_{tid}"):
                        feedback[str(tid)] = "legit"
                        save_feedback(feedback)
                        st.rerun()

                # show ground truth (for demo only — real system wouldn't have this)
                if TARGET_COL in row:
                    actual = "FRAUD" if row[TARGET_COL] == 1 else "LEGIT"
                    st.caption(f"[Demo only] Ground truth: {actual}")

with tab2:
    st.subheader("Queue analytics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Alerts in queue", len(alerts))
    if TARGET_COL in alerts.columns:
        actual_fraud = int(alerts[TARGET_COL].sum())
        col2.metric("Actual fraud", actual_fraud)
        col3.metric("Queue precision", f"{alerts[TARGET_COL].mean()*100:.1f}%")

    # score distribution
    st.markdown("**Score distribution of alerts**")
    score_bins = pd.cut(alerts["fraud_score"], bins=20)
    score_counts = score_bins.value_counts(sort=False)
    score_df = pd.DataFrame({
        "score_range": [f"{iv.left:.2f}–{iv.right:.2f}" for iv in score_counts.index],
        "count":       score_counts.values,
    }).set_index("score_range")
    st.bar_chart(score_df)

    # amount distribution
    st.markdown("**Amount distribution of alerts (capped at $1000)**")
    amt_bins = pd.cut(alerts["TransactionAmt"].clip(upper=1000), bins=20)
    amt_counts = amt_bins.value_counts(sort=False)
    amt_df = pd.DataFrame({
        "amount_range": [f"${iv.left:.0f}–${iv.right:.0f}" for iv in amt_counts.index],
        "count":        amt_counts.values,
    }).set_index("amount_range")
    st.bar_chart(amt_df)