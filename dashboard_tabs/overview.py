import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from charts import addiction_distribution
from dashboard_tabs.shared import (
    build_overview_insight_prompt,
    generate_overview_insights,
    risk_factor_impact_scores,
    train_overview_model,
)


def render_risk_factor_chart(dataframe):
    risk_scores = risk_factor_impact_scores(dataframe)
    labels = [item[0] for item in risk_scores]
    values = [item[1] for item in risk_scores]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.barh(labels, values, color=["#1d4ed8", "#0f766e", "#f59e0b", "#ef4444", "#7c3aed"])
    ax.invert_yaxis()
    ax.set_title("Top Risk Factors (Impact Score)", fontsize=13, pad=12)
    ax.set_xlabel("Impact Score")
    ax.set_xlim(0, max(values) + 0.8 if values else 1)
    ax.grid(axis="x", alpha=0.2)

    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=9)

    st.pyplot(fig)


def render_overview_tab(filtered_df):
    st.header("Dataset Overview")
    st.caption("This tab gives a quick summary of the dataset and the main risk patterns.")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Users", len(filtered_df))
    col2.metric("Avg Gaming Hours", round(filtered_df["daily_gaming_hours"].mean(), 2))
    col3.metric("Avg Sleep Hours", round(filtered_df["sleep_hours"].mean(), 2))
    col4.metric(
        "High Risk %",
        round((filtered_df["gaming_addiction_risk_level"] == "High").mean() * 100, 2),
    )
    col5.metric(
        "Severe Risk %",
        round((filtered_df["gaming_addiction_risk_level"] == "Severe").mean() * 100, 2),
    )

    chart_left, chart_right = st.columns(2, gap="large")

    with chart_left:
        with st.container(border=True):
            st.markdown("#### Top Risk Factors (Impact Score)")
            render_risk_factor_chart(filtered_df)

    with chart_right:
        with st.container(border=True):
            st.markdown("#### Risk Level Distribution")
            addiction_distribution(filtered_df)

    st.subheader("Key Insights")
    insight_prompt = build_overview_insight_prompt(filtered_df)
    with st.spinner("Generating overview insights..."):
        try:
            insights_text = generate_overview_insights(insight_prompt)
        except Exception:
            insights_text = (
                "- Users with longer daily gaming time tend to fall into higher risk groups.\n"
                "- Short sleep duration is associated with elevated addiction risk.\n"
                "- Withdrawal symptoms and isolation are strong warning signals.\n"
                "- Severe-risk users should be prioritized for support and monitoring."
            )

    st.info("OpenAI-generated summary")
    st.write(insights_text)

    st.subheader("Model Performance")
    overview_metrics = train_overview_model()
    performance_df = pd.DataFrame(
        [
            ["Accuracy", overview_metrics["accuracy"]],
            ["Precision", overview_metrics["precision"]],
            ["Recall", overview_metrics["recall"]],
            ["F1 Score", overview_metrics["f1"]],
        ],
        columns=["Metric", "Score"],
    )
    st.table(performance_df.style.format({"Score": "{:.3f}"}))
