import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from charts import build_clustering_summary
from charts import clustering_result


def render_clustering_kpi_cards(summary):
    card1, card2, card3, card4, card5 = st.columns(5)

    card1.metric("Total Clusters", summary["best_k"])
    card2.metric("Largest Cluster", f"{summary['largest_cluster_pct']:.0f}%")
    card3.metric("Highest Risk Cluster", summary["highest_risk_cluster_label"])
    card4.metric("Silhouette Score", f"{summary['silhouette_score']:.2f}")
    card5.metric("Avg Within Distance", summary["avg_within_distance_label"])


def render_cluster_profile_table(summary):
    st.subheader("Cluster Profile Table")
    st.caption("This table compares the main values for each cluster.")
    st.dataframe(summary["cluster_profile_table"], use_container_width=True)


def build_key_cluster_insights(summary):
    table = summary["cluster_profile_table"]

    gaming_hours = table.loc["Gaming Hours"].astype(float)
    sleep_hours = table.loc["Sleep Hours"].astype(float)
    risk_levels = table.loc["Risk Level"]

    highest_gaming_cluster = gaming_hours.idxmax()
    lowest_sleep_cluster = sleep_hours.idxmin()
    healthiest_cluster = sleep_hours.idxmax()

    severe_clusters = [cluster_name for cluster_name, value in risk_levels.items() if str(value).strip().lower() == "severe"]
    moderate_clusters = [cluster_name for cluster_name, value in risk_levels.items() if str(value).strip().lower() == "moderate"]

    insights = [
        f"{highest_gaming_cluster} users average {gaming_hours[highest_gaming_cluster]:.1f}+ gaming hours/day.",
        f"{lowest_sleep_cluster} users show the most sleep loss at {sleep_hours[lowest_sleep_cluster]:.1f} hours/night.",
        f"{healthiest_cluster} has the healthiest balance and highest average sleep at {sleep_hours[healthiest_cluster]:.1f} hours/night.",
    ]

    if moderate_clusters:
        insights.append(f"{', '.join(moderate_clusters)} users show moderate risk with clear sleep decline.")
    else:
        insights.append("One or more clusters show moderate risk with clear sleep decline.")

    if severe_clusters:
        insights.append(f"Severe users are concentrated in {' and '.join(severe_clusters)}.")
    else:
        insights.append("Severe users are concentrated in a single cluster segment.")

    return insights[:4]


def render_key_cluster_insights_panel(summary):
    insights = build_key_cluster_insights(summary)

    st.subheader("Key Cluster Insights")
    st.info("These insights are made from the cluster comparison table.")
    for insight in insights:
        st.write(f"- {insight}")

    st.caption("The goal is to make the cluster results easier to understand at a glance.")


def render_cluster_size_distribution(summary):
    cluster_counts = pd.Series(summary["cluster_labels"]).value_counts().sort_index()
    cluster_names = [summary["cluster_name_map"][cluster_id] for cluster_id in cluster_counts.index]
    colors = ["#2563eb", "#0f766e", "#f59e0b", "#7c3aed", "#ef4444"]

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    bars = ax.bar(cluster_names, cluster_counts.values, color=[colors[index % len(colors)] for index in range(len(cluster_names))])
    ax.set_title("Cluster Size Distribution", fontsize=13, pad=12)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Users")
    ax.grid(axis="y", alpha=0.2)

    for bar, value in zip(bars, cluster_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(int(value)), ha="center", va="bottom", fontsize=9)

    st.subheader("Cluster Size Distribution")
    st.caption("This bar chart shows how many users are in each cluster.")
    st.pyplot(fig)


def render_cluster_risk_overlay(summary):
    overlay = summary["cluster_risk_overlay_table"]
    risk_order = ["High", "Severe", "Moderate", "Low"]
    risk_colors = {
        "High": "#f59e0b",
        "Severe": "#ef4444",
        "Moderate": "#2563eb",
        "Low": "#10b981",
    }

    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    left = [0] * len(overlay.index)

    for risk_label in risk_order:
        if risk_label not in overlay.columns:
            continue

        values = overlay[risk_label].values
        bars = ax.barh(
            overlay.index.map(lambda cluster_id: summary["cluster_name_map"][cluster_id]),
            values,
            left=left,
            color=risk_colors[risk_label],
            label=risk_label,
        )

        for bar, value in zip(bars, values):
            if value >= 12:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )

        left = [current + value for current, value in zip(left, values)]

    ax.set_title("Cluster Risk Overlay", fontsize=13, pad=12)
    ax.set_xlabel("Users (%)")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.2)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))

    st.subheader("Cluster Risk Overlay")
    st.caption("This chart shows the risk mix inside each cluster.")
    st.pyplot(fig)


def render_clustering_tab(filtered_df):
    st.header("K-Means Clustering")
    st.caption("This tab groups similar users into clusters and compares their patterns.")

    summary = build_clustering_summary(filtered_df)
    if summary is None:
        st.warning("Not enough records to compute clustering metrics.")
        return

    render_clustering_kpi_cards(summary)

    chart_left, chart_right = st.columns(2, gap="large")
    with chart_left:
        with st.container(border=True):
            render_cluster_size_distribution(summary)
    with chart_right:
        with st.container(border=True):
            render_cluster_risk_overlay(summary)

    lower_left, lower_right = st.columns(2, gap="large")
    with lower_left:
        with st.container(border=True):
            render_cluster_profile_table(summary)
    with lower_right:
        with st.container(border=True):
            render_key_cluster_insights_panel(summary)

    with st.container(border=True):
        st.subheader("Cluster Map")
        clustering_result(filtered_df, summary)
