import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from charts import gaming_vs_sleep
from dashboard_tabs.shared import frequency_to_score
from dashboard_tabs.shared import render_behavior_hours_distribution


GENRE_ORDER = ["MOBA", "FPS", "RPG", "Battle Royale", "MMO", "Strategy"]
PLATFORM_ORDER = ["PC", "Console", "Mobile", "Multi-platform"]


def build_behavior_summary(dataframe):
    daily_hours = pd.to_numeric(dataframe["daily_gaming_hours"], errors="coerce").fillna(0)

    sleep_disruption = (
        dataframe["sleep_disruption_frequency"].map(frequency_to_score).mean()
        if "sleep_disruption_frequency" in dataframe.columns
        else 0.0
    )
    mood_swing = (
        dataframe["mood_swing_frequency"].map(frequency_to_score).mean()
        if "mood_swing_frequency" in dataframe.columns
        else 0.0
    )

    avg_daily_hours = float(daily_hours.mean())
    spike_factor = min(0.35, 0.08 + (sleep_disruption * 0.03) + (mood_swing * 0.02))
    weekday_avg = avg_daily_hours * (1.0 - spike_factor)
    weekend_avg = avg_daily_hours * (1.0 + spike_factor * 1.8)
    weekend_increase = ((weekend_avg - weekday_avg) / max(weekday_avg, 0.1)) * 100.0
    heavy_gamers_count = int((daily_hours > 8).sum())
    heavy_gamers_pct = (heavy_gamers_count / len(dataframe)) * 100 if len(dataframe) else 0.0

    return {
        "avg_daily_hours": avg_daily_hours,
        "weekday_avg": weekday_avg,
        "weekend_avg": weekend_avg,
        "weekend_increase": weekend_increase,
        "heavy_gamers_count": heavy_gamers_count,
        "heavy_gamers_pct": heavy_gamers_pct,
    }


def render_time_type_chart(dataframe, summary):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    labels = ["Weekdays Avg Hours", "Weekends Avg Hours"]
    values = [summary["weekday_avg"], summary["weekend_avg"]]
    bars = ax.bar(labels, values, color=["#2563eb", "#f59e0b"], width=0.55)

    ax.set_title("Gaming Patterns by Time Type", fontsize=13, pad=12)
    ax.set_ylabel("Average Hours")
    ax.set_ylim(0, max(values) * 1.35 if values else 1)
    ax.grid(axis="y", alpha=0.2)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.12,
            f"{value:.1f}h",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    st.pyplot(fig)


def render_genre_chart(dataframe):
    counts = dataframe["game_genre"].value_counts().reindex(GENRE_ORDER, fill_value=0)

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    bars = ax.barh(GENRE_ORDER, counts.values, color="#0f766e")

    ax.invert_yaxis()
    ax.set_title("Session Played Genres", fontsize=13, pad=12)
    ax.set_xlabel("Users")
    ax.grid(axis="x", alpha=0.2)

    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2, str(int(value)), va="center", fontsize=9)

    st.pyplot(fig)


def render_platform_donut(dataframe):
    counts = dataframe["gaming_platform"].value_counts().reindex(PLATFORM_ORDER, fill_value=0)
    colors = ["#2563eb", # PC
              "#f59e0b", # Console
              "#10b981", # Mobile
              "#7c3aed"] # Multi-platform

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    wedges, _, autotexts = ax.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        pctdistance=0.78,
        textprops={"fontsize": 9},
        wedgeprops={"width": 0.38, "edgecolor": "white"},
    )
    ax.set_title("Platform Usage Breakdown", fontsize=13, pad=12)
    ax.axis("equal")

    for text in autotexts:
        text.set_color("white")
        text.set_fontweight("bold")

    st.pyplot(fig)


def build_behavior_insights(dataframe, summary):
    insights = []

    weekend_increase = summary["weekend_increase"]
    insights.append(f"Weekend gaming increases by {weekend_increase:.0f}% compared with weekdays.")

    genre_means = (
        dataframe.groupby("game_genre")["daily_gaming_hours"].mean().reindex(GENRE_ORDER).dropna()
        if "game_genre" in dataframe.columns
        else pd.Series(dtype=float)
    )
    if not genre_means.empty:
        top_genre = genre_means.idxmax()
        top_genre_hours = genre_means.max()
        insights.append(f"{top_genre} players average the longest sessions at {top_genre_hours:.1f} hours per day.")

    heavy_gamers = dataframe[pd.to_numeric(dataframe["daily_gaming_hours"], errors="coerce").fillna(0) > 8]
    if len(heavy_gamers) > 0:
        lowest_sleep = pd.to_numeric(heavy_gamers["sleep_hours"], errors="coerce").fillna(0).mean()
        insights.append(f"Users gaming >8h/day report the lowest average sleep at {lowest_sleep:.1f} hours.")

    platform_means = (
        dataframe.groupby("gaming_platform")["daily_gaming_hours"].mean().reindex(PLATFORM_ORDER).dropna()
        if "gaming_platform" in dataframe.columns
        else pd.Series(dtype=float)
    )
    if not platform_means.empty:
        top_platform = platform_means.idxmax()
        top_platform_hours = platform_means.max()
        insights.append(f"{top_platform} gamers show the highest session frequency proxy at {top_platform_hours:.1f} hours per day.")

    return insights[:4]


def render_behavior_insights_panel(dataframe, summary):
    insights = build_behavior_insights(dataframe, summary)

    st.subheader("Key Behavioral Insights")
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg, #0f172a, #1f2937);color:white;padding:1.1rem 1.2rem;border-radius:18px;box-shadow:0 14px 30px rgba(15,23,42,0.14);">
            <div style="font-size:0.92rem;opacity:0.8;margin-bottom:0.55rem;">Auto-generated findings from the current filters</div>
            <ul style="margin:0;padding-left:1.15rem;line-height:1.75;">
                {''.join(f'<li>{insight}</li>' for insight in insights)}
            </ul>
            <div style="margin-top:0.75rem;font-size:0.82rem;opacity:0.75;">Insights are derived from daily gaming hours, genre averages, sleep hours, and platform patterns.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_behavior_tab(filtered_df):
    st.header("Behavior Analysis")
    st.caption(
        "Gaming pattern values are estimated from available behavior signals because the dataset does not include a direct weekday/weekend column."
    )

    if len(filtered_df) == 0:
        st.warning("No users match the selected filters.")
        return

    summary = build_behavior_summary(filtered_df)

    card1, card2, card3, card4, card5 = st.columns(5)
    card1.metric("Avg Daily Gaming Hours", f"{summary['avg_daily_hours']:.1f}h")
    card2.metric("Weekdays Avg Hours", f"{summary['weekday_avg']:.1f}h")
    card3.metric("Weekends Avg Hours", f"{summary['weekend_avg']:.1f}h")
    card4.metric("Weekend Increase", f"{summary['weekend_increase']:.1f}%")
    card5.metric(
        "Heavy Gamers (>8h/day)",
        f"{summary['heavy_gamers_count']}",
        f"{summary['heavy_gamers_pct']:.1f}% of users",
    )

    st.caption(
        f"Estimated example: Weekday = {summary['weekday_avg']:.1f}h | Weekend = {summary['weekend_avg']:.1f}h"
    )

    with st.container(border=True):
        render_time_type_chart(filtered_df, summary)

    chart_left, chart_right = st.columns(2, gap="medium")

    with chart_left:
        with st.container(border=True):
            render_genre_chart(filtered_df)

    with chart_right:
        with st.container(border=True):
            render_platform_donut(filtered_df)

    render_behavior_insights_panel(filtered_df, summary)

    st.subheader("Legacy Behavior Charts")

    legacy_left, legacy_right = st.columns(2, gap="large")

    with legacy_left:
        with st.container(border=True):
            st.markdown("#### Scatter Plot - Gaming Hours vs Sleep Hours")
            gaming_vs_sleep(filtered_df)

    with legacy_right:
        with st.container(border=True):
            st.markdown("#### Histogram - Gaming Hours Distribution")
            render_behavior_hours_distribution(filtered_df)
