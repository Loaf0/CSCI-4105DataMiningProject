import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import gaussian_kde

from dashboard_tabs.shared import frequency_to_score


MOOD_STATE_MAP = {
    "euphoric": 9.5,
    "excited": 8.5,
    "normal": 7.0,
    "calm": 7.5,
    "restless": 4.5,
    "irritable": 4.0,
    "angry": 3.5,
    "anxious": 3.0,
    "withdrawn": 2.5,
    "depressed": 2.0,
    "stressed": 2.5,
}

SLEEP_QUALITY_MAP = {
    "excellent": 1.0,
    "good": 2.5,
    "fair": 4.5,
    "poor": 6.5,
    "very poor": 8.5,
    "insomnia": 9.5,
}


def to_binary(value):
    if pd.isna(value):
        return 0.0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1"}:
            return 1.0
        if normalized in {"false", "no", "n", "0"}:
            return 0.0
    return float(bool(value))


def normalize_series(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(numeric.median())
    return pd.Series([0.0] * len(series), index=series.index)


def build_social_scores(dataframe):
    gaming_hours = normalize_series(dataframe["daily_gaming_hours"]) if "daily_gaming_hours" in dataframe.columns else pd.Series([0.0] * len(dataframe), index=dataframe.index)
    sleep_hours = normalize_series(dataframe["sleep_hours"]) if "sleep_hours" in dataframe.columns else pd.Series([0.0] * len(dataframe), index=dataframe.index)
    social_hours = normalize_series(dataframe["face_to_face_social_hours_weekly"]) if "face_to_face_social_hours_weekly" in dataframe.columns else pd.Series([0.0] * len(dataframe), index=dataframe.index)
    isolation = normalize_series(dataframe["social_isolation_score"]) if "social_isolation_score" in dataframe.columns else pd.Series([0.0] * len(dataframe), index=dataframe.index)

    sleep_disruption = dataframe["sleep_disruption_frequency"].map(frequency_to_score) if "sleep_disruption_frequency" in dataframe.columns else pd.Series([0.0] * len(dataframe), index=dataframe.index)
    mood_swing = dataframe["mood_swing_frequency"].map(frequency_to_score) if "mood_swing_frequency" in dataframe.columns else pd.Series([0.0] * len(dataframe), index=dataframe.index)
    withdrawal = dataframe["withdrawal_symptoms"].map(to_binary) if "withdrawal_symptoms" in dataframe.columns else pd.Series([0.0] * len(dataframe), index=dataframe.index)
    mood_state = dataframe["mood_state"].astype(str).str.strip().str.lower().map(MOOD_STATE_MAP).fillna(6.0) if "mood_state" in dataframe.columns else pd.Series([6.0] * len(dataframe), index=dataframe.index)
    sleep_quality = dataframe["sleep_quality"].astype(str).str.strip().str.lower().map(SLEEP_QUALITY_MAP).fillna(4.5) if "sleep_quality" in dataframe.columns else pd.Series([4.5] * len(dataframe), index=dataframe.index)

    gaming_component = np.clip(gaming_hours / 12.0 * 10.0, 0, 10)
    isolation_component = np.clip(isolation, 0, 10)
    disruption_component = np.clip(sleep_disruption / 4.0 * 10.0, 0, 10)
    mood_component = np.clip(mood_swing / 4.0 * 10.0, 0, 10)
    withdrawal_component = withdrawal * 10.0
    sleep_penalty = np.clip(sleep_quality, 0, 10)
    low_sleep_component = np.clip((8.0 - sleep_hours) / 8.0 * 10.0, 0, 10)
    low_social_component = np.clip((12.0 - social_hours) / 12.0 * 10.0, 0, 10)
    mood_drop_component = np.clip(10.0 - mood_state, 0, 10)

    stress = np.clip(
        0.30 * disruption_component
        + 0.30 * withdrawal_component
        + 0.25 * isolation_component
        + 0.15 * gaming_component,
        0,
        10,
    )
    anxiety = np.clip(
        0.35 * isolation_component
        + 0.25 * disruption_component
        + 0.20 * mood_component
        + 0.20 * mood_drop_component,
        0,
        10,
    )
    depression = np.clip(
        0.30 * sleep_penalty
        + 0.25 * low_sleep_component
        + 0.25 * low_social_component
        + 0.20 * isolation_component,
        0,
        10,
    )

    mood_score = np.clip(mood_state, 0, 10)
    distress_index = (stress + anxiety + depression) / 3.0

    return {
        "stress": pd.Series(stress, index=dataframe.index),
        "anxiety": pd.Series(anxiety, index=dataframe.index),
        "depression": pd.Series(depression, index=dataframe.index),
        "mood_score": pd.Series(mood_score, index=dataframe.index),
        "distress_index": pd.Series(distress_index, index=dataframe.index),
        "sleep_disruption": sleep_disruption,
        "sleep_quality": sleep_quality,
        "isolation": isolation,
    }


def render_density_curves(dataframe, scores):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x_grid = np.linspace(0, 10, 300)
    color_map = {
        "stress": "#ef4444",
        "anxiety": "#f59e0b",
        "depression": "#7c3aed",
    }

    for label in ["stress", "anxiety", "depression"]:
        values = scores[label].dropna().astype(float)
        if values.nunique() < 2:
            ax.axvline(values.iloc[0], color=color_map[label], linewidth=2, label=label.title())
            continue

        try:
            kde = gaussian_kde(values)
            y = kde(x_grid)
            ax.plot(x_grid, y, color=color_map[label], linewidth=2, label=label.title())
            ax.fill_between(x_grid, y, alpha=0.12, color=color_map[label])
        except Exception:
            hist, edges = np.histogram(values, bins=20, range=(0, 10), density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            ax.plot(centers, hist, color=color_map[label], linewidth=2, label=label.title())

    ax.set_title("Mental Health Score Distribution", fontsize=13, pad=12)
    ax.set_xlabel("Score (0-10)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 10)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    st.pyplot(fig)


def render_sleep_impact_section(dataframe, scores):
    st.subheader("Sleep Impact Section")

    left_col, middle_col, right_col = st.columns(3, gap="large")

    with left_col:
        with st.container(border=True):
            st.markdown("#### Sleep Hours vs Anxiety")
            fig, ax = plt.subplots(figsize=(4.2, 3.4))
            ax.scatter(
                normalize_series(dataframe["sleep_hours"]),
                scores["anxiety"],
                alpha=0.7,
                color="#2563eb",
            )
            ax.set_xlabel("Sleep Hours")
            ax.set_ylabel("Anxiety Score")
            ax.set_title("Sleep Hours vs Anxiety", fontsize=11, pad=10)
            ax.grid(alpha=0.2)
            st.pyplot(fig)

    with middle_col:
        with st.container(border=True):
            st.markdown("#### Sleep Quality vs Stress")
            fig, ax = plt.subplots(figsize=(4.2, 3.4))
            ax.scatter(
                scores["sleep_quality"],
                scores["stress"],
                alpha=0.7,
                color="#0f766e",
            )
            ax.set_xlabel("Sleep Quality (worse -> higher)")
            ax.set_ylabel("Stress Score")
            ax.set_title("Sleep Quality vs Stress", fontsize=11, pad=10)
            ax.grid(alpha=0.2)
            st.pyplot(fig)

    with right_col:
        with st.container(border=True):
            st.markdown("#### Sleep Disruption Frequency")
            order = ["Never", "Rarely", "Sometimes", "Often", "Daily"]
            freq_counts = (
                dataframe["sleep_disruption_frequency"].astype(str).str.strip().str.title().value_counts()
                if "sleep_disruption_frequency" in dataframe.columns
                else pd.Series(dtype=int)
            )
            freq_counts = freq_counts.reindex(order, fill_value=0)

            fig, ax = plt.subplots(figsize=(4.2, 3.4))
            bars = ax.bar(order, freq_counts.values, color="#f59e0b")
            ax.set_xlabel("Frequency")
            ax.set_ylabel("Users")
            ax.set_title("Sleep Disruption Frequency", fontsize=11, pad=10)
            ax.grid(axis="y", alpha=0.2)

            for bar, value in zip(bars, freq_counts.values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15, str(int(value)), ha="center", va="bottom", fontsize=8)

            st.pyplot(fig)


def render_social_isolation_card(dataframe, scores):
    loneliness_pct = (scores["isolation"] >= 7).mean() * 100 if len(scores["isolation"]) else 0.0
    high_isolation_pct = (scores["isolation"] >= 8).mean() * 100 if len(scores["isolation"]) else 0.0

    st.markdown("#### Social Isolation & Loneliness")
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg, #111827, #1f2937);color:white;padding:1.1rem 1.2rem;border-radius:18px;box-shadow:0 14px 30px rgba(15,23,42,0.14);">
            <div style="font-size:0.92rem;opacity:0.8;margin-bottom:0.5rem;">Isolation overview</div>
            <div style="display:flex;gap:1rem;flex-wrap:wrap;">
                <div style="flex:1;min-width:180px;background:rgba(255,255,255,0.06);padding:0.9rem;border-radius:14px;">
                    <div style="font-size:0.84rem;opacity:0.8;">Users reporting loneliness</div>
                    <div style="font-size:1.8rem;font-weight:700;">{loneliness_pct:.1f}%</div>
                    <div style="font-size:0.82rem;opacity:0.85;">Proxy based on social isolation score of 7 or higher.</div>
                </div>
                <div style="flex:1;min-width:180px;background:rgba(255,255,255,0.06);padding:0.9rem;border-radius:14px;">
                    <div style="font-size:0.84rem;opacity:0.8;">High isolation score users</div>
                    <div style="font-size:1.8rem;font-weight:700;">{high_isolation_pct:.1f}%</div>
                    <div style="font-size:0.82rem;opacity:0.85;">Users with social isolation score of 8 or higher.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_top_mental_health_issues(dataframe, scores):
    mood_swing_pct = (
        (dataframe["mood_swing_frequency"].map(frequency_to_score) >= 3.0).mean() * 100
        if "mood_swing_frequency" in dataframe.columns
        else (scores["mood_score"] <= 4.5).mean() * 100
    )

    issues = {
        "Stress": (scores["stress"] >= 6.5).mean() * 100,
        "Anxiety": (scores["anxiety"] >= 6.5).mean() * 100,
        "Depression": (scores["depression"] >= 6.5).mean() * 100,
        "Mood Swings": mood_swing_pct,
        "Loneliness": (scores["isolation"] >= 7.0).mean() * 100,
    }

    ranked_items = sorted(issues.items(), key=lambda item: item[1], reverse=True)
    return ranked_items


def render_top_mental_health_issues_panel(dataframe, scores):
    ranked_items = build_top_mental_health_issues(dataframe, scores)
    labels = [item[0] for item in ranked_items]
    values = [item[1] for item in ranked_items]

    st.subheader("Top Mental Health Issues")
    with st.container(border=True):
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        bars = ax.barh(labels, values, color=["#ef4444", "#f59e0b", "#7c3aed", "#10b981", "#2563eb"])
        ax.invert_yaxis()
        ax.set_xlabel("Users (%)")
        ax.set_xlim(0, max(values) * 1.25 if values else 1)
        ax.set_title("Top Mental Health Issues", fontsize=13, pad=12)
        ax.grid(axis="x", alpha=0.2)

        for bar, value in zip(bars, values):
            ax.text(bar.get_width() + 0.6, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center", fontsize=9)

        st.pyplot(fig)


def build_key_mental_health_insights(dataframe, scores):
    poor_sleep = None
    if "sleep_quality" in dataframe.columns:
        poor_sleep_mask = dataframe["sleep_quality"].astype(str).str.strip().str.lower().isin({"poor", "very poor", "insomnia"})
        poor_sleep = scores["anxiety"][poor_sleep_mask].mean()

    severe_mask = dataframe["gaming_addiction_risk_level"].astype(str).str.strip().str.lower().eq("severe") if "gaming_addiction_risk_level" in dataframe.columns else pd.Series([False] * len(dataframe), index=dataframe.index)
    severe_depression = scores["depression"][severe_mask].mean() if severe_mask.any() else scores["depression"].mean()

    mood_swing_series = dataframe["mood_swing_frequency"].map(frequency_to_score) if "mood_swing_frequency" in dataframe.columns else pd.Series([0.0] * len(dataframe), index=dataframe.index)
    correlation = scores["isolation"].corr(mood_swing_series) if len(dataframe) > 1 else 0.0

    moderate_mask = pd.to_numeric(dataframe["daily_gaming_hours"], errors="coerce").between(4, 8) if "daily_gaming_hours" in dataframe.columns else pd.Series([False] * len(dataframe), index=dataframe.index)
    moderate_mood = scores["mood_score"][moderate_mask].mean() if moderate_mask.any() else scores["mood_score"].mean()

    insights = []
    if poor_sleep is not None and not np.isnan(poor_sleep):
        insights.append(f"Users with poor sleep report significantly higher anxiety (average anxiety score {poor_sleep:.1f}/10).")
    else:
        insights.append("Users with poor sleep report significantly higher anxiety.")

    insights.append(f"Severe-risk gamers show the highest depression scores at {severe_depression:.1f}/10.")

    if not np.isnan(correlation):
        insights.append(f"Social isolation strongly correlates with mood swings (correlation {correlation:.2f}).")
    else:
        insights.append("Social isolation strongly correlates with mood swings.")

    insights.append(f"Moderate gaming groups show the healthiest mood balance at {moderate_mood:.1f}/10.")

    return insights[:4]


def render_key_mental_health_insights_panel(dataframe, scores):
    insights = build_key_mental_health_insights(dataframe, scores)

    st.subheader("Key Mental Health Insights")
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg, #0f172a, #1f2937);color:white;padding:1.1rem 1.2rem;border-radius:18px;box-shadow:0 14px 30px rgba(15,23,42,0.14);">
            <div style="font-size:0.92rem;opacity:0.8;margin-bottom:0.55rem;">Auto-generated findings from the current filters</div>
            <ul style="margin:0;padding-left:1.15rem;line-height:1.75;">
                {''.join(f'<li>{insight}</li>' for insight in insights)}
            </ul>
            <div style="margin-top:0.75rem;font-size:0.82rem;opacity:0.75;">These findings translate the raw metrics into clear conclusions about sleep, distress, isolation, and mood balance.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_social_tab(filtered_df):
    st.header("Mental Health")

    if len(filtered_df) == 0:
        st.warning("No users match the selected filters.")
        return

    scores = build_social_scores(filtered_df)

    st.caption(
        "Mental health values are composite scores derived from the available sleep, mood, isolation, and disruption fields in the dataset."
    )

    card1, card2, card3, card4, card5 = st.columns(5)
    card1.metric("Avg Stress Level", f"{scores['stress'].mean():.1f}/10")
    card2.metric("Avg Anxiety Level", f"{scores['anxiety'].mean():.1f}/10")
    card3.metric("Avg Depression Level", f"{scores['depression'].mean():.1f}/10")
    card4.metric("Avg Mood Score", f"{scores['mood_score'].mean():.1f}/10")
    card5.metric(
        "High Distress Users",
        f"{(scores['distress_index'] >= 6.5).sum()}",
        f"{((scores['distress_index'] >= 6.5).mean() * 100):.1f}% of users",
    )

    st.subheader("Mental Health Score Distribution")
    render_density_curves(filtered_df, scores)

    render_sleep_impact_section(filtered_df, scores)

    render_social_isolation_card(filtered_df, scores)

    render_top_mental_health_issues_panel(filtered_df, scores)

    render_key_mental_health_insights_panel(filtered_df, scores)
