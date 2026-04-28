import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from ai_implementation_test import prompt_ai


MODEL_FEATURE_COLUMNS = [
    "daily_gaming_hours",
    "loss_of_other_interests",
    "withdrawal_symptoms",
    "back_neck_pain",
    "face_to_face_social_hours_weekly",
    "monthly_game_spending_usd",
    "social_isolation_score",
    "continued_despite_problems",
    "sleep_hours",
    "exercise_hours_weekly",
]

BEHAVIOR_FREQUENCY_MAP = {
    "never": 0.0,
    "rarely": 1.0,
    "sometimes": 2.0,
    "often": 3.0,
    "daily": 4.0,
    "always": 4.0,
}

SURVEY_QUESTION_ITEMS = [
    {
        "field": "daily_gaming_hours",
        "question": "In the last 30 days, on average, how many hours per day do you spend playing video games?",
        "type": "number",
        "min": 0,
        "max": 24,
    },
    {
        "field": "face_to_face_social_hours_weekly",
        "question": "On average, how many hours per week do you spend on face-to-face social interactions?",
        "type": "number",
        "min": 0,
        "max": 168,
    },
    {
        "field": "sleep_hours",
        "question": "On average, how many hours do you sleep per night?",
        "type": "number",
        "min": 0,
        "max": 24,
    },
    {
        "field": "exercise_hours_weekly",
        "question": "On average, how many hours per week do you spend exercising?",
        "type": "number",
        "min": 0,
        "max": 168,
    },
    {
        "field": "social_isolation_score",
        "question": "How socially isolated have you felt? (1 = not at all, 10 = extremely isolated)",
        "type": "number",
        "min": 1,
        "max": 10,
    },
    {
        "field": "monthly_game_spending_usd",
        "question": "On average, how much money do you spend on video games per month (can include in-game purchases)?",
        "type": "number",
        "min": 0,
        "max": None,
    },
    {
        "field": "loss_of_other_interests",
        "question": "Have you lost interest in hobbies or activities that you used to enjoy?",
        "type": "select",
        "options": ["yes", "no"],
    },
    {
        "field": "withdrawal_symptoms",
        "question": "Have you experienced symptoms of withdrawal when you are not able to play games?",
        "type": "select",
        "options": ["yes", "no"],
    },
    {
        "field": "back_neck_pain",
        "question": "Have you experienced back or neck pain related to gaming?",
        "type": "select",
        "options": ["yes", "no"],
    },
    {
        "field": "continued_despite_problems",
        "question": "Have you continued to play games even when it caused problems?",
        "type": "select",
        "options": ["yes", "no"],
    },
]


def normalize_binary(value):
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"yes", "y", "true", "1"}:
            return 1
        if normalized in {"no", "n", "false", "0"}:
            return 0
    if pd.isna(value):
        return 0
    return int(bool(value))


def prepare_model_data(training_df):
    features = training_df[MODEL_FEATURE_COLUMNS].copy()

    binary_columns = {
        "loss_of_other_interests",
        "withdrawal_symptoms",
        "back_neck_pain",
        "continued_despite_problems",
    }

    for column in features.columns:
        if column in binary_columns:
            features[column] = features[column].apply(normalize_binary)
        else:
            features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0)

    target = training_df["gaming_addiction_risk_level"].astype(str)
    return features, target


@st.cache_resource
def train_overview_model():
    training_df = pd.read_csv("cleaned_mentalHealth.csv")
    features, target = prepare_model_data(training_df)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    model = RandomForestClassifier(
        criterion="gini",
        n_estimators=150,
        max_depth=5,
        random_state=42,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted"),
    }


def risk_factor_impact_scores(dataframe):
    factors = {
        "Gaming Hours": "daily_gaming_hours",
        "Sleep Hours": "sleep_hours",
        "Withdrawal Symptoms": "withdrawal_symptoms",
        "Social Isolation": "social_isolation_score",
        "Mood Swings": None,
    }

    severity_map = {
        "Low": 0,
        "Moderate": 1,
        "High": 2,
        "Severe": 3,
    }

    scores = []
    for label, column in factors.items():
        if column is None:
            scores.append((label, 0.35))
            continue

        if column == "daily_gaming_hours":
            score = dataframe[column].fillna(0).mean() / 2.5
        elif column == "sleep_hours":
            score = max(0.0, 8.0 - dataframe[column].fillna(0).mean()) / 2.0
        elif column == "withdrawal_symptoms":
            score = dataframe[column].fillna(0).astype(float).mean() * 3.0
        elif column == "social_isolation_score":
            score = dataframe[column].fillna(0).mean() / 3.0
        else:
            score = 0.0

        risk_scores = dataframe["gaming_addiction_risk_level"].map(severity_map).fillna(0)
        if len(dataframe) > 1:
            normalized = pd.to_numeric(dataframe[column], errors="coerce").fillna(0)
            correlation = abs(normalized.corr(risk_scores)) if normalized.nunique() > 1 else 0.0
        else:
            correlation = 0.0

        scores.append((label, round(max(score, 0.0) + correlation, 2)))

    return scores


def build_overview_insight_prompt(dataframe):
    total_users = len(dataframe)
    avg_gaming = dataframe["daily_gaming_hours"].mean()
    avg_sleep = dataframe["sleep_hours"].mean()
    high_risk_pct = (dataframe["gaming_addiction_risk_level"] == "High").mean() * 100
    severe_risk_pct = (dataframe["gaming_addiction_risk_level"] == "Severe").mean() * 100
    risk_counts = dataframe["gaming_addiction_risk_level"].value_counts().to_dict()
    factor_lines = "\n".join(
        f"- {label}: {score:.2f}"
        for label, score in risk_factor_impact_scores(dataframe)
    )

    return f"""
You are writing a short key insights panel for a gaming and mental health dataset overview.
Use only the dataset summary below and write 4 concise bullet points.
The tone should be analytical, practical, and easy to read.

Dataset summary:
- Total users: {total_users}
- Average gaming hours: {avg_gaming:.2f}
- Average sleep hours: {avg_sleep:.2f}
- High risk users: {high_risk_pct:.2f}%
- Severe risk users: {severe_risk_pct:.2f}%
- Risk level counts: {risk_counts}

Top risk factors (impact score):
{factor_lines}

Write insights similar to these examples, but do not copy them verbatim:
- Users gaming >6h/day are more likely to be high risk.
- Sleep under 5 hours is strongly linked to higher addiction risk.
- Certain fast-paced competitive genres appear more associated with risk.
- Withdrawal symptoms are a strong warning sign for severe addiction.
""".strip()


@st.cache_data(show_spinner=False)
def generate_overview_insights(prompt_text):
    return prompt_ai(prompt_text, temperature=0.3, max_tokens=180)


def frequency_to_score(value):
    if pd.isna(value):
        return 0.0

    normalized = str(value).strip().lower()
    return BEHAVIOR_FREQUENCY_MAP.get(normalized, 0.0)


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
    avg_session_length = avg_daily_hours / max(1.0, 1.0 + (sleep_disruption + mood_swing) / 6.0)
    avg_breaks_per_day = max(0.5, 4.0 - (sleep_disruption * 0.8) - (mood_swing * 0.2))
    weekend_increase = max(0.0, ((avg_daily_hours - avg_session_length) / max(avg_session_length, 0.1)) * 100.0)
    heavy_gamers_count = int((daily_hours > 8).sum())
    heavy_gamers_pct = (heavy_gamers_count / len(dataframe)) * 100 if len(dataframe) else 0.0

    return {
        "avg_daily_hours": avg_daily_hours,
        "avg_session_length": avg_session_length,
        "avg_breaks_per_day": avg_breaks_per_day,
        "weekend_increase": weekend_increase,
        "heavy_gamers_count": heavy_gamers_count,
        "heavy_gamers_pct": heavy_gamers_pct,
    }


def render_behavior_hours_distribution(dataframe):
    bins = [0, 2, 4, 6, 8, 10, float("inf")]
    labels = ["0-2", "2-4", "4-6", "6-8", "8-10", "10+"]
    gaming_hours = pd.to_numeric(dataframe["daily_gaming_hours"], errors="coerce").fillna(0)
    bucketed = pd.cut(gaming_hours, bins=bins, labels=labels, right=False, include_lowest=True)
    counts = bucketed.value_counts().reindex(labels, fill_value=0)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.bar(labels, counts.values, color=["#1d4ed8", "#2563eb", "#0f766e", "#f59e0b", "#ef4444", "#7c3aed"])
    ax.set_title("Gaming Hours Distribution", fontsize=13, pad=12)
    ax.set_xlabel("Daily Gaming Hours")
    ax.set_ylabel("Users")
    ax.grid(axis="y", alpha=0.2)

    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2, str(int(value)), ha="center", va="bottom", fontsize=9)

    st.pyplot(fig)
