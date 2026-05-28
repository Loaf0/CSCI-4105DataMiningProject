import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from dashboard_tabs.shared import prepare_model_data


GLOBAL_IMPORTANCE_FEATURES = [
    "daily_gaming_hours",
    "sleep_hours",
    "withdrawal_symptoms",
    "continued_despite_problems",
    "social_isolation_score",
]


FEATURE_BINARY_COLUMNS = {
    "loss_of_other_interests",
    "withdrawal_symptoms",
    "back_neck_pain",
    "continued_despite_problems",
}


USER_EXPLANATION_FEATURES = {
    "daily_gaming_hours": {
        "label": "Gaming hours",
        "direction": "higher",
        "risk_label": "Gaming {value:.1f} hours/day",
    },
    "sleep_hours": {
        "label": "Sleep hours",
        "direction": "lower",
        "risk_label": "Sleep only {value:.1f} hours",
    },
    "withdrawal_symptoms": {
        "label": "Withdrawal symptoms",
        "direction": "higher",
        "risk_label": "High withdrawal symptoms",
        "binary": True,
    },
    "continued_despite_problems": {
        "label": "Continued despite problems",
        "direction": "higher",
        "risk_label": "Continued despite problems",
        "binary": True,
    },
    "social_isolation_score": {
        "label": "Social isolation",
        "direction": "higher",
        "risk_label": "Strong social isolation",
    },
    "face_to_face_social_hours_weekly": {
        "label": "Face-to-face social time",
        "direction": "lower",
        "risk_label": "Low face-to-face social time",
    },
    "monthly_game_spending_usd": {
        "label": "Game spending",
        "direction": "higher",
        "risk_label": "High monthly game spending",
    },
    "exercise_hours_weekly": {
        "label": "Exercise hours",
        "direction": "lower",
        "risk_label": "Low exercise time",
    },
    "loss_of_other_interests": {
        "label": "Loss of other interests",
        "direction": "higher",
        "risk_label": "Loss of other interests",
        "binary": True,
    },
    "back_neck_pain": {
        "label": "Back or neck pain",
        "direction": "higher",
        "risk_label": "Back or neck pain",
        "binary": True,
    },
}


@st.cache_resource
def train_global_model():
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
    return model, features.columns


def build_prediction_features(row):
    feature_columns = [
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

    features = row[feature_columns].copy()

    for column in features.index:
        if column in FEATURE_BINARY_COLUMNS:
            value = features.loc[column]
            if isinstance(value, str):
                normalized = value.strip().lower()
                features.loc[column] = 1 if normalized in {"yes", "y", "true", "1"} else 0
            else:
                features.loc[column] = 1 if bool(value) else 0
        else:
            features.loc[column] = pd.to_numeric(features.loc[column], errors="coerce")

    return pd.DataFrame([features.values], columns=features.index)


def predict_single_user(model, record_row):
    prediction_frame = build_prediction_features(record_row)
    prediction = model.predict(prediction_frame)[0]
    confidence = float(max(model.predict_proba(prediction_frame)[0]))
    return prediction, confidence


def _format_explanation_value(feature_name, raw_value):
    feature_meta = USER_EXPLANATION_FEATURES[feature_name]
    if feature_meta.get("binary"):
        return "Yes" if int(raw_value) == 1 else "No"
    return f"{float(raw_value):.1f}"


def _simple_user_summary(prediction):
    if prediction == "Severe":
        return "This user has a very high risk level and may need support soon."
    if prediction == "High":
        return "This user has a high risk level and should be watched closely."
    if prediction == "Moderate":
        return "This user has some warning signs but is not in the highest group."
    return "This user is currently in the lower risk group."


def build_user_explanation(record_row, model, feature_frame):
    training_df = pd.read_csv("cleaned_mentalHealth.csv")
    training_features, _ = prepare_model_data(training_df)
    feature_importances = pd.Series(model.feature_importances_, index=feature_frame.columns)
    user_values = feature_frame.iloc[0]
    medians = training_features.median(numeric_only=True)

    contributions = []
    for feature_name, feature_meta in USER_EXPLANATION_FEATURES.items():
        importance = float(feature_importances.get(feature_name, 0.0))
        if importance <= 0:
            continue

        user_value = float(user_values.get(feature_name, 0.0))
        median_value = float(medians.get(feature_name, 0.0))

        if feature_meta.get("binary"):
            signal_strength = 1.0 if user_value >= 1 else 0.0
        elif feature_meta["direction"] == "higher":
            signal_strength = max(user_value - median_value, 0.0)
        else:
            signal_strength = max(median_value - user_value, 0.0)

        score = importance * (1.0 + signal_strength)
        if score > 0:
            contributions.append(
                {
                    "feature": feature_name,
                    "label": feature_meta["label"],
                    "risk_text": feature_meta["risk_label"].format(value=user_value),
                    "score": score,
                    "value_text": _format_explanation_value(feature_name, user_value),
                }
            )

    contributions.sort(key=lambda item: item["score"], reverse=True)
    return contributions[:4]


def render_explainable_ai_panel(record_row, model, prediction_frame):
    contributions = build_user_explanation(record_row, model, prediction_frame)

    st.subheader("Explainable AI Panel")
    st.info("These are the main things that affected the prediction.")
    for item in contributions:
        st.write(f"- {item['risk_text']} ({item['label']}: {item['value_text']})")

    st.caption("The list compares model feature importance with the user's values.")


def render_global_model_importance(dataframe):
    model, feature_names = train_global_model()
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.reindex(GLOBAL_IMPORTANCE_FEATURES).fillna(0)

    labels = [
        "Daily Gaming Hours",
        "Sleep Hours",
        "Withdrawal Symptoms",
        "Continued Despite Problems",
        "Social Isolation",
    ]
    values = [float(importances.get(feature, 0.0)) for feature in GLOBAL_IMPORTANCE_FEATURES]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.barh(labels, values, color=["#2563eb", "#0f766e", "#f59e0b", "#ef4444", "#7c3aed"])
    ax.invert_yaxis()
    ax.set_title("Global Model Importance", fontsize=13, pad=12)
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x", alpha=0.2)

    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=9)

    st.pyplot(fig)


def render_risk_distribution_chart(dataframe):
    risk_counts = dataframe["gaming_addiction_risk_level"].value_counts()
    colors = ["#4caf50", "#ffb74d", "#ef4444", "#8e24aa"]

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.pie(
        risk_counts,
        labels=risk_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors[: len(risk_counts)],
        textprops={"color": "#111827", "fontsize": 9},
    )
    ax.set_title("Risk Level Distribution", fontsize=13, pad=12)
    ax.axis("equal")

    st.pyplot(fig)


def render_per_user_prediction_card(filtered_df):
    model, _ = train_global_model()

    if len(filtered_df) == 0:
        st.warning("No users match the selected filters.")
        return

    st.subheader("Per-User Prediction")

    user_options = ["Select a user"] + filtered_df["record_id"].tolist()
    selected_record = st.selectbox(
        "Select a user record",
        user_options,
        key="risk_prediction_record",
    )

    if selected_record == "Select a user":
        st.info("Choose a specific user record to view the prediction and explanation panel.")
        return

    user_row = filtered_df[filtered_df["record_id"] == selected_record].iloc[0]
    prediction_frame = build_prediction_features(user_row)
    prediction, confidence = predict_single_user(model, user_row)

    st.success(f"Predicted risk level: {prediction}")
    st.write(f"Selected user: {selected_record}")
    st.write(f"Model confidence: {confidence:.0%}")
    st.write(_simple_user_summary(prediction))

    render_explainable_ai_panel(user_row, model, prediction_frame)


def render_risk_predictions_tab(filtered_df):
    st.header("Risk Predictions")
    st.caption("This tab shows the model prediction for user risk and the main reasons behind it.")

    if len(filtered_df) == 0:
        st.warning("No users match the selected filters.")
        return

    total_users = len(filtered_df)
    high_risk_pct = (filtered_df["gaming_addiction_risk_level"].astype(str).str.strip().str.lower() == "high").mean() * 100
    severe_risk_pct = (filtered_df["gaming_addiction_risk_level"].astype(str).str.strip().str.lower() == "severe").mean() * 100
    low_risk_pct = (filtered_df["gaming_addiction_risk_level"].astype(str).str.strip().str.lower() == "low").mean() * 100
    moderate_risk_pct = (filtered_df["gaming_addiction_risk_level"].astype(str).str.strip().str.lower() == "moderate").mean() * 100

    card1, card2, card3, card4 = st.columns(4)
    card1.metric("Total Users", total_users)
    card2.metric("High Risk %", f"{high_risk_pct:.1f}%")
    card3.metric("Severe Risk %", f"{severe_risk_pct:.1f}%")
    card4.metric("Low + Moderate %", f"{low_risk_pct + moderate_risk_pct:.1f}%")

    render_per_user_prediction_card(filtered_df)

    st.subheader("Risk Charts")
    chart_left, chart_right = st.columns(2, gap="medium")

    with chart_left:
        with st.container(border=True):
            render_risk_distribution_chart(filtered_df)

    with chart_right:
        with st.container(border=True):
            render_global_model_importance(filtered_df)
