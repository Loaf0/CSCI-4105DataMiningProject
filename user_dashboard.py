import pandas as pd
import streamlit as st
from ai_implementation_test import get_ai_advice
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = [
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


SURVEY_QUESTIONS = [
    {
        "field": "daily_gaming_hours",
        "question": "In the last 30 days, on average, how many hours per day do you spend playing video games?",
        "type": "number",
        "min": 0,
        "max": 24,
        "step": 0.5,
        "default": 2.0,
    },
    {
        "field": "face_to_face_social_hours_weekly",
        "question": "On average, how many hours per week do you spend on face-to-face social interactions?",
        "type": "number",
        "min": 0,
        "max": 168,
        "step": 1.0,
        "default": 10.0,
    },
    {
        "field": "sleep_hours",
        "question": "On average, how many hours do you sleep per night?",
        "type": "number",
        "min": 0,
        "max": 24,
        "step": 0.5,
        "default": 7.0,
    },
    {
        "field": "exercise_hours_weekly",
        "question": "On average, how many hours per week do you spend exercising?",
        "type": "number",
        "min": 0,
        "max": 168,
        "step": 1.0,
        "default": 3.0,
    },
    {
        "field": "social_isolation_score",
        "question": "How socially isolated have you felt? (1 = not at all, 10 = extremely isolated)",
        "type": "number",
        "min": 1,
        "max": 10,
        "step": 1.0,
        "default": 4.0,
    },
    {
        "field": "monthly_game_spending_usd",
        "question": "On average, how much money do you spend on video games per month, including in-game purchases?",
        "type": "number",
        "min": 0,
        "max": None,
        "step": 5.0,
        "default": 20.0,
    },
    {
        "field": "loss_of_other_interests",
        "question": "Have you lost interest in hobbies or activities you used to enjoy?",
        "type": "binary",
        "default": "no",
    },
    {
        "field": "withdrawal_symptoms",
        "question": "Have you experienced symptoms of withdrawal when you are not able to play games?",
        "type": "binary",
        "default": "no",
    },
    {
        "field": "back_neck_pain",
        "question": "Have you experienced back or neck pain related to gaming?",
        "type": "binary",
        "default": "no",
    },
    {
        "field": "continued_despite_problems",
        "question": "Have you continued to play games even when it caused problems?",
        "type": "binary",
        "default": "no",
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


def prepare_training_data(training_df):
    features = training_df[FEATURE_COLUMNS].copy()

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
def train_model():
    training_df = pd.read_csv("cleaned_mentalHealth.csv")
    features, target = prepare_training_data(training_df)

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

    metrics = {
        "accuracy": accuracy_score(y_test, model.predict(x_test)),
        "f1": f1_score(y_test, model.predict(x_test), average="weighted"),
        "recall": recall_score(y_test, model.predict(x_test), average="weighted"),
    }

    return model, metrics


def build_survey_row():
    values = {}

    for item in SURVEY_QUESTIONS:
        if item["type"] == "binary":
            answer = st.radio(
                item["question"],
                ["No", "Yes"],
                horizontal=True,
                index=0 if item["default"] == "no" else 1,
                key=f"survey_{item['field']}",
            )
            values[item["field"]] = normalize_binary(answer)
        else:
            answer = st.number_input(
                item["question"],
                min_value=float(item["min"]) if item["min"] is not None else None,
                max_value=float(item["max"]) if item["max"] is not None else None,
                value=float(item["default"]),
                step=float(item["step"]),
                key=f"survey_{item['field']}",
            )
            values[item["field"]] = float(answer)

    return pd.DataFrame([values], columns=FEATURE_COLUMNS)


def score_concern(value, field):
    if field == "daily_gaming_hours":
        return min(max(value / 8.0, 0.0), 1.0)
    if field == "face_to_face_social_hours_weekly":
        return min(max((20.0 - value) / 20.0, 0.0), 1.0)
    if field == "sleep_hours":
        return min(max((8.0 - value) / 8.0, 0.0), 1.0)
    if field == "exercise_hours_weekly":
        return min(max((5.0 - value) / 5.0, 0.0), 1.0)
    if field == "social_isolation_score":
        return min(max((value - 1.0) / 9.0, 0.0), 1.0)
    if field == "monthly_game_spending_usd":
        return min(value / 150.0, 1.0)
    return 1.0 if value == 1 else 0.0


def explain_prediction(model, survey_df):
    importance_map = dict(zip(FEATURE_COLUMNS, model.feature_importances_))
    reasons = []

    reason_text = {
        "daily_gaming_hours": "gaming time is high",
        "loss_of_other_interests": "you reported losing interest in other activities",
        "withdrawal_symptoms": "you reported withdrawal symptoms when not gaming",
        "back_neck_pain": "you reported physical discomfort related to gaming",
        "face_to_face_social_hours_weekly": "your face-to-face social time is limited",
        "monthly_game_spending_usd": "your monthly game spending is elevated",
        "social_isolation_score": "your isolation score is relatively high",
        "continued_despite_problems": "you continued gaming even when it caused problems",
        "sleep_hours": "your sleep duration is below a healthy range",
        "exercise_hours_weekly": "your weekly exercise time is low",
    }

    for field in FEATURE_COLUMNS:
        value = survey_df.iloc[0][field]
        concern = score_concern(value, field)
        weighted_score = concern * importance_map.get(field, 0.0)
        if weighted_score > 0:
            reasons.append((weighted_score, reason_text[field]))

    reasons.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in reasons[:3]]


def format_survey_for_ai(survey_df):
    row = survey_df.iloc[0]
    lines = []

    for field in FEATURE_COLUMNS:
        value = row[field]
        lines.append(f"{field}: {value}")

    return "\n".join(lines)



st.markdown(
    """
    <style>
    .user-hero {
        background: linear-gradient(135deg, #112233, #2d4a66);
        color: white;
        padding: 1.4rem 1.5rem;
        border-radius: 20px;
        margin-bottom: 1rem;
        box-shadow: 0 16px 40px rgba(10, 20, 35, 0.22);
    }
    .user-hero h2 {
        margin: 0 0 0.35rem 0;
        font-size: 1.7rem;
    }
    .user-hero p {
        margin: 0;
        opacity: 0.92;
        line-height: 1.5;
    }
    .question-card {
        background: #ffffff;
        border: 1px solid #e6e8ee;
        border-radius: 16px;
        padding: 1rem 1rem 0.85rem 1rem;
        margin-bottom: 0.85rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
    }
    .question-card p {
        margin: 0 0 0.75rem 0;
        font-weight: 600;
        color: #1f2937;
        line-height: 1.45;
    }
    </style>
    <div class="user-hero">
        <h2>Gaming Addiction Survey</h2>
        <p>Answer the questions below to receive your predicted risk level, a short reason, and practical recommendations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

model, metrics = train_model()



with st.form("user_survey_form"):
    survey_df = build_survey_row()
    submitted = st.form_submit_button("Get My Result")


if submitted:
    prediction = model.predict(survey_df)[0]
    probability = float(max(model.predict_proba(survey_df)[0]))
    reasons = explain_prediction(model, survey_df)
    st.session_state["latest_result"] = {
        "survey_df": survey_df.copy(),
        "prediction": prediction,
        "probability": probability,
        "reasons": reasons,
    }
    st.session_state["ai_life_recommendations"] = ""


latest_result = st.session_state.get("latest_result")

if latest_result:
    survey_df = latest_result["survey_df"]
    prediction = latest_result["prediction"]
    probability = latest_result["probability"]
    reasons = latest_result["reasons"]

    st.markdown(
        f"""
        <div style="background:#0f172a;color:white;padding:1rem 1.2rem;border-radius:16px;margin-bottom:1rem;">
            <div style="font-size:0.9rem;opacity:0.85;">Predicted risk level</div>
            <div style="font-size:1.8rem;font-weight:700;">{prediction}</div>
            <div style="font-size:0.95rem;opacity:0.9;">Model confidence: {probability:.0%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Reason")
        if reasons:
            for item in reasons:
                st.write(f"- {item}")
        else:
            st.write("- Your answers do not show a strong risk signal in the model.")

    with right_col:
        st.subheader("AI Generated Recommendations")
        st.markdown(
            """
            <div style="background:linear-gradient(135deg, #0f172a, #1f2937);color:white;padding:1rem 1.15rem;border-radius:16px;box-shadow:0 12px 24px rgba(15,23,42,0.12);">
            """,
            unsafe_allow_html=True,
        )
        button_col, output_col = st.columns([1, 1.45], gap="medium")

        with button_col:
            if st.button("Get Recommened Life Changes", key="get_ai_life_recommendations", use_container_width=True):
                with st.spinner("Getting Life Recommendations..."):
                    try:
                        survey_summary = format_survey_for_ai(survey_df)
                        st.session_state["ai_life_recommendations"] = get_ai_advice(survey_summary)
                    except Exception:
                        st.session_state["ai_life_recommendations"] = ""
                        st.error("Our serivce is currently down. Please try again later.")

        with output_col:
            ai_recommendations = st.session_state.get("ai_life_recommendations", "")
            if ai_recommendations:
                st.write(ai_recommendations)
            else:
                st.info("Generate AI recommendations for this survey using the button on the left.")

        st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Your Answers")
    display_df = survey_df.copy()
    display_df["loss_of_other_interests"] = display_df["loss_of_other_interests"].map({0: "No", 1: "Yes"})
    display_df["withdrawal_symptoms"] = display_df["withdrawal_symptoms"].map({0: "No", 1: "Yes"})
    display_df["back_neck_pain"] = display_df["back_neck_pain"].map({0: "No", 1: "Yes"})
    display_df["continued_despite_problems"] = display_df["continued_despite_problems"].map({0: "No", 1: "Yes"})
    st.dataframe(display_df.T.rename(columns={0: "Answer"}), use_container_width=True)
