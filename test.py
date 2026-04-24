import streamlit as st
import pandas as pd

#from dataHandling import load_clean_data
from filters import apply_filters
from charts import *

st.title("Gaming Habits & Mental Health Dashboard")

# Load data

df = pd.read_csv("cleaned_mentalHealth.csv")

survey_question_items = [
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

# Apply filters
filtered_df = apply_filters(df)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Overview",
    "Behavior Analysis",
    "Social Impact",
    "Gaming Behavior",
    "User Explorer",
    "Decision Tree",
    "Survey",
    "Clustering"
])


# Overview


with tab1:

    st.header("Dataset Overview")

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Total Users", len(filtered_df))
    col2.metric("Avg Gaming Hours", round(filtered_df["daily_gaming_hours"].mean(),2))
    col3.metric("Avg Sleep Hours", round(filtered_df["sleep_hours"].mean(),2))
    col4.metric(
        "High Risk %",
        round((filtered_df["gaming_addiction_risk_level"]=="High").mean()*100,2)
    )

    addiction_distribution(filtered_df)


# Behavior


with tab2:

    st.header("Gaming Behavior")

    col1,col2 = st.columns(2)

    with col1:
        gaming_vs_sleep(filtered_df)

    with col2:
        gaming_vs_gpa(filtered_df)


# Social


with tab3:

    st.header("Social Impact")

    social_isolation(filtered_df)


# Gaming


with tab4:

    st.header("Genre Distribution")

    st.bar_chart(filtered_df["game_genre"].value_counts())

    st.header("Platform Distribution")

    st.bar_chart(filtered_df["gaming_platform"].value_counts())


# User Explorer


with tab5:

    st.header("User Explorer")

    if len(filtered_df) > 0:

        user_id = st.selectbox(
            "Select Record ID",
            filtered_df["record_id"]
        )

        user = filtered_df[filtered_df["record_id"] == user_id]

        st.dataframe(user)

    else:

        st.warning("No users match the selected filters.")

# Decision Tree

with tab6:

    st.header("Binary Decision Tree")

    if "tree_mode" not in st.session_state:
        st.session_state.tree_mode = "entropy"

    if st.button("Entropy / Gini"):
        st.session_state.tree_mode = "gini" if st.session_state.tree_mode == "entropy" else "entropy"

    tree_depth = st.number_input("Max Depth", min_value=1, max_value=10, value=3, step=1)

    binary_decision_tree(filtered_df, mode=st.session_state.tree_mode, tree_max_depth=int(tree_depth))


# Survey

with tab7:

    st.markdown("## Survey")
    st.markdown(
        """
        <style>
        .survey-hero {
            background: linear-gradient(135deg, rgba(26, 37, 48, 0.96), rgba(47, 62, 78, 0.96));
            color: white;
            padding: 1.1rem 1.25rem;
            border-radius: 16px;
            margin-bottom: 1rem;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .survey-hero h3 {
            margin: 0 0 0.35rem 0;
            font-size: 1.15rem;
        }
        .survey-hero p {
            margin: 0;
            opacity: 0.9;
            line-height: 1.5;
        }
        .survey-section {
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: #5b6570;
        }
        .survey-question {
            padding: 1rem 1rem 0.75rem 1rem;
            border: 1px solid #e6e8eb;
            border-radius: 14px;
            background: white;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
            margin-bottom: 0.85rem;
        }
        .survey-question p {
            margin: 0 0 0.65rem 0;
            font-weight: 600;
            color: #1f2937;
        }
        </style>
        <div class="survey-hero">
            <h3>Survey</h3>
            <p>Use the fields below to answer the survey. Press Review Answers when you are finished to see your responses.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("survey_form"):
        responses = []

        for item in survey_question_items:
            st.markdown('<div class="survey-question">', unsafe_allow_html=True)
            st.markdown(f"<p>{item['question']}</p>", unsafe_allow_html=True)

            left, right = st.columns([3, 2])

            if item["type"] == "select":
                with right:
                    answer = st.selectbox(
                        "Select answer",
                        item["options"],
                        key=f"survey_{item['field']}"
                    )
            else:
                with right:
                    answer = st.number_input(
                        "Enter value",
                        min_value=float(item["min"]) if item["min"] is not None else None,
                        max_value=float(item["max"]) if item["max"] is not None else None,
                        value=float(item["min"]) if item["min"] is not None else 0.0,
                        step=1.0,
                        key=f"survey_{item['field']}"
                    )

            responses.append({
                "Field": item["field"],
                "Question": item["question"],
                "Answer": answer,
            })

            st.markdown('</div>', unsafe_allow_html=True)

        submitted = st.form_submit_button("Review Answers")

    if submitted:
        st.success("Survey answers captured successfully.")
        st.dataframe(pd.DataFrame(responses), use_container_width=True, hide_index=True)


    # Clustering

    with tab8:

        st.header("K-Means Clustering")
        clustering_result(filtered_df)