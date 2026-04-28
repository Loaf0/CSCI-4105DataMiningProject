import pandas as pd
import streamlit as st

from dashboard_tabs.shared import SURVEY_QUESTION_ITEMS


def render_survey_tab():
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

        for item in SURVEY_QUESTION_ITEMS:
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
