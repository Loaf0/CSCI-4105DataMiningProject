import pandas as pd
import streamlit as st

from dashboard_tabs.shared import SURVEY_QUESTION_ITEMS


def render_survey_tab():
    st.header("Survey")
    st.caption("Fill out the questions below, then click Review Answers to see your responses.")

    with st.form("survey_form"):
        responses = []

        for item in SURVEY_QUESTION_ITEMS:
            st.write(item["question"])

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

        submitted = st.form_submit_button("Review Answers")

    if submitted:
        st.success("Survey answers captured successfully.")
        st.dataframe(pd.DataFrame(responses), use_container_width=True, hide_index=True)
