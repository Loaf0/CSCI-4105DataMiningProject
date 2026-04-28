import pandas as pd
import streamlit as st

from dashboard_tabs.behavior import render_behavior_tab
from dashboard_tabs.clustering import render_clustering_tab
from dashboard_tabs.decision_tree import render_decision_tree_tab
from dashboard_tabs.overview import render_overview_tab
from dashboard_tabs.mentalhealth import render_social_tab
from dashboard_tabs.risk_predictions import render_risk_predictions_tab
from dashboard_tabs.survey import render_survey_tab
from dashboard_tabs.user_explorer import render_user_explorer_tab
from filters import apply_filters


st.title("Gaming Habits & Mental Health Dashboard")


# Load data

df = pd.read_csv("cleaned_mentalHealth.csv")
filtered_df = apply_filters(df)


# Tabs

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "Overview",
        "Behavior Analysis",
        "Mental Health",
        "Risk Predictions",
        "User Explorer",
        "Decision Tree",
        "Survey",
        "Clustering",
    ]
)


with tab1:
    render_overview_tab(filtered_df)

with tab2:
    render_behavior_tab(filtered_df)

with tab3:
    render_social_tab(filtered_df)

with tab4:
    render_risk_predictions_tab(filtered_df)

with tab5:
    render_user_explorer_tab(filtered_df)

with tab6:
    render_decision_tree_tab(filtered_df)

with tab7:
    render_survey_tab()

with tab8:
    render_clustering_tab(filtered_df)
