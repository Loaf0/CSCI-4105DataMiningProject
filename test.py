import streamlit as st
import pandas as pd

#from dataHandling import load_clean_data
from filters import apply_filters
from charts import *

st.set_page_config(layout="wide")

st.title("Gaming Habits & Mental Health Dashboard")

# Load data

df = pd.read_csv("cleaned_mentalHealth.csv")

# Apply filters
filtered_df = apply_filters(df)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Behavior Analysis",
    "Social Impact",
    "Gaming Behavior",
    "User Explorer"
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