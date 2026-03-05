import streamlit as st

def apply_filters(df):

    st.sidebar.header("Dashboard Filters")

    # Age filter
    min_age = int(df["age"].min())
    max_age = int(df["age"].max())

    age_range = st.sidebar.slider(
        "Age Range",
        min_age,
        max_age,
        (min_age, max_age)
    )

    # Gaming hours filter
    min_hours = float(df["daily_gaming_hours"].min())
    max_hours = float(df["daily_gaming_hours"].max())

    gaming_hours_range = st.sidebar.slider(
        "Daily Gaming Hours",
        min_hours,
        max_hours,
        (min_hours, max_hours)
    )

    # Platform filter
    platform_list = df["gaming_platform"].unique()

    selected_platforms = st.sidebar.multiselect(
        "Gaming Platform",
        platform_list,
        default=platform_list
    )

    # Genre filter
    genre_list = df["game_genre"].unique()

    selected_genres = st.sidebar.multiselect(
        "Game Genre",
        genre_list,
        default=genre_list
    )

    filtered_df = df[
        (df["age"].between(age_range[0], age_range[1])) &
        (df["daily_gaming_hours"].between(gaming_hours_range[0], gaming_hours_range[1])) &
        (df["gaming_platform"].isin(selected_platforms)) &
        (df["game_genre"].isin(selected_genres))
    ]

    return filtered_df