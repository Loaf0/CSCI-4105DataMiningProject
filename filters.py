import streamlit as st

def apply_filters(df):

    st.sidebar.header("Dashboard Filters")

    def set_default_state(key, value):
        if key not in st.session_state:
            st.session_state[key] = value

    min_age = int(df["age"].min())
    max_age = int(df["age"].max())

    min_hours = float(df["daily_gaming_hours"].min())
    max_hours = float(df["daily_gaming_hours"].max())

    platform_list = list(df["gaming_platform"].unique())
    genre_list = list(df["game_genre"].unique())

    age_key = "filter_age_range"
    hours_key = "filter_gaming_hours_range"
    platform_key = "filter_selected_platforms"
    genre_key = "filter_selected_genres"

    set_default_state(age_key, (min_age, max_age))
    set_default_state(hours_key, (min_hours, max_hours))
    set_default_state(platform_key, platform_list)
    set_default_state(genre_key, genre_list)

    if st.sidebar.button("Reset Filters", use_container_width=True):
        st.session_state[age_key] = (min_age, max_age)
        st.session_state[hours_key] = (min_hours, max_hours)
        st.session_state[platform_key] = platform_list
        st.session_state[genre_key] = genre_list
        st.rerun()

    # Age filter
    age_range = st.sidebar.slider(
        "Age Range",
        min_age,
        max_age,
        key=age_key
    )

    # Gaming hours filter
    gaming_hours_range = st.sidebar.slider(
        "Daily Gaming Hours",
        min_hours,
        max_hours,
        key=hours_key
    )

    # Platform filter
    selected_platforms = st.sidebar.multiselect(
        "Gaming Platform",
        platform_list,
        key=platform_key
    )

    # Genre filter
    selected_genres = st.sidebar.multiselect(
        "Game Genre",
        genre_list,
        key=genre_key
    )

    filtered_df = df[
        (df["age"].between(age_range[0], age_range[1])) &
        (df["daily_gaming_hours"].between(gaming_hours_range[0], gaming_hours_range[1])) &
        (df["gaming_platform"].isin(selected_platforms)) &
        (df["game_genre"].isin(selected_genres))
    ]

    return filtered_df