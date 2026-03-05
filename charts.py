import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def addiction_distribution(df):

    fig, ax = plt.subplots()

    df["gaming_addiction_risk_level"].value_counts().plot(
        kind="bar",
        ax=ax,
        color=["green","orange","red"]
    )

    ax.set_xlabel("Addiction Risk Level")
    ax.set_ylabel("Users")

    st.pyplot(fig)


def gaming_vs_sleep(df):

    fig, ax = plt.subplots()

    sns.scatterplot(
        x="daily_gaming_hours",
        y="sleep_hours",
        data=df,
        ax=ax
    )

    ax.set_title("Gaming Hours vs Sleep")

    st.pyplot(fig)


def gaming_vs_gpa(df):

    fig, ax = plt.subplots()

    sns.scatterplot(
        x="daily_gaming_hours",
        y="grades_gpa",
        data=df,
        ax=ax
    )

    ax.set_title("Gaming Hours vs GPA")

    st.pyplot(fig)


def social_isolation(df):

    fig, ax = plt.subplots()

    sns.scatterplot(
        x="daily_gaming_hours",
        y="social_isolation_score",
        data=df,
        ax=ax
    )

    ax.set_title("Gaming Hours vs Social Isolation")

    st.pyplot(fig)