import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def binary_decision_tree(df, mode = 'entropy', tree_max_depth = 3):
    data = df.copy()
    
    # add binary column for being at risk
    data['at_risk'] = data['gaming_addiction_risk_level'].isin(['High', 'Severe'])
    
    # remove unnecessary columns
    features = data.drop(columns=['gaming_addiction_risk_level', 'at_risk'], axis=1)
    features = features.drop(columns=['record_id'])

    # convert features into numbers
    X = pd.get_dummies(features)
    Y = data['at_risk']
    
    if mode not in ['entropy', 'gini']:
        mode = 'entropy'

    clf = tree.DecisionTreeClassifier(criterion=mode, max_depth=tree_max_depth)
    clf = clf.fit(X, Y)
    fig, ax = plt.subplots(figsize=(14, 5))
    tree.plot_tree(
        clf,
        feature_names=X.columns,
        class_names=['Lower Risk', 'Higher risk'],
        filled=True,
        ax=ax,
    )

    title = f"Decision Tree ({mode.title()} | Max Depth: {tree_max_depth})"
    ax.set_title(title, fontsize=12, pad=14)
    st.pyplot(fig)

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