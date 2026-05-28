import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn import tree
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def binary_decision_tree(df, mode = 'entropy', tree_max_depth = 3):
    data = df.copy()
    
    # add binary column for being at risk
    data['at_risk'] = data['gaming_addiction_risk_level'].isin(['High', 'Severe'])
    
    # remove unnecessary columns
    features = data.drop(columns=['gaming_addiction_risk_level', 'at_risk'])
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

    fig, ax = plt.subplots(figsize=(5.2, 5.2))

    risk_counts = df["gaming_addiction_risk_level"].value_counts()
    colors = ["#4caf50", "#ffb74d", "#ef5350", "#8e24aa"]

    ax.pie(
        risk_counts,
        labels=risk_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors[: len(risk_counts)],
        textprops={"color": "#111827", "fontsize": 10},
    )
    ax.set_title("Risk Level Distribution")
    ax.axis("equal")

    st.pyplot(fig)


def gaming_vs_sleep(df):

    fig, ax = plt.subplots()

    ax.scatter(df["daily_gaming_hours"], df["sleep_hours"], alpha=0.7, color="#2563eb")

    ax.set_title("Gaming Hours vs Sleep Hours")
    ax.set_xlabel("Daily Gaming Hours")
    ax.set_ylabel("Sleep Hours")

    st.pyplot(fig)


def gaming_vs_gpa(df):

    fig, ax = plt.subplots()

    ax.scatter(df["daily_gaming_hours"], df["grades_gpa"], alpha=0.7, color="#0f766e")

    ax.set_title("Gaming Hours vs GPA")
    ax.set_xlabel("Daily Gaming Hours")
    ax.set_ylabel("GPA")

    st.pyplot(fig)


def social_isolation(df):

    fig, ax = plt.subplots()

    ax.scatter(df["daily_gaming_hours"], df["social_isolation_score"], alpha=0.7, color="#ef4444")

    ax.set_title("Gaming Hours vs Social Isolation")
    ax.set_xlabel("Daily Gaming Hours")
    ax.set_ylabel("Social Isolation Score")

    st.pyplot(fig)


def build_clustering_summary(df):
    data = df.copy()
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    data = data.drop(columns=['record_id', 'gaming_addiction_risk_level'])

    features = [
        'daily_gaming_hours',
        'sleep_hours',
    ]

    imputer = SimpleImputer(strategy='median')
    x = imputer.fit_transform(data[features])

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    best_k = None
    best_score = -1
    best_labels = None
    best_model = None

    max_k = min(10, len(data) - 1)
    if max_k < 2:
        return None

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(x_scaled)
        score = silhouette_score(x_scaled, labels)
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
            best_model = kmeans

    if best_k is None or best_labels is None or best_model is None:
        return None

    cluster_frame = pd.DataFrame(
        {
            "cluster": best_labels,
            "daily_gaming_hours": pd.to_numeric(df["daily_gaming_hours"], errors="coerce"),
            "sleep_hours": pd.to_numeric(df["sleep_hours"], errors="coerce"),
            "social_isolation_score": pd.to_numeric(df["social_isolation_score"], errors="coerce"),
            "withdrawal_symptoms": df["withdrawal_symptoms"].map({"Yes": 1, "No": 0, True: 1, False: 0, "yes": 1, "no": 0}).fillna(0),
            "risk": pd.to_numeric(df["gaming_addiction_risk_level"].map({"Low": 0, "Moderate": 1, "High": 2, "Severe": 3}), errors="coerce").fillna(0),
            "risk_label": df["gaming_addiction_risk_level"].astype(str).str.strip().str.title(),
        }
    )
    cluster_sizes = cluster_frame["cluster"].value_counts().sort_index()
    cluster_distribution = (cluster_sizes / len(cluster_frame) * 100).sort_index()
    cluster_risk_means = cluster_frame.groupby("cluster")["risk"].mean().sort_values(ascending=True)
    cluster_distance = pd.Series(
        ((x_scaled - best_model.cluster_centers_[best_labels]) ** 2).sum(axis=1) ** 0.5
    )

    cluster_name_map = {cluster_id: f"Cluster {cluster_id}" for cluster_id in cluster_sizes.index}
    highest_risk_cluster_id = cluster_risk_means.idxmax()
    largest_cluster_id = cluster_sizes.idxmax()
    avg_within_distance = float(cluster_distance.mean())

    if avg_within_distance < 0.85:
        within_distance_label = "Low"
    elif avg_within_distance < 1.2:
        within_distance_label = "Moderate"
    else:
        within_distance_label = "High"

    def describe_isolation(value):
        if value < 4:
            return "Low"
        if value < 7:
            return "Medium"
        return "High"

    def describe_withdrawal(value):
        if value < 0.33:
            return "Low"
        if value < 0.67:
            return "Medium"
        return "High"

    def describe_risk(value):
        if value < 0.75:
            return "Low"
        if value < 1.75:
            return "Moderate"
        return "Severe"

    cluster_means = cluster_frame.groupby("cluster").agg(
        {
            "daily_gaming_hours": "mean",
            "sleep_hours": "mean",
            "social_isolation_score": "mean",
            "withdrawal_symptoms": "mean",
            "risk": "mean",
        }
    ).sort_index()

    risk_levels = ["Low", "Moderate", "High", "Severe"]
    cluster_risk_overlay_table = (
        pd.crosstab(cluster_frame["cluster"], cluster_frame["risk_label"])
        .reindex(index=cluster_sizes.index, columns=risk_levels, fill_value=0)
    )
    cluster_risk_overlay_table = cluster_risk_overlay_table.div(cluster_risk_overlay_table.sum(axis=1), axis=0).fillna(0) * 100

    cluster_profile_table = pd.DataFrame(index=["Gaming Hours", "Sleep Hours", "Isolation", "Withdrawal", "Risk Level"])
    for cluster_id in cluster_means.index:
        cluster_label = cluster_name_map[cluster_id]
        cluster_profile_table[cluster_label] = [
            f"{cluster_means.loc[cluster_id, 'daily_gaming_hours']:.1f}",
            f"{cluster_means.loc[cluster_id, 'sleep_hours']:.1f}",
            describe_isolation(cluster_means.loc[cluster_id, 'social_isolation_score']),
            describe_withdrawal(cluster_means.loc[cluster_id, 'withdrawal_symptoms']),
            describe_risk(cluster_means.loc[cluster_id, 'risk']),
        ]

    return {
        "best_k": int(best_k),
        "best_score": float(best_score),
        "cluster_labels": best_labels,
        "cluster_name_map": cluster_name_map,
        "largest_cluster_label": cluster_name_map[largest_cluster_id],
        "largest_cluster_pct": float(cluster_distribution.loc[largest_cluster_id]),
        "highest_risk_cluster_label": cluster_name_map[highest_risk_cluster_id],
        "silhouette_score": float(best_score),
        "avg_within_distance_label": within_distance_label,
        "avg_within_distance": avg_within_distance,
        "cluster_profile_table": cluster_profile_table,
        "cluster_risk_overlay_table": cluster_risk_overlay_table,
        "x_scaled": x_scaled,
    }


def clustering_result(df, summary=None):
    if summary is None:
        summary = build_clustering_summary(df)

    if summary is None:
        st.warning("Not enough records to build clustering metrics.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(summary["x_scaled"][:, 0], summary["x_scaled"][:, 1], c=summary["cluster_labels"], cmap="viridis")
    ax.set_title("K-Means Clustering")
    ax.set_xlabel("Scaled Daily Gaming Hours")
    ax.set_ylabel("Scaled Sleep Hours")
    ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")

    st.write(f"Best k: {summary['best_k']} | Silhouette: {round(summary['silhouette_score'], 3)}")
    st.pyplot(fig)