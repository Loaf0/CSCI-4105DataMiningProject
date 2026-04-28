import pandas as pd
import streamlit as st

from charts import build_clustering_summary
from dashboard_tabs.mentalhealth import build_social_scores
from dashboard_tabs.risk_predictions import build_prediction_features
from dashboard_tabs.risk_predictions import build_user_explanation
from dashboard_tabs.risk_predictions import predict_single_user
from dashboard_tabs.risk_predictions import train_global_model


def _format_value(value, suffix=""):
    if pd.isna(value):
        return "Unknown"
    if isinstance(value, (int, float)):
        if float(value).is_integer():
            return f"{int(value)}{suffix}"
        return f"{float(value):.1f}{suffix}"
    return f"{value}{suffix}"


def _add_cluster_labels(dataframe):
    summary = build_clustering_summary(dataframe)
    if summary is None:
        working_df = dataframe.copy()
        working_df["cluster_label"] = "Unavailable"
        return working_df

    working_df = dataframe.copy()
    working_df["cluster_label"] = [f"Cluster {cluster_id}" for cluster_id in summary["cluster_labels"]]
    return working_df


def _risk_color(prediction):
    palette = {
        "Low": "#15803d",
        "Moderate": "#d97706",
        "High": "#ea580c",
        "Severe": "#b91c1c",
    }
    return palette.get(str(prediction).title(), "#374151")


def _cluster_risk_segment(cluster_label, prediction):
    if prediction == "Severe":
        return f"{cluster_label} – Severe High-Risk Users"
    if prediction == "High":
        return f"{cluster_label} – High-Risk Users"
    if prediction == "Moderate":
        return f"{cluster_label} – Moderate-Risk Users"
    return f"{cluster_label} – Lower-Risk Users"


def _severity_label(score, inverse=False):
    value = float(score)
    if inverse:
        if value < 4.5:
            return "Low"
        if value < 7.0:
            return "Moderate"
        return "High"

    if value < 3.5:
        return "Low"
    if value < 6.5:
        return "Moderate"
    return "High"


def _render_snapshot_card(title, value, level, accent):
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg, {accent}, #111827);color:white;padding:1rem 1.05rem;border-radius:16px;box-shadow:0 12px 24px rgba(15,23,42,0.12);min-height:96px;">
            <div style="font-size:0.82rem;opacity:0.8;margin-bottom:0.35rem;">{title}</div>
            <div style="font-size:1.45rem;font-weight:800;line-height:1.1;">{value}</div>
            <div style="margin-top:0.35rem;font-size:0.9rem;opacity:0.9;">{level}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_mental_health_snapshot(user_row):
    st.subheader("Mental Health Snapshot")

    user_frame = user_row.to_frame().T
    scores = build_social_scores(user_frame)

    mood_score = float(scores["mood_score"].iloc[0])
    anxiety_score = float(scores["anxiety"].iloc[0])
    stress_score = float(scores["stress"].iloc[0])
    isolation_score = float(scores["isolation"].iloc[0])

    card1, card2, card3, card4 = st.columns(4)
    with card1:
        _render_snapshot_card("Mood State", f"{mood_score:.1f}/10", _severity_label(mood_score, inverse=True), "#0f766e")
    with card2:
        _render_snapshot_card("Anxiety", f"{anxiety_score:.1f}/10", _severity_label(anxiety_score), "#d97706")
    with card3:
        _render_snapshot_card("Stress", f"{stress_score:.1f}/10", _severity_label(stress_score), "#ea580c")
    with card4:
        _render_snapshot_card("Isolation", f"{isolation_score:.1f}/10", _severity_label(isolation_score), "#7c3aed")


def build_risk_flags(user_row, reference_df):
    flags = []

    daily_hours = float(pd.to_numeric(user_row["daily_gaming_hours"], errors="coerce"))
    sleep_hours = float(pd.to_numeric(user_row["sleep_hours"], errors="coerce"))
    social_hours = float(pd.to_numeric(user_row["face_to_face_social_hours_weekly"], errors="coerce"))
    isolation_score = float(pd.to_numeric(user_row["social_isolation_score"], errors="coerce"))

    avg_daily = pd.to_numeric(reference_df["daily_gaming_hours"], errors="coerce").mean()
    avg_isolation = pd.to_numeric(reference_df["social_isolation_score"], errors="coerce").mean()

    if sleep_hours < 7:
        flags.append("Sleep below healthy range")
    if daily_hours > avg_daily:
        flags.append("Gaming exceeds population average")
    if isolation_score >= max(7.0, avg_isolation + 1.0):
        flags.append("Social isolation elevated")
    if social_hours < 5:
        flags.append("Low offline social time")

    return flags


def render_risk_flags_section(user_row, reference_df):
    st.subheader("Risk Flags Section")
    flags = build_risk_flags(user_row, reference_df)

    if not flags:
        st.success("No major risk flags detected for the current selection.")
        return

    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg, #111827, #1f2937);color:white;padding:1rem 1.15rem;border-radius:16px;box-shadow:0 12px 24px rgba(15,23,42,0.12);">
            <div style="font-size:0.92rem;opacity:0.8;margin-bottom:0.5rem;">Auto alerts</div>
            <ul style="margin:0;padding-left:1.15rem;line-height:1.75;">
                {''.join(f'<li>⚠ {flag}</li>' for flag in flags)}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_personalized_recommendations(user_row, prediction):
    recommendations = []

    daily_hours = float(pd.to_numeric(user_row["daily_gaming_hours"], errors="coerce"))
    sleep_hours = float(pd.to_numeric(user_row["sleep_hours"], errors="coerce"))
    social_hours = float(pd.to_numeric(user_row["face_to_face_social_hours_weekly"], errors="coerce"))
    exercise_hours = float(pd.to_numeric(user_row["exercise_hours_weekly"], errors="coerce"))

    if daily_hours >= 5:
        recommendations.append("Reduce gaming to under 5h/day")
    if sleep_hours < 7:
        recommendations.append("Sleep 7+ hours consistently")
    if social_hours < 6:
        recommendations.append("Schedule offline social activities")
    if daily_hours >= 4:
        recommendations.append("Take breaks every hour")
    if exercise_hours < 3:
        recommendations.append("Add at least one short exercise session each week")

    if prediction == "Severe":
        recommendations.append("Consider speaking with a counselor or support professional")

    if not recommendations:
        recommendations.append("Maintain the current healthy balance and recheck habits periodically")

    return recommendations[:5]


def render_personalized_recommendations(user_row, prediction):
    st.subheader("Personalized Recommendations")
    recommendations = build_personalized_recommendations(user_row, prediction)

    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg, #0f172a, #1f2937);color:white;padding:1rem 1.15rem;border-radius:16px;box-shadow:0 12px 24px rgba(15,23,42,0.12);">
            <div style="font-size:0.92rem;opacity:0.8;margin-bottom:0.5rem;">Tailored suggestions based on the selected user's profile</div>
            <ul style="margin:0;padding-left:1.1rem;line-height:1.75;">
                {''.join(f'<li>{item}</li>' for item in recommendations)}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_user_search_panel(dataframe):
    st.subheader("User Search / Selection Panel")

    search_col, risk_col, cluster_col = st.columns([1.6, 1, 1], gap="medium")

    with search_col:
        search_term = st.text_input("Search by User ID", placeholder="e.g. GD0001")

    risk_options = ["Low", "Moderate", "High", "Severe"]
    available_risks = [risk for risk in risk_options if risk in dataframe["gaming_addiction_risk_level"].astype(str).str.title().unique()]

    with risk_col:
        selected_risks = st.multiselect(
            "Filter by Risk Level",
            options=risk_options,
            default=available_risks if available_risks else risk_options,
        )

    available_clusters = sorted(dataframe["cluster_label"].dropna().astype(str).unique(), key=lambda value: (value == "Unavailable", value))
    with cluster_col:
        selected_clusters = st.multiselect(
            "Filter by Cluster",
            options=available_clusters,
            default=available_clusters,
        )

    filtered_users = dataframe.copy()
    if search_term.strip():
        filtered_users = filtered_users[
            filtered_users["record_id"].astype(str).str.contains(search_term.strip(), case=False, na=False)
        ]

    if selected_risks:
        filtered_users = filtered_users[
            filtered_users["gaming_addiction_risk_level"].astype(str).str.title().isin(selected_risks)
        ]

    if selected_clusters:
        filtered_users = filtered_users[filtered_users["cluster_label"].isin(selected_clusters)]

    user_options = ["Select a user"] + filtered_users["record_id"].astype(str).tolist()
    selected_user = None

    if len(user_options) > 0:
        selected_user = st.selectbox("Select User", user_options, format_func=lambda value: value if value == "Select a user" else f"# {value}")
        if selected_user == "Select a user":
            selected_user = None
    else:
        st.warning("No users match the selected filters.")

    return filtered_users, selected_user


def render_user_profile_card(user_row):
    st.subheader("User Profile Card")

    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg, #111827, #1f2937);color:white;padding:1.15rem 1.25rem;border-radius:18px;box-shadow:0 14px 30px rgba(15,23,42,0.14);">
            <div style="font-size:0.92rem;opacity:0.8;margin-bottom:0.55rem;">Basic user details</div>
            <div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0.7rem 1rem;">
                <div><strong>User ID:</strong> {user_row['record_id']}</div>
                <div><strong>Age:</strong> {_format_value(user_row['age'])}</div>
                <div><strong>Gender:</strong> {_format_value(user_row['gender'])}</div>
                <div><strong>Primary Platform:</strong> {_format_value(user_row['gaming_platform'])}</div>
                <div><strong>Favorite Genre:</strong> {_format_value(user_row['game_genre'])}</div>
                <div><strong>Years Gaming:</strong> {_format_value(user_row['years_gaming'])}</div>
                <div><strong>Risk Level:</strong> {_format_value(user_row['gaming_addiction_risk_level'])}</div>
                <div><strong>Cluster:</strong> {_format_value(user_row['cluster_label'])}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_personal_behavior_metrics(user_row, reference_df):
    st.subheader("Personal Behavior Metrics")

    daily_hours = float(pd.to_numeric(user_row["daily_gaming_hours"], errors="coerce"))
    sleep_hours = float(pd.to_numeric(user_row["sleep_hours"], errors="coerce"))
    exercise_hours = float(pd.to_numeric(user_row["exercise_hours_weekly"], errors="coerce"))
    social_hours = float(pd.to_numeric(user_row["face_to_face_social_hours_weekly"], errors="coerce"))
    spending = float(pd.to_numeric(user_row["monthly_game_spending_usd"], errors="coerce"))

    ref_daily = pd.to_numeric(reference_df["daily_gaming_hours"], errors="coerce").mean()
    ref_sleep = pd.to_numeric(reference_df["sleep_hours"], errors="coerce").mean()
    ref_exercise = pd.to_numeric(reference_df["exercise_hours_weekly"], errors="coerce").mean()
    ref_social = pd.to_numeric(reference_df["face_to_face_social_hours_weekly"], errors="coerce").mean()
    ref_spending = pd.to_numeric(reference_df["monthly_game_spending_usd"], errors="coerce").mean()

    card1, card2, card3, card4, card5 = st.columns(5)
    card1.metric("Daily Gaming Hours", f"{daily_hours:.1f}h", f"{daily_hours - ref_daily:.1f} vs avg")
    card2.metric("Sleep Hours", f"{sleep_hours:.1f}h", f"{sleep_hours - ref_sleep:.1f} vs avg")
    card3.metric("Exercise Weekly", f"{exercise_hours:.1f}h", f"{exercise_hours - ref_exercise:.1f} vs avg")
    card4.metric("Social Hours", f"{social_hours:.1f}h", f"{social_hours - ref_social:.1f} vs avg")
    card5.metric("Monthly Spending", f"${spending:.0f}", f"${spending - ref_spending:.0f} vs avg")


def render_cluster_membership_card(user_row):
    st.subheader("Cluster Membership Card")
    cluster_label = str(user_row.get("cluster_label", "Unavailable"))

    if cluster_label == "Unavailable":
        summary_text = "Cluster membership is unavailable for this filtered view."
    else:
        summary_text = _cluster_risk_segment(cluster_label, str(user_row.get("gaming_addiction_risk_level", "")).title())

    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg, #0f172a, #1f2937);color:white;padding:1.1rem 1.2rem;border-radius:18px;box-shadow:0 14px 30px rgba(15,23,42,0.14);">
            <div style="font-size:0.92rem;opacity:0.8;margin-bottom:0.5rem;">Behavioral segment</div>
            <div style="font-size:1.4rem;font-weight:700;margin-bottom:0.35rem;">{summary_text}</div>
            <div style="font-size:0.95rem;opacity:0.9;">Assigned Cluster: <strong>{cluster_label}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_prediction_card(user_row):
    st.subheader("Risk Prediction Result")

    model, _ = train_global_model()
    prediction_frame = build_prediction_features(user_row)
    prediction, confidence = predict_single_user(model, user_row)
    accent = _risk_color(prediction)

    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg, {accent}, #111827);color:white;padding:1.15rem 1.25rem;border-radius:18px;box-shadow:0 14px 30px rgba(15,23,42,0.14);">
            <div style="font-size:0.92rem;opacity:0.82;margin-bottom:0.45rem;">AI-generated personal classification</div>
            <div style="font-size:1.7rem;font-weight:800;letter-spacing:0.2px;">Prediction: {str(prediction).upper()} RISK</div>
            <div style="margin-top:0.35rem;font-size:0.98rem;opacity:0.92;">Confidence: {confidence:.0%}</div>
            <div style="margin-top:0.65rem;font-size:0.9rem;opacity:0.88;">Green = Low | Yellow = Moderate | Orange = High | Red = Severe</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return model, prediction_frame, prediction


def render_explainable_ai_panel(user_row, model, prediction_frame):
    st.subheader("Explainable AI Panel")
    contributions = build_user_explanation(user_row, model, prediction_frame)

    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg, #111827, #1f2937);color:white;padding:1.1rem 1.2rem;border-radius:18px;box-shadow:0 14px 30px rgba(15,23,42,0.14);">
            <div style="font-size:0.92rem;opacity:0.8;margin-bottom:0.55rem;">Why the user received this prediction</div>
            <div style="font-weight:700;margin-bottom:0.55rem;">Top Risk Drivers:</div>
            <ol style="margin:0;padding-left:1.15rem;line-height:1.75;">
                {''.join(f'<li>{item["risk_text"]}</li>' for item in contributions)}
            </ol>
            <div style="margin-top:0.75rem;font-size:0.82rem;opacity:0.75;">This combines feature importance with the selected user’s values compared against the dataset baseline.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_user_explorer_tab(filtered_df):
    st.header("User Explorer")

    if len(filtered_df) == 0:
        st.warning("No users match the selected filters.")
        return

    explorer_df = _add_cluster_labels(filtered_df)
    filtered_users, selected_user = render_user_search_panel(explorer_df)

    if selected_user is None:
        return

    user_row = filtered_users[filtered_users["record_id"].astype(str) == str(selected_user)].iloc[0]
    render_user_profile_card(user_row)
    render_personal_behavior_metrics(user_row, explorer_df)
    render_mental_health_snapshot(user_row)

    model, prediction_frame, prediction = render_risk_prediction_card(user_row)
    render_cluster_membership_card(user_row)
    render_explainable_ai_panel(user_row, model, prediction_frame)
    render_risk_flags_section(user_row, explorer_df)
    render_personalized_recommendations(user_row, prediction)
