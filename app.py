import streamlit as st
from pathlib import Path


st.set_page_config(page_title="Gaming Dashboards", page_icon="🎮", layout="wide")

if "dashboard_view" not in st.session_state:
    st.session_state.dashboard_view = "admin"

st.sidebar.header("Dashboard View")
st.sidebar.radio(
    "Choose a dashboard",
    ["Admin Dashboard", "User Dashboard"],
    index=0 if st.session_state.dashboard_view == "admin" else 1,
    key="dashboard_choice",
)
st.session_state.dashboard_view = "admin" if st.session_state.dashboard_choice == "Admin Dashboard" else "user"

st.title("Gaming Habit Dashboards")
st.caption("This is the admin dashboard. You can switch to the user dashboard from the sidebar.")


def render_script(script_name):
    script_path = Path(__file__).with_name(script_name)
    script_globals = {
        "__name__": "__main__",
        "__file__": str(script_path),
    }
    exec(compile(script_path.read_text(encoding="utf-8"), str(script_path), "exec"), script_globals)


if st.session_state.dashboard_view == "admin":
    render_script("test.py")
else:
    render_script("user_dashboard.py")
