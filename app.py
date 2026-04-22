import streamlit as st
from pathlib import Path


st.set_page_config(page_title="Gaming Dashboards", page_icon="🎮", layout="wide")

if "dashboard_view" not in st.session_state:
    st.session_state.dashboard_view = "admin"


st.markdown(
    """
    <style>
    .launcher {
        max-width: 900px;
        margin: 4rem auto 0 auto;
        padding: 2rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #0f172a, #243b55);
        color: white;
        box-shadow: 0 18px 48px rgba(15, 23, 42, 0.25);
    }
    .launcher h1 {
        margin: 0 0 0.5rem 0;
        font-size: 2rem;
    }
    .launcher p {
        margin: 0;
        opacity: 0.92;
        line-height: 1.6;
    }
    </style>
    <div class="launcher">
        <h1>Gaming Habit Dashboards</h1>
        <p>Use the icons in the upper-right corner to switch between the admin dashboard and the user survey dashboard.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


top_left, admin_col, user_col = st.columns([8, 1, 1])

with admin_col:
    if st.button("🛠️", help="Open admin dashboard", use_container_width=True):
        st.session_state.dashboard_view = "admin"
        st.rerun()

with user_col:
    if st.button("👤", help="Open user dashboard", use_container_width=True):
        st.session_state.dashboard_view = "user"
        st.rerun()


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
