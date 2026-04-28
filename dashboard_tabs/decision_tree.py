import streamlit as st

from charts import binary_decision_tree


def render_decision_tree_tab(filtered_df):
    st.header("Binary Decision Tree")

    if "tree_mode" not in st.session_state:
        st.session_state.tree_mode = "entropy"

    if st.button("Entropy / Gini"):
        st.session_state.tree_mode = "gini" if st.session_state.tree_mode == "entropy" else "entropy"

    tree_depth = st.number_input("Max Depth", min_value=1, max_value=10, value=3, step=1)
    binary_decision_tree(filtered_df, mode=st.session_state.tree_mode, tree_max_depth=int(tree_depth))
