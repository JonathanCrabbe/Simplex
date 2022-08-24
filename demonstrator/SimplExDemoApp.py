import streamlit as st

st.set_page_config(
    page_title="Welcome",
    page_icon="👋",
)

st.write("# Welcome to the SimplEx demonstrator!")

st.sidebar.success("Select a view above.")

st.markdown(
    """
    SimplEx is an explainability model to run alongside any black box model.
    When reviewing why a black box model has given a certain prediction for
    a given input, it can both show you how it would treat similar examples
    and say which features in those examples it is paying most attention to.

    **👈 Select a data/model type from the sidebar** to see some examples
    of what SimplEx can do!
    ### Want to learn more?
    Check out the [paper](https://arxiv.org/pdf/2110.15355.pdf).

"""
)
