# C:\app\gemini\test_link.py
import streamlit as st

st.set_page_config(page_title="Test Link")

st.write(f"Streamlit version: {st.__version__}")

st.header("Test st.page_link with query_params")

st.page_link("app.py", label="Go to Login", icon="🔑", query_params={"action": "login"})
st.header("Query Params Received:")
st.write(st.query_params)

# For older Streamlit versions that used experimental_get_query_params
# if hasattr(st, 'experimental_get_query_params'):
#     st.write("Experimental Query Params:", st.experimental_get_query_params())