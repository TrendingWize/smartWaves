
import os
from neo4j import GraphDatabase
import streamlit as st

@st.cache_resource
def get_neo4j_driver():
    """Return a cached Neo4j driver using credentials from env vars or Streamlit secrets."""
    # Order of precedence: Streamlit secrets -> Environment -> None (error)
    if st.secrets.get("neo4j"):
        creds = st.secrets["neo4j"]
        uri = creds.get("uri")
        user = creds.get("user")
        password = creds.get("password")
    else:
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
    if not all([uri, user, password]):
        raise ValueError("Neo4j credentials not configured. Please set Streamlit secrets or environment variables.")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    return driver
