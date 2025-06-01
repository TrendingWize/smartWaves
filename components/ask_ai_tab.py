# components/ask_ai_tab.py
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import traceback

from langchain_neo4j import Neo4jGraph  # ✅ Correct location
from ask_neo4j import (
    initialize_llm_and_embeddings_askai,
    ask_neo4j_logic,
    NEO4J_URI_ASKAI,
    NEO4J_USERNAME_ASKAI,
    NEO4J_PASSWORD_ASKAI
)
from utils.openai_helpers import ask_ai
from utils import get_neo4j_driver
# --- Load Neo4j Credentials ---
NEO4J_URI = st.secrets.get("NEO4J_URI") or os.getenv("NEO4J_URI")
NEO4J_USER = st.secrets.get("NEO4J_USER") or os.getenv("NEO4J_USER")
NEO4J_PASSWORD = st.secrets.get("NEO4J_PASSWORD") or os.getenv("NEO4J_PASSWORD")

if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
    st.error("❌ Missing Neo4j credentials. Please check your `.streamlit/secrets.toml` or environment variables.")
    st.stop()

# --- Initialize Neo4j Graph ---
try:
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD
    )
except Exception as e:
    st.error(f"Failed to connect to Neo4j: {e}")
    st.stop()

# Initialize LLM and vector store (must return `llm`, `vectorstore`, etc.)
llm, vectorstore = initialize_llm_and_embeddings_askai('openai')


def ask_ai_tab_content():
    # --- Centering the input section ---
    st.markdown("""
        <style>
        .main-ask-ai-container {
            display: flex;
            flex-direction: column;
            align-items: center; /* Horizontally center */
            padding-top: 2rem; 
        }
        .ask-ai-input-area {
            width: 70%; 
            max-width: 900px;
        }
        .ask-ai-button-container {
            width: 70%;
            max-width: 900px;
            display: flex;
            justify-content: center; 
            margin-top: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- AI Provider Selection ---
    # For simplicity, hardcoding. Make this a st.selectbox or st.radio if needed.
    # Ensure the selected provider matches what initialize_llm_and_embeddings_askai expects.
    llm_provider = "openai" 
    # llm_provider = st.radio("Select AI Provider:", ("openai", "gemini"), horizontal=True, key="ask_ai_provider_select_tab")


    # --- Main Input Container (Centered) ---
    st.markdown("<div class='main-ask-ai-container'>", unsafe_allow_html=True)
    
    st.markdown("<div class='ask-ai-input-area'>", unsafe_allow_html=True)
    question = st.text_area(
        "Ask anything about the financial data...",
        height=100, 
        key="ask_ai_question_input_tab", # Unique key for this instance
        placeholder="e.g., What was Apple's revenue in 2022? or Companies similar to Nvidia in semiconductors?"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='ask-ai-button-container'>", unsafe_allow_html=True)
    ask_button = st.button("🤖 Ask AI", key="ask_ai_submit_button_tab", type="primary", use_container_width=True) # Unique key
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True) 
    
    st.markdown("---") 

    # --- Session State for Results ---
    if 'ask_ai_response_tab' not in st.session_state: # Using suffix _tab for uniqueness
        st.session_state.ask_ai_response_tab = None
    if 'ask_ai_cypher_tab' not in st.session_state:
        st.session_state.ask_ai_cypher_tab = None
    if 'ask_ai_params_tab' not in st.session_state:
        st.session_state.ask_ai_params_tab = None


    if ask_button and question:
        # Clear previous results for this tab
        st.session_state.ask_ai_response_tab = None 
        st.session_state.ask_ai_cypher_tab = None
        st.session_state.ask_ai_params_tab = None

        with st.spinner("Thinking... 🧠 Please wait."): # Added "Please wait"
            try:
                # 1. Initialize LLM and Embeddings (from ask_neo4j.py)
                llm, embeddings, llm_provider_name_used = initialize_llm_and_embeddings_askai(llm_provider)
                
                # 2. Initialize Neo4jGraph instance
                # This was the source of the TypeError. Neo4jGraph takes url, user, pass.
                graph = Neo4jGraph(
                    url=NEO4J_URI,
                    username=NEO4J_USER,
                    password=NEO4J_PASSWORD,
                    database="neo4j" # Specify your database if not default
                )
                # Optional: A quick test to ensure graph connection works before LLM calls
                try:
                    graph.query("RETURN 1 AS test")
                except Exception as graph_e:
                    st.error(f"Failed to connect to Neo4j with Neo4jGraph: {graph_e}")
                    st.stop() # Stop execution if graph connection fails

                # 3. Call the refactored logic from ask_neo4j.py
                cypher_query, params_str, final_answer, raw_result = ask_neo4j_logic(
                    graph_instance=graph,       # Pass the Neo4jGraph instance
                    question_text=question,
                    llm_instance=llm,
                    embeddings_instance=embeddings,
                    llm_provider_name=llm_provider_name_used, 
                    explain_flag=True
                )
                st.session_state.ask_ai_cypher_tab = cypher_query
                st.session_state.ask_ai_params_tab = params_str
                st.session_state.ask_ai_response_tab = final_answer

            except ValueError as ve: # Catch API key errors or other ValueErrors from init
                st.error(str(ve))
                st.session_state.ask_ai_response_tab = f"Initialization Error: {str(ve)}"
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                traceback.print_exc() 
                st.session_state.ask_ai_response_tab = f"An unexpected error occurred. Please check server logs. Error: {str(e)}"
        st.rerun() # Rerun to display new results or clear old ones if an error occurred mid-process

    # Display the results from session state
    if st.session_state.ask_ai_response_tab:
        st.subheader("💡 AI Response:")
        st.markdown(st.session_state.ask_ai_response_tab, unsafe_allow_html=True) # Allow HTML if answer has Markdown tables etc.

        # Only show Cypher if it was successfully generated and not an error message
        if st.session_state.ask_ai_cypher_tab and \
           st.session_state.ask_ai_cypher_tab not in ["No Cypher generated.", "Prompt Formatting Error", 
                                                     "LLM JSON Parsing Error", "Invalid Cypher from LLM"]:
            with st.expander("Show Generated Cypher Query & Parameters"):
                st.code(st.session_state.ask_ai_cypher_tab, language="cypher")
                if st.session_state.ask_ai_params_tab and st.session_state.ask_ai_params_tab != "{}":
                    try: # Try to pretty print JSON if params_str is valid JSON
                        params_dict = json.loads(st.session_state.ask_ai_params_tab)
                        st.json(params_dict)
                    except json.JSONDecodeError:
                        st.code(st.session_state.ask_ai_params_tab, language="json") # Fallback to code block
    elif ask_button and not question: # If ask was clicked but question was empty
        st.warning("Please enter a question to ask the AI.")


# --- Standalone run mock (ensure ask_neo4j.py mocks are also robust) ---
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Ask AI Tab Test")

    # Mock functions from ask_neo4j.py
    class MockLLMStandalone:
        def invoke(self, prompt_str_arg): # Renamed arg to avoid conflict
            class Content: 
                # If it's a Cypher generation prompt, return JSON-like string
                content = ('{"cypher": "MATCH (c:Company) WHERE c.name CONTAINS $company RETURN c.name, c.revenue LIMIT 1", "params": {"company":"TestCorp"}}' 
                           if "Generate a Cypher query" in prompt_str_arg 
                           else "This is a **mocked final answer** from the LLM for your question about TestCorp.")
            return Content()

    class MockEmbeddingsStandalone:
        def embed_query(self, text_arg): return [0.1] * 768 # Dummy embedding, ensure dim matches if used

    @st.cache_resource # Mocking the cached function
    def initialize_llm_and_embeddings_askai(provider_arg): # Renamed arg
        print(f"Mock: Initializing AI for provider: {provider_arg}")
        return MockLLMStandalone(), MockEmbeddingsStandalone(), provider_arg

    class MockNeo4jGraphStandalone:
        def query(self, cypher_str_arg, params=None): # Renamed arg
            print(f"MockNeo4jGraph Query: {cypher_str_arg} with params: {params}")
            if "TestCorp" in params.get("company", ""):
                return [{"c.name": "TestCorp", "c.revenue": 1000000}]
            return [{"message": "Mock Neo4j DB Result"}]
        def get_schema(self): # Mock schema
            return "Node Company {name:STRING, revenue:INTEGER} Relationship WORKS_AT {since:INTEGER}"
    
    # This is the main function from ask_neo4j.py that needs to be mocked
    # if you are not importing the actual ask_neo4j.py (e.g. if it has complex deps)
    def ask_neo4j_logic(graph_instance, question_text, llm_instance, 
                        embeddings_instance, llm_provider_name, explain_flag):
        print(f"Mock: ask_neo4j_logic called for: {question_text}")
        # Simulate the tuple it returns
        mock_cypher = "MATCH (c:Company {name: 'MockCo'}) RETURN c.revenue"
        mock_params_str = '{"company_name": "MockCo"}'
        mock_final_answer = f"The AI analyzed your question: *'{question_text}'*. For MockCo, the revenue is **$1.2M** (mocked)."
        mock_raw_result = [{"c.revenue": 1200000}]
        return mock_cypher, mock_params_str, mock_final_answer, mock_raw_result

    # If you have `from streamlit_financial_dashboard.utils import get_neo4j_driver` and utils.py isn't fully mocked for standalone,
    # you might need to mock get_neo4j_driver too.
    # For this test, ask_ai_tab.py creates Neo4jGraph directly with hardcoded/env creds.
    
    ask_ai_tab_content()