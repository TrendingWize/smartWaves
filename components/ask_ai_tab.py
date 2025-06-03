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
    st.markdown("## 🤖 Ask AI")

    tab1, tab2 = st.tabs(["🧠 o3", "🌟 Gemini"])

    with tab1:
        run_ask_ai_ui("openai")

    with tab2:
        run_ask_ai_ui("gemini")


def run_ask_ai_ui(provider: str):
    st.markdown(f"### Chat with {provider.upper()}")

    st.markdown("<div class='main-ask-ai-container'>", unsafe_allow_html=True)
    
    st.markdown("<div class='ask-ai-input-area'>", unsafe_allow_html=True)
    question = st.text_area(
        "Ask anything about the financial data...",
        height=100,
        key=f"{provider}_ask_input",
        placeholder="e.g., What was Apple's revenue in 2022?"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='ask-ai-button-container'>", unsafe_allow_html=True)
    ask_button = st.button("🤖 Ask AI", key=f"{provider}_ask_button", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

    if ask_button and question:
        with st.spinner(f"Sending your question to {provider.upper()}..."):
            try:
                llm, embeddings, llm_provider = initialize_llm_and_embeddings_askai(provider)
                graph = Neo4jGraph(
                    url=NEO4J_URI,
                    username=NEO4J_USER,
                    password=NEO4J_PASSWORD
                )
                context = ask_neo4j_logic(
                    graph_instance=graph,
                    question_text=question,
                    llm_instance=llm,
                    embeddings_instance=embeddings,
                    llm_provider_name=llm_provider,
                    explain_flag=True
                )

                prompt = f"{context}\n\nUser question:\n{question}"

                if provider == "gemini":
                    import google.generativeai as genai
                    gemini = genai.GenerativeModel('gemini-pro')
                    response = gemini.generate_content(prompt)
                    st.markdown(response.text)
                else:
                    import openai
                    client = openai.OpenAI()
                    stream = client.chat.completions.create(
                        model="o3",
                        messages=[{"role": "user", "content": prompt}],
                        stream=True,
                        temperature=0.3
                    )
                    response_box = st.empty()
                    full_response, buffer, counter = "", "", 0
                    for chunk in stream:
                        delta = chunk.choices[0].delta.content or ""
                        buffer += delta
                        counter += 1
                        if counter % 5 == 0:
                            full_response += buffer
                            response_box.markdown(f"🤖 **AI:** {full_response}", unsafe_allow_html=True)
                            buffer = ""
                    full_response += buffer
                    response_box.markdown(f"🤖 **AI:** {full_response}", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Error using {provider.upper()}: {e}")


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
