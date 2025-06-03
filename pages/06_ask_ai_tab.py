# pages/11_ask_ai_tab.py

import streamlit as st
st.set_page_config(page_title="Ask AI - Smart Waves", layout="wide")


import os
import openai
from langchain_neo4j import Neo4jGraph
from ask_neo4j import ask_neo4j_logic, initialize_llm_and_embeddings_askai

# Page config
st.markdown("## 🤖 Ask AI (Dual Analysis)")
st.markdown("Ask one question and get responses from both GPT-4 and Gemini.")

# Load credentials
NEO4J_URI = st.secrets.get("NEO4J_URI") or os.getenv("NEO4J_URI")
NEO4J_USERNAME = st.secrets.get("NEO4J_USER") or os.getenv("NEO4J_USER")
NEO4J_PASSWORD = st.secrets.get("NEO4J_PASSWORD") or os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Connect to Neo4j
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
)

# UI: Single input for both models
question = st.text_input("Ask your financial question:", placeholder="e.g., What are the top risks facing Apple in 2024?")
submitted = st.button("🔍 Analyze with GPT-4 & Gemini")

if submitted and question:
    with st.spinner("🔄 Fetching context from Neo4j..."):
        try:
            # Initialize for GPT-4
            llm_gpt, embeddings_gpt, _ = initialize_llm_and_embeddings_askai("openai")
            context_gpt = ask_neo4j_logic(graph, question, llm_gpt, embeddings_gpt, "openai", explain_flag=True)

            # Initialize for Gemini
            llm_gemini, embeddings_gemini, _ = initialize_llm_and_embeddings_askai("gemini")
            context_gemini = ask_neo4j_logic(graph, question, llm_gemini, embeddings_gemini, "gemini", explain_flag=True)

        except Exception as e:
            st.error(f"❌ Failed to generate context: {e}")
            st.stop()

    # Columns for parallel output
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧠 GPT-4 Response")
        try:
            client = openai.OpenAI()
            stream = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": f"{context_gpt}\n\nUser question:\n{question}"}],
                stream=True,
                temperature=0.3
            )
            gpt_box = st.empty()
            gpt_answer, buffer, counter = "", "", 0
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                buffer += delta
                counter += 1
                if counter % 5 == 0:
                    gpt_answer += buffer
                    gpt_box.markdown(gpt_answer)
                    buffer = ""
            gpt_answer += buffer
            gpt_box.markdown(gpt_answer)
        except Exception as e:
            st.error(f"❌ GPT-4 Error: {e}")

    with col2:
        st.subheader("🌟 Gemini Response")
        try:
            import google.generativeai as genai
            gemini = genai.GenerativeModel("gemini-pro")
            response = gemini.generate_content(f"{context_gemini}\n\nUser question:\n{question}")
            st.markdown(response.text)
        except Exception as e:
            st.error(f"❌ Gemini Error: {e}")
