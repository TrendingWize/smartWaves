# pages/11_ask_ai_tab.py

import streamlit as st
st.set_page_config(page_title="Ask AI - Smart Waves", layout="wide")

import openai
import os
from ask_neo4j import ask_neo4j_logic
from langchain_neo4j import Neo4jGraph
from ask_neo4j import (
    initialize_llm_and_embeddings_askai,
    ask_neo4j_logic,
    NEO4J_URI_ASKAI,
    NEO4J_USERNAME_ASKAI,
    NEO4J_PASSWORD_ASKAI
)

# --- Configure Streamlit page ---
st.markdown("## 🤖 Ask AI")
st.markdown("Ask anything about financial data, filings, or related entities.")

# Load credentials
NEO4J_URI = st.secrets.get("NEO4J_URI") or os.getenv("NEO4J_URI")
NEO4J_USERNAME = st.secrets.get("NEO4J_USER") or os.getenv("NEO4J_USER")
NEO4J_PASSWORD = st.secrets.get("NEO4J_PASSWORD") or os.getenv("NEO4J_PASSWORD")
OPENAI_MODEL = st.secrets.get("OPENAI_MODEL", "gpt-4o")  # Default model

# ✅ Initialize graph
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
)

# Initialize LLM and vector store
llm_provider = 'openai'
llm, embeddings, llm_provider = initialize_llm_and_embeddings_askai(llm_provider)
client = openai.OpenAI()

# --- Chat UI Layout ---
st.markdown("### 💬 Chat")

with st.form("chat_form"):
    question = st.text_input("Ask your question:")
    submitted = st.form_submit_button("Send")

if submitted and question:
    try:
        # Step 1: Get context from Neo4j
        context = ask_neo4j_logic(
            graph_instance=graph,
            question_text=question,
            llm_instance=llm,
            embeddings_instance=embeddings,
            llm_provider_name=llm_provider,
            explain_flag=True
        )

        # Step 2: Compose single-turn message
        full_prompt = f"{context}\n\nUser question:\n{question}"

        # Step 3: Stream OpenAI response
        stream = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            stream=True,
            temperature=0.3
        )

        # Step 4: Stream output
        response_box = st.empty()
        full_response = ""
        buffer = ""
        counter = 0
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
        st.error(f"⚠️ Error: {e}")
