# pages/11_ask_ai_tab.py

import streamlit as st
st.set_page_config(page_title="Ask AI - Smart Waves", layout="wide")

import os
import openai
from ask_neo4j import ask_neo4j_logic, initialize_llm_and_embeddings_askai
from langchain_neo4j import Neo4jGraph

# Page config

st.markdown("## 🤖 Ask AI")
st.markdown("Ask anything about financial data, filings, or related entities.")

# Load Neo4j credentials
NEO4J_URI = st.secrets.get("NEO4J_URI") or os.getenv("NEO4J_URI")
NEO4J_USERNAME = st.secrets.get("NEO4J_USER") or os.getenv("NEO4J_USER")
NEO4J_PASSWORD = st.secrets.get("NEO4J_PASSWORD") or os.getenv("NEO4J_PASSWORD")

# Initialize Neo4j connection
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
)

# Tabs for GPT-4 and Gemini
tab1, tab2 = st.tabs(["🧠 GPT-4", "🌟 Gemini"])

def run_tab(provider: str):
    question = st.text_input(f"Ask your question ({provider.upper()}):", key=f"q_{provider}")
    if st.button(f"Send to {provider.upper()}", key=f"b_{provider}"):
        with st.spinner(f"Generating response from {provider.upper()}..."):
            try:
                llm, embeddings, provider_used = initialize_llm_and_embeddings_askai(provider)
                context = ask_neo4j_logic(
                    graph_instance=graph,
                    question_text=question,
                    llm_instance=llm,
                    embeddings_instance=embeddings,
                    llm_provider_name=provider_used,
                    explain_flag=True
                )
                prompt = f"{context}\n\nUser question:\n{question}"

                if provider == "gemini":
                    import google.generativeai as genai
                    gemini = genai.GenerativeModel("gemini-pro")
                    response = gemini.generate_content(prompt)
                    st.markdown(response.text)
                else:
                    client = openai.OpenAI()
                    stream = client.chat.completions.create(
                        model=st.secrets.get("OPENAI_MODEL", "gpt-4o"),
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
                st.error(f"❌ Error with {provider.upper()}: {e}")

with tab1:
    run_tab("openai")

with tab2:
    run_tab("gemini")
