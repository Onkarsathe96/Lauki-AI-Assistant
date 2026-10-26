import os
import streamlit as st
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Page Configuration
st.set_page_config(
    page_title="Lauki AI Assistant",
    page_icon="🥒",
    layout="centered"
)

# Configure Gemini API Key from Environment Variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Load Data, Embeddings, and FAISS Index (Cached for performance)
@st.cache_resource
def load_agent():
    csv_path = "lauki_qna.csv"
    if not os.path.exists(csv_path):
        st.error(f"Dataset {csv_path} not found!")
        return None, None, None, None
        
    df = pd.read_csv(csv_path)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    texts = (df['question'] + " " + df['answer']).tolist()
    embeddings = model.encode(texts, show_progress_bar=False)
    embeddings = np.array(embeddings).astype('float32')
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    llm_model = genai.GenerativeModel('gemini-2.5-flash')
    return df, model, index, llm_model

df, model, index, llm_model = load_agent()

# App Header
st.title("🥒 Lauki (Bottle Gourd) AI Assistant")
st.markdown("Ask anything about lauki, its health benefits, recipes, and wellness tips powered by **Gemini** & **FAISS RAG**!")

if df is not None:
    # Initialize Chat History in Session State
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display Prior Chat Messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Capture User Chat Input
    if prompt := st.chat_input("Ask a question about lauki..."):
        # Append and display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Retrieve relevant context using FAISS
        query_vector = model.encode([prompt]).astype('float32')
        distances, indices = index.search(query_vector, k=2)
        retrieved_context = "\n".join([df.iloc[idx]['answer'] for idx in indices[0]])

        # Construct Prompt for Gemini
        rag_prompt = f"""You are an expert AI assistant specializing in Lauki (bottle gourd). Use the retrieved context below to answer the user's question accurately and conversationally.

Context:
{retrieved_context}

Question: {prompt}
Answer:"""

        # Generate Response from Gemini
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    if not GEMINI_API_KEY:
                        response_text = "Warning: GEMINI_API_KEY environment variable is not set."
                    else:
                        response = llm_model.generate_content(rag_prompt)
                        response_text = response.text
                except Exception as e:
                    response_text = f"Error generating response from Gemini: {str(e)}"
                
                st.markdown(response_text)
                
        # Append assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response_text})