import streamlit as st
import os

import torch
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Constants
FAISS_SAVE_PATH = "faiss_index_uploaded_data"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.5-flash"

# Set page configuration
st.set_page_config(page_title="Chatbot for International relations", layout="wide")

# Hide Streamlit menu and footer
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# Main app title
st.title("Chatbot for International Relations")
st.markdown("Ask questions about the documents mentioned. The system will retrieve relevant context and give you an answer.")

# Secure API key retrieval via Streamlit Secrets
if "GOOGLE_API_KEY" in st.secrets:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
else:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# API key validation
if not GOOGLE_API_KEY:
    st.error("🚨 GOOGLE_API_KEY not found in Streamlit Secrets.")
    st.info("Please go to your Streamlit Cloud dashboard, navigate to App Settings > Secrets, and add your key like this: GOOGLE_API_KEY = 'your_key'")
    st.stop()

try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"🚨 Error configuring Google Generative AI SDK: {e}")
    st.stop()

@st.cache_resource
def load_embeddings_model(model_name, device):
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device}
        )
        return embeddings
    except Exception as e:
        st.error(f"🚨 Error loading embedding model: {e}")
        st.stop()

@st.cache_resource
def load_faiss_index(save_path, _embeddings):
    if not os.path.exists(save_path):
        st.error(f"🚨 Error: FAISS index directory '{save_path}' not found.")
        st.stop()
    try:
        db = FAISS.load_local(
            save_path,
            _embeddings,
            allow_dangerous_deserialization=True
        )
        retriever = db.as_retriever(search_kwargs={"k": 3})
        return retriever
    except Exception as e:
        st.error(f"🚨 Error loading FAISS index: {e}")
        st.stop()

@st.cache_resource
def load_llm(model_name, api_key):
    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.3,
        )
        return llm
    except Exception as e:
        st.error(f"🚨 Error initializing LLM: {e}")
        st.stop()

def get_rag_chain(_retriever, _llm):
    template = """
    You are an assistant for question-answering tasks.
    Use only the following pieces of retrieved context to answer the question.
    If you don't know the answer from the context, just say that you don't know.
    Do not make up an answer. Keep the answer concise.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": _retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | _llm
        | StrOutputParser()
    )
    return rag_chain

# Device detection for GPU/CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    st.sidebar.success(f"Using GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()

# Load models and create RAG chain
embeddings_model = load_embeddings_model(EMBEDDING_MODEL_NAME, device)
retriever = load_faiss_index(FAISS_SAVE_PATH, embeddings_model)
llm = load_llm(GEMINI_MODEL_NAME, GOOGLE_API_KEY)

if retriever and llm:
    rag_chain = get_rag_chain(retriever, llm)
else:
    st.error("🚨 RAG chain could not be initialized due to previous errors.")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input and response
if query := st.chat_input("What is your question?"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            with st.spinner("Thinking... 🤔"):
                answer = rag_chain.invoke(query)
                full_response = answer

            message_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"🚨 An error occurred: {e}")
            full_response = "Sorry, I encountered an error while processing your request."
            message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Sidebar content
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info("You can ask questions about the content in these academic resources: Essentials Of International Relations by Karen A. Mingst, Pax Indica, Does the Elephant Dance?: Contemporary Indian Foreign Policy, Challenge and Strategy: Rethinking India's Foreign Policy, International Relations: A Self-Study Guide")
