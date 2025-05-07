import streamlit as st
import os
import torch
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
# from dotenv import load_dotenv # No longer needed if hardcoding

# --- 0. Configuration & Constants ---
FAISS_SAVE_PATH = "faiss_index_uploaded_data"  # Make sure this path is correct
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "models/gemini-1.5-flash-latest" # or "gemini-pro" if preferred

# --- 1. Load API Key ---
# !!! WARNING: HARDCODING API KEYS IS A SECURITY RISK !!!
# !!! Only do this for quick, local, temporary testing. !!!
# !!! DO NOT commit this to version control or share it. !!!
GOOGLE_API_KEY = "AIzaSyBkjbOBbc8XpWGNN8lStFCnIbu3MO9vhug" # <--- PASTE YOUR KEY HERE

# Remove or comment out the previous API key loading logic:
# load_dotenv()
# GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")


if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_ACTUAL_GOOGLE_API_KEY_HERE": # Added a check for placeholder
    st.error("🚨 GOOGLE_API_KEY not found or is still the placeholder. Please replace 'YOUR_ACTUAL_GOOGLE_API_KEY_HERE' with your actual key in the code.")
    st.warning("Reminder: Hardcoding API keys is a security risk. Consider using Streamlit secrets or environment variables for better security.")
    st.stop()

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    # print("Google Generative AI SDK configured with API Key.") # For debugging
except Exception as e:
    st.error(f"🚨 Error configuring Google Generative AI SDK: {e}")
    st.error("Please ensure your API key is valid and the Generative Language API is enabled in your Google Cloud Project.")
    st.stop()

# --- 2. Caching Functions for Loading Models and Index ---
# (The rest of your code remains the same)
@st.cache_resource
def load_embeddings_model(model_name, device):
    """Loads the HuggingFace embedding model."""
    st.info(f"Loading embedding model: {model_name} on {device}...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device}
        )
        st.success("Embedding model loaded successfully!")
        return embeddings
    except Exception as e:
        st.error(f"🚨 Error loading embedding model: {e}")
        st.stop()

@st.cache_resource
def load_faiss_index(save_path, _embeddings): # _embeddings to ensure it's passed for cache key
    """Loads the FAISS vector store."""
    st.info(f"Loading FAISS index from: {save_path}...")
    if not os.path.exists(save_path):
        st.error(f"🚨 Error: FAISS index directory '{save_path}' not found.")
        st.error("Please ensure the data vectorization script ran successfully and created the index in the correct location.")
        st.stop()
    try:
        db = FAISS.load_local(
            save_path,
            _embeddings, # Use the passed embeddings object
            allow_dangerous_deserialization=True
        )
        retriever = db.as_retriever(search_kwargs={"k": 3})
        st.success("FAISS index loaded and retriever created!")
        return retriever
    except Exception as e:
        st.error(f"🚨 Error loading FAISS index: {e}")
        st.error("Ensure the embedding model used here matches the one used for creating the index.")
        st.stop()

@st.cache_resource
def load_llm(model_name, api_key):
    """Initializes the ChatGoogleGenerativeAI LLM."""
    st.info(f"Initializing LLM: {model_name}...")
    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.3,
            # convert_system_message_to_human=True # May be needed for some models/prompts
        )
        st.success("LLM initialized successfully!")
        return llm
    except Exception as e:
        st.error(f"🚨 Error initializing LLM: {e}")
        st.error("Please check your API key, model name, Google Cloud Project settings, and internet connection.")
        st.error("Ensure the Generative Language API or Vertex AI API is enabled in your project and billing is active.")
        st.stop()

# --- 3. Define RAG Chain Logic ---
def get_rag_chain(_retriever, _llm):
    """Defines and returns the RAG chain."""
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

# --- 4. Streamlit App UI and Logic ---
st.set_page_config(page_title="📄 RAG Q&A with Gemini", layout="wide")
st.title("📄 Retrieval Augmented Generation (RAG) Q&A")
st.markdown("Ask questions about your vectorized data. The system will retrieve relevant context and use Gemini to generate an answer.")

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    st.sidebar.success(f"Using GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()
else:
    st.sidebar.info("Using CPU. For faster processing, a GPU is recommended.")

# Load models and index
embeddings_model = load_embeddings_model(EMBEDDING_MODEL_NAME, device)
retriever = load_faiss_index(FAISS_SAVE_PATH, embeddings_model)
llm = load_llm(GEMINI_MODEL_NAME, GOOGLE_API_KEY) # GOOGLE_API_KEY is now the hardcoded one

# Get the RAG chain
if retriever and llm:
    rag_chain = get_rag_chain(retriever, llm)
else:
    st.error("🚨 RAG chain could not be initialized due to previous errors.")
    st.stop()


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if query := st.chat_input("What is your question?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)

    # Display assistant response in chat message container
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

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info(
    "This app uses a RAG pipeline with a FAISS vector store, "
    f"HuggingFace embeddings ({EMBEDDING_MODEL_NAME}), "
    f"and Google's Gemini model ({GEMINI_MODEL_NAME}) via LangChain."
)
st.sidebar.markdown(
    "**Important:** Ensure the `faiss_index_uploaded_data` directory "
    " (containing your vectorized data) is in the same directory as this `app.py` file."
)
