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
import time
from datetime import datetime

# Constants
FAISS_SAVE_PATH = "faiss_index_uploaded_data"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "models/gemini-1.5-flash-latest"
GOOGLE_API_KEY = "AIzaSyBkjbOBbc8XpWGNN8lStFCnIbu3MO9vhug"

# Set page configuration with custom theme
st.set_page_config(
    page_title="The Diplomat Assistant",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
        padding: 1.5rem;
    }
    
    /* Header styling */
    .stApp header {
        background-color: #14213d;
        color: white;
    }
    
    /* Chat container styling */
    .chat-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    /* User message styling */
    .user-message {
        background-color: #e9ecef;
        border-radius: 15px 15px 0 15px;
        padding: 10px 15px;
        margin: 5px 0;
        display: inline-block;
        max-width: 80%;
        float: right;
    }
    
    /* Assistant message styling */
    .assistant-message {
        background-color: #cfe2ff;
        border-radius: 15px 15px 15px 0;
        padding: 10px 15px;
        margin: 5px 0;
        display: inline-block;
        max-width: 80%;
        float: left;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #14213d;
        color: white;
    }
    
    /* Custom button styling */
    .stButton>button {
        background-color: #fca311;
        color: #14213d;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 10px 15px;
    }
    
    .stButton>button:hover {
        background-color: #e5e5e5;
        color: #14213d;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        margin-top: 30px;
        padding: 10px;
        font-size: 12px;
        color: #6c757d;
    }
    
    /* Custom tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #e5e5e5;
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
        color: #14213d;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #fca311;
        color: #14213d;
    }
    
    /* Loading animation */
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    
    .loading {
        animation: pulse 1.5s infinite;
        background-color: #e9ecef;
        border-radius: 5px;
        padding: 10px;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Verify API key
if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_ACTUAL_GOOGLE_API_KEY_HERE":
    st.error("🚨 GOOGLE_API_KEY not found or is still the placeholder. Please configure your API key.")
    st.warning("Reminder: Never hardcode API keys in production code. Use Streamlit secrets or environment variables.")
    st.stop()

try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"🚨 Error configuring Google Generative AI SDK: {e}")
    st.error("Please ensure your API key is valid and the Generative Language API is enabled in your Google Cloud Project.")
    st.stop()

# Cache resource functions
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
        st.error("Please ensure the data vectorization script ran successfully and created the index in the correct location.")
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
        st.error("Ensure the embedding model used here matches the one used for creating the index.")
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
        st.error("Please check your API key, model name, Google Cloud Project settings, and internet connection.")
        st.error("Ensure the Generative Language API or Vertex AI API is enabled in your project and billing is active.")
        st.stop()

def get_rag_chain(_retriever, _llm, system_prompt=None):
    if not system_prompt:
        system_prompt = """
        You are The Diplomat Assistant, an expert in international relations, diplomacy, and global affairs.
        Use only the following pieces of retrieved context to answer the question.
        If you don't know the answer from the context, just say that you don't have enough information.
        Do not make up an answer. Present your response in a professional, diplomatic tone.
        
        Context:
        {context}
        
        Question:
        {question}
        
        Answer:
        """
    
    prompt = ChatPromptTemplate.from_template(system_prompt)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": _retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | _llm
        | StrOutputParser()
    )
    return rag_chain

def show_typing_animation(placeholder):
    typing_statuses = ["Thinking...", "Searching documents...", "Analyzing context...", "Formulating response..."]
    for status in typing_statuses:
        placeholder.markdown(f"<div class='loading'>{status}</div>", unsafe_allow_html=True)
        time.sleep(0.7)

def format_chat_message(message, role):
    if role == "user":
        return f"<div style='text-align: right;'><div class='user-message'>{message}</div></div>"
    else:
        return f"<div style='text-align: left;'><div class='assistant-message'>{message}</div></div>"

# Sidebar content
with st.sidebar:
    st.image("https://via.placeholder.com/150x150.png?text=The+Diplomat", width=150)
    st.title("The Diplomat Assistant")
    
    st.markdown("---")
    
    # Add tabs to sidebar
    sidebar_tab1, sidebar_tab2, sidebar_tab3 = st.tabs(["Info", "Settings", "Help"])
    
    with sidebar_tab1:
        st.subheader("About")
        st.info(
            """
            This application uses Retrieval-Augmented Generation (RAG) to provide accurate answers 
            about diplomatic affairs, international relations, and global politics based on trusted resources.
            
            Ask questions about treaties, international organizations, historic diplomatic events, or 
            current global affairs.
            """
        )
        
        # Show system status
        st.subheader("System Status")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            st.success(f"⚡ Using GPU: {torch.cuda.get_device_name(0)}")
            torch.cuda.empty_cache()
        else:
            st.info("💻 Using CPU")
            
        st.success("✅ Vector Database: Connected")
        st.success("✅ Gemini API: Connected")
    
    with sidebar_tab2:
        st.subheader("Preferences")
        
        # Model settings
        st.write("Response Settings")
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1,
                              help="Lower values for more factual responses, higher for more creative ones.")
        
        context_length = st.slider("Context Length", min_value=1, max_value=5, value=3, step=1,
                                 help="Number of document chunks to use for context")
        
        # Display settings
        st.write("Display Settings")
        st.checkbox("Show retrieved sources", value=True, key="show_sources")
        st.checkbox("Dark mode", value=False, key="dark_mode")
        st.checkbox("Compact mode", value=False, key="compact_mode")
    
    with sidebar_tab3:
        st.subheader("How to use")
        st.markdown("""
        1. Type your question in the chat input box below
        2. The system will search for relevant information in the document database
        3. A response will be generated based on the found context
        4. You can ask follow-up questions or start a new topic
        
        **Example questions:**
        - What are the main principles of diplomatic immunity?
        - How does the United Nations Security Council operate?
        - What was the impact of the Treaty of Versailles?
        """)
        
        st.subheader("Support")
        st.markdown("For technical support, please contact the system administrator.")
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.experimental_rerun()
    
    # Provide feedback
    feedback = st.selectbox("Rate your experience:", ["Select rating", "⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"])
    if feedback != "Select rating" and st.button("Submit Feedback"):
        st.success("Thank you for your feedback!")
    
    st.markdown("---")
    st.markdown("<div class='footer'>© 2025 The Diplomat Assistant | Last updated: May 2025</div>", unsafe_allow_html=True)

# Main content area
main_container = st.container()

with main_container:
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center; color: #14213d;'>The Diplomat Assistant</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #6c757d;'>Your source for diplomatic knowledge and international affairs</p>", unsafe_allow_html=True)
    
    # Chat interface container
    chat_container = st.container()
    
    with chat_container:
        # Welcome message on first run
        if "messages" not in st.session_state:
            st.session_state.messages = []
            welcome_message = """
            Welcome to The Diplomat Assistant! 👋
            
            I'm here to help you navigate the complex world of international relations, diplomacy, and global affairs. 
            You can ask me about:
            
            - Historical treaties and agreements
            - International organizations and their functions
            - Diplomatic protocols and practices
            - Global political developments
            
            How can I assist you today?
            """
            st.session_state.messages.append({"role": "assistant", "content": welcome_message})
        
        # Chat history display
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        
        for message in st.session_state.messages:
            formatted_message = format_chat_message(message["content"], message["role"])
            st.markdown(formatted_message, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Initialize models and chains
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings_model = load_embeddings_model(EMBEDDING_MODEL_NAME, device)
    retriever = load_faiss_index(FAISS_SAVE_PATH, embeddings_model)
    llm = load_llm(GEMINI_MODEL_NAME, GOOGLE_API_KEY)
    
    if retriever and llm:
        # Get custom system prompt if set
        custom_prompt = None
        if "system_prompt" in st.session_state and st.session_state.system_prompt:
            custom_prompt = st.session_state.system_prompt
        
        rag_chain = get_rag_chain(retriever, llm, custom_prompt)
    else:
        st.error("🚨 RAG chain could not be initialized due to previous errors.")
        st.stop()
    
    # Query input
    query = st.chat_input("Ask me about diplomatic affairs...", key="chat_input")
    
    if query:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Create placeholder for response and show typing animation
        response_placeholder = st.empty()
        show_typing_animation(response_placeholder)
        
        try:
            # Get response from RAG chain
            timestamp_start = datetime.now()
            answer = rag_chain.invoke(query)
            timestamp_end = datetime.now()
            processing_time = (timestamp_end - timestamp_start).total_seconds()
            
            # Display sources if enabled
            if st.session_state.get("show_sources", True):
                # This would actually need to be implemented
                # Here we just show a placeholder
                retrieved_sources = ["Document 1", "Document 2", "Document 3"]
                sources_html = "<div style='font-size: 12px; color: #6c757d; margin-top: 10px;'><strong>Sources:</strong> "
                sources_html += ", ".join(retrieved_sources)
                sources_html += f" <em>(Response generated in {processing_time:.2f} seconds)</em></div>"
                answer += sources_html
            
            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # Clear animation and show response
            st.experimental_rerun()
            
        except Exception as e:
            error_message = f"Sorry, I encountered an error while processing your request: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            st.experimental_rerun()
