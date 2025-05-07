import streamlit as st
import os
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch # Keep this import
# Make sure you have these libraries installed:
# pip install streamlit langchain langchain-community faiss-cpu transformers torch accelerate bitsandbytes pynvml

st.set_page_config(
    page_title="IR Knowledge Hub",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# Specify the path to the FAISS index directory as a constant or get it via text_input
FAISS_INDEX_PATH = "faiss_index_uploaded_data" # Default path

st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #6200EA;        /* Deep purple */
        --secondary-color: #00BFA5;      /* Teal accent */
        --background-color: #FFFFFF;     /* Changed to White for clarity against custom elements */
        --text-color: #212121;           /* Dark grey */
        --light-accent: #E1F5FE;         /* Light blue accent */
        --dark-accent: #311B92;          /* Darker purple */
        --success-color: #00C853;        /* Green success */
        --warning-color: #FFD600;        /* Yellow warning */
        --error-color: #D50000;          /* Red error */
    }

    /* General container styling */
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
    }

    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1200px;
        margin: 0 auto;
    }

    /* Custom card styling */
    .custom-card {
        background-color: #FFFFFF; /* White background */
        border-radius: 16px;
        padding: 25px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        margin-bottom: 24px;
        border-top: 5px solid var(--primary-color);
        color: var(--text-color); /* Ensure text is readable */
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #F5F5F5; /* Light grey sidebar */
        box-shadow: 2px 0 5px rgba(0, 0, 0, 0.05);
    }
     [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
         color: var(--primary-color) !important; /* Ensure sidebar headers are visible */
     }

    /* Header styling with gradient text */
    h1 {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        font-size: 2.8rem;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }

    h2 {
        font-family: 'Poppins', sans-serif;
        color: var(--primary-color);
        font-weight: 600;
        font-size: 1.8rem;
        margin-top: 0.5rem;
    }

    h3 {
        font-family: 'Poppins', sans-serif;
        color: var(--text-color);
        font-weight: 600;
        font-size: 1.4rem;
        margin-top: 1rem;
    }

    /* Button styling */
    button, .stButton > button {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color)) !important;
        color: white !important; /* Changed button text to white */
        border: none !important;
        border-radius: 50px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.3s ease !important;
    }

    button:hover, .stButton > button:hover {
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.2) !important;
        transform: translateY(-2px) !important;
    }

    /* Input field styling */
    .stTextInput input, .stTextArea textarea {
        border-radius: 10px !important;
        border: 2px solid #E0E0E0 !important;
        padding: 12px !important;
        transition: all 0.3s ease;
    }

    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: var(--secondary-color) !important;
        box-shadow: 0 0 0 2px rgba(0, 191, 165, 0.2) !important;
    }

    /* Answer box styling */
    .answer-box {
        background-color: #F5F5F5; /* Light grey background for answer */
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        margin: 20px 0;
        border-left: 5px solid var(--secondary-color);
        line-height: 1.7;
        font-size: 1.05rem;
        color: var(--text-color); /* Ensure text is readable */
    }

    /* Status indicators */
    .status-indicator {
        padding: 8px 16px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
        margin-bottom: 10px;
    }

    .status-ready {
        background-color: rgba(0, 200, 83, 0.1);
        color: var(--success-color);
    }

    .status-waiting {
        background-color: rgba(255, 214, 0, 0.1);
        color: #FF6D00;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        color: var(--primary-color) !important;
        background-color: rgba(98, 0, 234, 0.05) !important;
        border-radius: 8px !important;
        padding: 10px 20px !important; /* Added padding */
    }

    .streamlit-expanderContent {
        border: none !important;
        border-top: none !important;
        background-color: #FFFFFF !important; /* White background */
        border-radius: 0 0 8px 8px !important;
        padding: 20px !important;
        color: var(--text-color); /* Ensure text is readable */
    }

    /* Spinner styling */
    .stSpinner > div > div {
        border-color: var(--primary-color) transparent transparent !important;
    }

    /* Progress bar styling */
    .stProgress > div > div > div {
        background-color: var(--secondary-color) !important;
    }

    /* Info, success, warning, error boxes */
    .stInfo, .stSuccess, .stWarning, .stError {
        border-radius: 12px !important;
        padding: 20px !important;
    }

    /* Sidebar header - Adjusted target selector */
    [data-testid="stSidebar"] h1 {
        font-size: 1.5rem !important;
        background: none !important;
        -webkit-background-clip: unset !important;
        background-clip: unset !important;
        color: var(--primary-color) !important; /* Use primary color for sidebar header */
    }

    /* Footer styling */
    .footer {
        background-color: #F5F5F5; /* Light grey footer */
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 30px;
        border-top: 3px solid var(--light-accent);
        color: var(--text-color); /* Ensure text is readable */
    }

    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: var(--text-color);
        color: #fff; /* White text on dark background */
        text-align: center;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }

    /* Custom tabs if needed */
    [data-baseweb="tab-list"] {
        background-color: #E0E0E0; /* Light grey tabs background */
        border-radius: 10px;
        padding: 5px;
    }

    [data-baseweb="tab"] {
        border-radius: 5px;
        padding: 10px 20px;
        margin-right: 5px;
        color: var(--text-color) !important; /* Ensure tab text is readable */
    }

    [data-baseweb="tab"][aria-selected="true"] {
        background-color: var(--primary-color) !important;
        color: white !important; /* White text on selected tab */
    }
</style>
""", unsafe_allow_html=True)

# Load custom fonts
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# --- Cached Functions for Loading Resources ---

@st.cache_resource
def load_embedding_model(model_name):
    """Loads the embedding model using st.cache_resource."""
    try:
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        st.success("✅ Embedding model loaded successfully.")
        return embeddings
    except Exception as e:
        st.error(f"❌ Error loading embedding model: {e}")
        st.exception(e)
        return None

@st.cache_resource
def load_faiss_database(path, embeddings):
    """Loads the FAISS database using st.cache_resource."""
    if embeddings is None:
        st.error("Embedding model not loaded, cannot load FAISS index.")
        return None
    try:
        if not os.path.exists(path):
             st.error(f"❌ FAISS index directory not found at: {path}")
             return None
        # Check for required files
        if not os.path.exists(os.path.join(path, "index.faiss")) or not os.path.exists(os.path.join(path, "index.pkl")):
             st.error(f"❌ FAISS index files (index.faiss, index.pkl) not found in: {path}")
             return None

        db = FAISS.load_local(
            path,
            embeddings,
            allow_dangerous_deserialization=True # Required for loading pickle files
        )
        st.success(f"✅ Successfully loaded FAISS index from `{path}`")
        return db
    except Exception as e:
        st.error(f"❌ Error loading FAISS index from `{path}`: {e}")
        st.exception(e)
        return None

@st.cache_resource
def load_llm_and_pipeline(model_name):
    """Loads the Language Model and Pipeline using st.cache_resource."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"⚙️ Using device: {device}")

        model = None
        tokenizer = None
        pipe = None

        # Configure quantization if CUDA is available
        bnb_config = None
        if device == "cuda":
            try:
                 # Check compute capability for bfloat16 (usually required for optimal nf4)
                major, minor = torch.cuda.get_device_capability(0)
                if major >= 8: # Devices with compute capability 8.0+ support bfloat16 natively
                     bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16 # Use bfloat16 if supported
                     )
                     st.info("⚙️ GPU supports bfloat16. Loading with 4-bit quantization.")
                else:
                     bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        # compute_dtype will default, often float16 on older GPUs
                     )
                     st.warning("⚠️ GPU does not natively support bfloat16 (compute capability < 8.0). Loading with 4-bit quantization (compute_dtype might be float16).")

            except Exception as e:
                 st.warning(f"⚠️ Could not configure BitsAndBytes, attempting without quantization: {e}")
                 bnb_config = None # Fallback if BNB config fails

        # Load model with quantization if config exists, otherwise load normally
        if bnb_config:
             model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto", # Auto device mapping is recommended with quantization
                trust_remote_code=False,
             )
             st.success("✅ Language model loaded with 4-bit quantization.")
        else:
             st.warning("⚠️ BitsAndBytes quantization not applied. Loading model without quantization (might require more memory).")
             # Load without quantization, explicitly moving to device
             model = AutoModelForCausalLM.from_pretrained(
                 model_name,
                 trust_remote_code=False,
             ).to(device) # Move to CUDA or CPU

             st.success("✅ Language model loaded in full precision.")


        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Set pad token id if missing - crucial for batching and pipeline generation
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
                st.warning("⚠️ Tokenizer pad_token_id not set, using eos_token_id as pad_token_id.")
            else:
                 # Fallback: Add a new pad token
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                model.resize_token_embeddings(len(tokenizer)) # Resize model embeddings to fit new token
                st.warning("⚠️ Tokenizer missing pad_token and eos_token. Added a new PAD token and resized model embeddings.")

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512, # Keep max_new_tokens reasonable to limit output size and memory
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            # Use explicit pad_token_id and eos_token_id from tokenizer
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            # device is handled by device_map="auto" or model.to(device) above
        )
        st.success("✅ Text generation pipeline created.")

        return pipe # Return the pipeline directly
    except ImportError:
        st.error("❌ Error: `bitsandbytes` and/or `accelerate` not installed. These are required for 4-bit quantization.")
        st.markdown("Please install required libraries: `pip install bitsandbytes accelerate`")
        return None
    except Exception as e:
        st.error(f"❌ Error loading Language Model or creating pipeline: {e}")
        st.exception(e)
        return None

@st.cache_resource
def create_retrieval_qa_chain(llm_pipeline, db_retriever):
    """Creates the RetrievalQA chain using st.cache_resource."""
    if llm_pipeline is None or db_retriever is None:
        st.error("LLM Pipeline or Retriever not available, cannot create QA chain.")
        return None

    try:
        llm = HuggingFacePipeline(pipeline=llm_pipeline)

        # Define QA prompt (Keep your existing prompt)
        qa_prompt_template = """### System:
You are a helpful AI assistant focused on international relations, using the provided context to answer the question at the end.
If you don't know the answer based on the context, just say that you don't know. Do not make up an answer.
Provide ONE concise, direct answer only. Format lists as numbered points.
Focus only on information found in the context.
Do not repeat the question or include phrases like "Based on the context" or "According to the documents".
Do not include any prefixes like "Answer:" or "Response:".
Do not reference any previous questions or answers as you have no memory of past interactions.

### Context:
{context}

### Question:
{question}

### Assistant:
"""
        QA_PROMPT = PromptTemplate.from_template(qa_prompt_template)

        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", # 'stuff' is simplest, puts all context into one prompt
            retriever=db_retriever,
            return_source_documents=False, # As per original request
            chain_type_kwargs={"prompt": QA_PROMPT}
        )
        st.success("✅ Retrieval QA chain created.")
        return qa_chain
    except Exception as e:
        st.error(f"❌ Error creating Retrieval QA chain: {e}")
        st.exception(e)
        return None

# --- Streamlit App Structure ---

# --- Header Area with Logo and Title ---
col1, col2 = st.columns([1, 3])
with col1:
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <div style="font-size: 64px; color: #6200EA;">🔍</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.title("IR Knowledge Hub")
    st.markdown("""
    <p style="font-size: 1.2rem; color: #666; margin-top: -5px;">
        Your intelligent assistant for International Relations research
    </p>
    """, unsafe_allow_html=True)

# System Status Indicator
# Check if the chain exists to indicate readiness
if st.session_state.get('qa_chain') is not None:
    st.markdown("""
    <div class="status-indicator status-ready">
        ✅ System Ready - Ask your questions below
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="status-indicator status-waiting">
        ⏳ System Awaiting Configuration - Please load required models
    </div>
    """, unsafe_allow_html=True)

# --- Configuration Section ---
with st.expander("💻 System Configuration", expanded=False):
    st.markdown("""
    <div class="custom-card">
        <h3>FAISS Index Configuration</h3>
        <p>Specify the path to your pre-built FAISS index directory containing your knowledge base.</p>
    </div>
    """, unsafe_allow_html=True)

    # Path to the FAISS index directory
    # Keep the path input, but the cached functions will load based on its value
    faiss_index_path_input = st.text_input(
        "FAISS Index Directory Path:",
        value=FAISS_INDEX_PATH, # Use the default constant
        help="The directory containing your pre-built FAISS vector store"
    )

    col1, col2 = st.columns([3, 1])
    with col2:
        # The button now just triggers the loading process which uses cached functions
        load_system_button = st.button("Load System", use_container_width=True)

    # Loading logic triggered by the button
    if load_system_button:
         # Clear previous chain/status to show loading animation accurately
         # Note: The cached resources themselves are NOT cleared, just the state variables
         st.session_state.qa_chain = None
         st.session_state.model_loaded = False
         st.session_state.llm_pipeline = None # Clear previous states

         with st.spinner("Initializing knowledge system (this may take a few minutes the first time)..."):
             # Call the cached functions
             embeddings = load_embedding_model(EMBEDDING_MODEL_NAME)
             if embeddings:
                # Use the value from the text input for loading the index
                db = load_faiss_database(faiss_index_path_input, embeddings)
                if db:
                     # Set up retriever
                     retriever = db.as_retriever(
                        search_type="mmr", # Use MMR as specified
                        search_kwargs={
                            "k": 3,
                            "fetch_k": 6,
                            "lambda_mult": 0.7,
                        }
                     )
                     st.success("✅ Document retriever configured.")

                     # Load LLM and pipeline
                     llm_pipeline = load_llm_and_pipeline(LLM_MODEL_NAME)
                     if llm_pipeline:
                         st.session_state.llm_pipeline = llm_pipeline # Store pipeline if needed elsewhere
                         st.session_state.model_loaded = True # Indicate model is loaded

                         # Create the QA chain
                         qa_chain = create_retrieval_qa_chain(llm_pipeline, retriever)
                         if qa_chain:
                             st.session_state.qa_chain = qa_chain # Store the final chain
                             st.success("✨ System fully initialized and ready!")
                         else:
                            st.error("❌ Failed to create QA chain.")
                            # Reset states if chain creation failed
                            st.session_state.qa_chain = None
                            st.session_state.model_loaded = False
                            st.session_state.llm_pipeline = None

                else:
                    st.error("❌ Failed to load FAISS database.")
                    # Reset states if FAISS failed
                    st.session_state.qa_chain = None
                    st.session_state.model_loaded = False
                    st.session_state.llm_pipeline = None
             else:
                 st.error("❌ Failed to load embedding model.")
                 # Reset states if embeddings failed
                 st.session_state.qa_chain = None
                 st.session_state.model_loaded = False
                 st.session_state.llm_pipeline = None


# Create tabs for different sections
tab1, tab2 = st.tabs(["🔎 Research Assistant", "ℹ️ About"])

with tab1:
    # --- Chat Interface ---
    st.markdown("""
    <div class="custom-card">
        <h2>Query the Knowledge Base</h2>
        <p>Ask specific questions about international relations topics and get concise, evidence-based answers.</p>
    </div>
    """, unsafe_allow_html=True)

    # Check if the system is ready before showing the input
    if st.session_state.get('qa_chain') is not None:
        query = st.text_area(
            "What would you like to know about International Relations?",
            key="question_input",
            height=120,
            placeholder="Example: What are the key principles of constructivism in international relations theory?"
        )

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            generate_button = st.button("Generate Response", use_container_width=True)

        if generate_button:
            if query:
                try:
                    with st.spinner("Analyzing knowledge base and formulating response..."):
                        # Access the chain from session state
                        qa_chain = st.session_state.qa_chain
                        if qa_chain:
                            # This call returns the full text output including the prompt template parts
                            response = qa_chain({"query": query})

                            # Get the raw output string from the LLM
                            raw_output = response.get('result', '')

                            # --- START: Logic to extract only the content after "### Assistant:" ---
                            # Ensure the prefix matches exactly what the template uses
                            answer_prefix = "### Assistant:"
                            answer = raw_output # Default to full output if prefix isn't found

                            # Find the index where the assistant's response is supposed to start
                            # Add 1 to the end to account for the newline after the colon
                            prefix_index = raw_output.find(answer_prefix)

                            if prefix_index != -1:
                                # The actual answer starts right after the prefix and potential newline/space
                                answer = raw_output[prefix_index + len(answer_prefix):].lstrip() # Use lstrip to remove leading whitespace/newlines
                            else:
                                # Fallback: If the prefix isn't found (e.g., model didn't follow format)
                                st.warning("Could not find the expected response format in the output. Displaying raw output.")
                                answer = raw_output.strip() # Still strip leading/trailing whitespace
                            # --- END: Logic to extract only the content after "### Assistant:" ---

                            # Display ONLY the extracted answer using the custom styling
                            st.markdown("### Expert Response:")
                            # Ensure the answer is treated as markdown potentially within the styled box
                            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
                        else:
                             st.error("System is not initialized. Please click 'Load System' above.")


                except Exception as e:
                    st.error(f"An error occurred during response generation: {e}")
                    st.exception(e) # Display full traceback for debugging
            else:
                st.warning("Please enter a question before generating a response.")

    else:
        st.markdown("""
        <div style="background-color: rgba(255, 214, 0, 0.1); padding: 20px; border-radius: 10px; border-left: 4px solid #FF6D00;">
            <h3 style="color: #FF6D00; margin-top: 0;">System Not Ready</h3>
            <p>Please initialize the knowledge system first by using the System Configuration panel above.</p>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    # --- About Section ---
    st.markdown("""
    <div class="custom-card">
        <h2>About IR Knowledge Hub</h2>
        <p>This intelligent assistant uses state-of-the-art AI technology to help researchers, students, and professionals
        access accurate information about international relations topics.</p>

        <h3>How It Works</h3>
        <p>The system uses a Retrieval-Augmented Generation (RAG) architecture to provide accurate, evidence-based answers:</p>
        <ol>
            <li>Your specialized documents are embedded and stored in a FAISS vector database</li>
            <li>When you ask a question, the system retrieves the most relevant information from your knowledge base</li>
            <li>A fine-tuned language model generates a concise, accurate response based on the retrieved context</li>
        </ol>

        <h3>Technical Details</h3>
        <p>This system uses the following technologies:</p>
        <ul>
            <li>Embedding Model: <code>sentence-transformers/all-MiniLM-L6-v2</code></li>
            <li>Language Model: <code>TinyLlama/TinyLlama-1.1B-Chat-v1.0</code></li>
            <li>Vector Database: FAISS (Facebook AI Similarity Search)</li>
            <li>Framework: LangChain + Streamlit</li>
            <li>Memory Optimization: 4-bit Quantization (requires GPU and `bitsandbytes`, `accelerate`)</li>
        </ul>
         <p><strong>Note:</strong> Loading the models requires significant resources. The system uses caching and quantization to manage memory, but performance may vary depending on the available resources on the hosting platform.</p>
    </div>
    """, unsafe_allow_html=True)

# --- Sidebar for System Info ---
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 10px;">
        <h2 style="color: #6200EA;">System Status</h2>
    </div>
    """, unsafe_allow_html=True)

    # Create a stylized info card for system status
    st.markdown("""
    <div style="background-color: #FFFFFF; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
    """, unsafe_allow_html=True) # <-- This div starts the main info card


    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_icon = "⚡" if device == "cuda" else "💻"

    st.markdown(f"""
    <div style="margin-bottom: 15px;">
        <span style="font-weight: 600; color: var(--primary-color);">Processing Device:</span>
        <span style="background-color: {'rgba(0, 200, 83, 0.1)' if device == 'cuda' else 'rgba(255, 214, 0, 0.1)'};
               padding: 3px 10px; border-radius: 50px; font-size: 0.9rem; color: {'#00C853' if device == 'cuda' else '#FF6D00'};">
            {device_icon} {device.upper()}
        </span>
    </div>
    """, unsafe_allow_html=True)

    if device == "cuda" and torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            st.markdown(f"""
            <div style="margin-bottom: 15px;">
                <span style="font-weight: 600; color: var(--primary-color);">GPU:</span> {gpu_name}
            </div>
            """, unsafe_allow_html=True)

            # Check if pynvml is available for better memory stats
            try:
                from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown
                nvmlInit()
                handle = nvmlDeviceGetHandleByIndex(0)
                info = nvmlDeviceGetMemoryInfo(handle)

                # Calculate percentage used
                percent_used = (info.used / info.total) * 100

                st.markdown(f"""
                <div style="margin-bottom: 15px;">
                    <span style="font-weight: 600; color: var(--primary-color);">GPU Memory:</span>
                    <div style="margin-top: 5px; height: 10px; background-color: #E0E0E0; border-radius: 5px;">
                        <div style="width: {percent_used:.2f}%; height: 10px; background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
                             border-radius: 5px;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-top: 5px;">
                        <span>Used: {info.used/1024**3:.2f} GB</span>
                        <span>Total: {info.total/1024**3:.2f} GB</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                nvmlShutdown()
            except ImportError:
                st.markdown("""
                <div style="margin-bottom: 15px; font-size: 0.9rem; color: #FF6D00;">
                    <i>Install pynvml (`pip install pynvml`) for detailed GPU memory statistics</i>
                </div>
                """, unsafe_allow_html=True)
                # Fallback to torch stats
                try:
                    # Wrap torch calls in a check to avoid errors if CUDA isn't actually working
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated(0)/1024**2
                        cached = torch.cuda.memory_cached(0)/1024**2
                        st.markdown(f"""
                        <div style="margin-bottom: 15px;">
                            <span style="font-weight: 600; color: var(--primary-color);">GPU Memory (torch):</span>
                            <div style="margin-top: 8px;">
                                <span>Allocated: {allocated:.2f} MB</span><br>
                                <span>Cached: {cached:.2f} MB</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                         st.markdown("""
                        <div style="margin-bottom: 15px; color: #D50000; font-size: 0.9rem;">
                            <span>CUDA not available for torch memory info.</span>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as torch_mem_e:
                     st.markdown(f"""
                        <div style="margin-bottom: 15px; color: #D50000; font-size: 0.9rem;">
                            <span>Could not retrieve torch GPU memory info: {str(torch_mem_e)}</span>
                        </div>
                        """, unsafe_allow_html=True)
            except Exception as gpu_info_e:
                st.markdown(f"""
                <div style="margin-bottom: 15px; color: #D50000; font-size: 0.9rem;">
                    <span>Could not retrieve GPU info: {str(gpu_info_e)}</span>
                </div>
                """, unsafe_allow_html=True)


    # Models info
    st.markdown("""
    <div style="margin-top: 25px; margin-bottom: 15px;">
        <span style="font-weight: 600; color: var(--primary-color); font-size: 1.1rem;">Models & Index</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-bottom: 10px;">
        <span style="font-weight: 600; color: var(--primary-color);">Embedding:</span>
        <div style="background-color: rgba(98, 0, 234, 0.05); padding: 8px; border-radius: 5px; margin-top: 5px;">
            <code>{EMBEDDING_MODEL_NAME}</code>
        </div>
    </div>

    <div style="margin-bottom: 10px;">
        <span style="font-weight: 600; color: var(--primary-color);">Language Model:</span>
        <div style="background-color: rgba(98, 0, 234, 0.05); padding: 8px; border-radius: 5px; margin-top: 5px;">
            <code>{LLM_MODEL_NAME}</code>
        </div>
    </div>

    <div style="margin-bottom: 20px;">
        <span style="font-weight: 600; color: var(--primary-color);">FAISS Index:</span>
        <div style="background-color: rgba(98, 0, 234, 0.05); padding: 8px; border-radius: 5px; margin-top: 5px;
                   overflow-wrap: break-word; word-wrap: break-word;">
            <code>{FAISS_INDEX_PATH}</code>
        </div>
    </div>

    <div style="margin-top: 25px; margin-bottom: 15px;">
        <span style="font-weight: 600; color: var(--primary-color); font-size: 1.1rem;">System Status</span>
    </div>

    <div style="margin-bottom: 15px;">
        <span style="font-weight: 600; color: var(--primary-color);">Ready Status:</span>
        <span style="background-color: {'rgba(0, 200, 83, 0.1)' if st.session_state.get('qa_chain') else 'rgba(255, 214, 0, 0.1)'};
               padding: 3px 10px; border-radius: 50px; font-size: 0.9rem;
               color: {'#00C853' if st.session_state.get('qa_chain') else '#FF6D00'};">
            {'✅ Ready' if st.session_state.get('qa_chain') else '⏳ Waiting'}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # <<< --- CORRECTED LINE POSITION --- >>>
    st.markdown("</div>", unsafe_allow_html=True) # Close the main sidebar info card div

    # Help section (now correctly outside the main info card div)
    st.markdown("""
    <div style="margin-top: 30px; background-color: #E1F5FE; padding: 20px; border-radius: 10px;">
        <h3 style="color: #0277BD; margin-top: 0;">Quick Tips</h3>
        <ul style="padding-left: 20px; margin-bottom: 0;">
            <li>Ask specific questions for more accurate answers</li>
            <li>Include key terms relevant to your research</li>
            <li>For complex topics, break down into multiple queries</li>
            <li>System works best with focused, clear questions</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# --- Footer ---
st.markdown("---")
st.markdown("""
<div class="footer">
    <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 10px;">
        <div style="font-size: 24px; margin-right: 10px;">🔍</div>
        <div style="font-weight: 600; font-size: 1.2rem; background: linear-gradient(90deg, #6200EA, #00BFA5);
                   -webkit-background-clip: text; background-clip: text; color: transparent;">
            IR Knowledge Hub
        </div>
    </div>
    <p style="color: #666; font-size: 0.9rem;">Built with Streamlit, LangChain, Hugging Face, and FAISS</p>
    <p style="color: #666; font-size: 0.8rem;">© 2025 | International Relations Research Assistant</p>
</div>
""", unsafe_allow_html=True)
