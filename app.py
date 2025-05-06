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


st.set_page_config(
    page_title="IR Knowledge Hub", 
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded" 
)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #6200EA;        /* Deep purple */
        --secondary-color: #00BFA5;      /* Teal accent */
        --background-color: #8B0000;     /* Light grey background */
        --text-color: #000000;           /* Dark blue-grey */
        --light-accent: #E1F5FE;         /* Light blue accent */
        --dark-accent: #311B92;          /* Darker purple */
        --success-color: #00C853;        /* Green success */
        --warning-color: #FFD600;        /* Yellow warning */
        --error-color: #D50000;          /* Red error */
    }
    
    /* General container styling */
    .reportview-container {
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
        background-color: black;
        border-radius: 16px;
        padding: 25px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        margin-bottom: 24px;
        border-top: 5px solid var(--primary-color);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #FFFFFF;
        box-shadow: 2px 0 5px rgba(0, 0, 0, 0.05);
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
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        color: black !important;
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
        background-color: black;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        margin: 20px 0;
        border-left: 5px solid var(--secondary-color);
        line-height: 1.7;
        font-size: 1.05rem;
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
    }
    
    .streamlit-expanderContent {
        border: none !important;
        border-top: none !important;
        background-color: black !important;
        border-radius: 0 0 8px 8px !important;
        padding: 20px !important;
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
    
    /* Sidebar header */
    .sidebar .sidebar-content h1 {
        font-size: 1.5rem !important;
    }
    
    /* Footer styling */
    .footer {
        background-color: black;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 30px;
        border-top: 3px solid var(--light-accent);
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
        color: black;
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
    .stTabs [data-baseweb="tab-list"] {
        background-color: black;
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 5px;
        padding: 10px 20px;
        margin-right: 5px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color) !important;
        color: black !important;
    }
</style>
""", unsafe_allow_html=True)

# Load custom fonts
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# --- Initialize session state variables if they don't exist ---
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'llm_pipeline' not in st.session_state:
    st.session_state.llm_pipeline = None

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
if st.session_state.qa_chain is not None:
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
    faiss_index_path = st.text_input(
        "FAISS Index Directory Path:",
        value="faiss_index_uploaded_data",
        help="The directory containing your pre-built FAISS vector store"
    )
    
    col1, col2 = st.columns([3, 1])
    with col2:
        load_index_button = st.button("Load System", use_container_width=True)

    if load_index_button:
        if st.session_state.qa_chain is not None:
            st.info("System is already loaded and ready for queries.")
        else:
            with st.spinner("Initializing knowledge system..."):
                try:
                    # Load embeddings
                    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
                    st.success("Embedding model loaded successfully.")

                    # Load existing FAISS index
                    db = FAISS.load_local(
                        faiss_index_path,
                        embeddings,
                        allow_dangerous_deserialization=True # Required for loading pickle files
                    )
                    st.success(f"Successfully loaded FAISS index from `{faiss_index_path}`")

                    # Set up retriever
                    retriever = db.as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            "k": 3,
                            "fetch_k": 6,
                            "lambda_mult": 0.7,
                        }
                    )
                    st.success("Document retriever configured.")

                    # Load the model if it hasn't been loaded yet
                    if not st.session_state.model_loaded:
                        with st.spinner("Loading language model (this may take a few minutes)..."):
                            device = "cuda" if torch.cuda.is_available() else "cpu"
                            st.info(f"Using device: {device}")

                            # Check if GPU is available and handle bfloat16 if not
                            bnb_config = None # Default to no quantization
                            if device == "cuda" and torch.cuda.is_available():
                                # Check compute capability for bfloat16
                                major, minor = torch.cuda.get_device_capability(0)
                                if major >= 8: # Devices with compute capability 8.0+ support bfloat16 natively
                                    bnb_config = BitsAndBytesConfig(
                                        load_in_4bit=True,
                                        bnb_4bit_use_double_quant=True,
                                        bnb_4bit_quant_type="nf4",
                                        bnb_4bit_compute_dtype=torch.bfloat16
                                    )
                                    st.info("GPU supports bfloat16. Loading with 4-bit quantization.")
                                else:
                                    bnb_config = BitsAndBytesConfig( # Still try 4bit without bfloat16
                                        load_in_4bit=True,
                                        bnb_4bit_use_double_quant=True,
                                        bnb_4bit_quant_type="nf4",
                                        # bnb_4bit_compute_dtype will default based on GPU or be None
                                    )
                                    st.warning("GPU does not natively support bfloat16 (compute capability < 8.0). Loading with 4-bit quantization (compute_dtype might differ).")

                            if bnb_config is None and device == "cuda":
                                st.warning("GPU detected but not suitable for 4-bit quantization. Loading model in full precision on GPU (if memory allows).")
                                # Load in full precision on GPU if no bnb_config but cuda is available
                                model = AutoModelForCausalLM.from_pretrained(
                                    LLM_MODEL_NAME,
                                    trust_remote_code=False,
                                ).to(device) # Explicitly move to device

                            elif bnb_config is None and device == "cpu":
                                st.warning("GPU not available. Loading model in full precision on CPU (slow).")
                                # Load in full precision on CPU
                                model = AutoModelForCausalLM.from_pretrained(
                                    LLM_MODEL_NAME,
                                    trust_remote_code=False,
                                ) # Defaults to CPU

                            else: # bnb_config is not None (meaning on GPU with 4-bit)
                                model = AutoModelForCausalLM.from_pretrained(
                                    LLM_MODEL_NAME,
                                    quantization_config=bnb_config,
                                    device_map="auto", # Use device_map="auto" with quantization
                                    trust_remote_code=False,
                                )

                            tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
                            # Set pad token id if missing
                            if tokenizer.pad_token_id is None:
                                if tokenizer.eos_token_id is not None:
                                    tokenizer.pad_token_id = tokenizer.eos_token_id
                                else:
                                    # Fallback if neither is set - common for some models
                                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                                    model.resize_token_embeddings(len(tokenizer)) # Resize model embeddings
                                    tokenizer.pad_token_id = tokenizer.pad_token_id # Get the new pad_token_id
                                    st.warning("Tokenizer missing pad_token, added a new PAD token.")

                            pipe = pipeline(
                                "text-generation",
                                model=model,
                                tokenizer=tokenizer,
                                max_new_tokens=512,
                                do_sample=True,
                                temperature=0.7,
                                top_p=0.95,
                                # Use explicit pad_token_id and eos_token_id from tokenizer
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                                # device=0 if device == "cuda" else -1 # pipeline handles device with device_map="auto" or model.to(device)
                            )

                            llm = HuggingFacePipeline(pipeline=pipe)
                            st.session_state.llm_pipeline = pipe # Store the pipeline
                            st.session_state.model_loaded = True
                            st.success("Language model loaded and ready for inference.")
                    else:
                        # Model is already loaded, retrieve the pipeline
                        llm = HuggingFacePipeline(pipeline=st.session_state.llm_pipeline)
                        st.info("Language model was already loaded.")

                    # Define QA prompt
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
""" # Keep the ### Assistant: marker here - the parsing below will handle it
                    QA_PROMPT = PromptTemplate.from_template(qa_prompt_template)

                    # Create QA chain
                    st.session_state.qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=retriever,
                        return_source_documents=False, # Do NOT return source documents
                        chain_type_kwargs={"prompt": QA_PROMPT}
                    )

                    st.success("Knowledge system fully initialized and ready!")

                except Exception as e:
                    st.error(f"An error occurred during loading: {e}")
                    st.exception(e) # Display full traceback for loading errors
                    # Reset state on error
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
    if st.session_state.qa_chain is not None:
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
                        # This call returns the full text output including the prompt template parts
                        response = st.session_state.qa_chain({"query": query})

                        # Get the raw output string from the LLM
                        raw_output = response.get('result', '')

                        # --- START: Logic to extract only the content after "### Assistant:" ---
                        answer_prefix = "### Assistant:"
                        answer = raw_output # Default to full output if prefix isn't found

                        # Find the index where the assistant's response is supposed to start
                        prefix_index = raw_output.find(answer_prefix)

                        if prefix_index != -1:
                            # The actual answer starts right after the prefix
                            answer = raw_output[prefix_index + len(answer_prefix):].strip()
                        else:
                            # Fallback: If the prefix isn't found (e.g., model didn't follow format)
                            st.warning("Could not find the expected response format in the output. Displaying raw output.")
                            answer = raw_output.strip() # Still strip leading/trailing whitespace
                        # --- END: Logic to extract only the content after "### Assistant:" ---

                        # Display ONLY the extracted answer using the custom styling
                        st.markdown("### Expert Response:")
                        # Ensure the answer is treated as markdown potentially within the styled box
                        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

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
        </ul>
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
    <div style="background-color: black; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
    """, unsafe_allow_html=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_icon = "⚡" if device == "cuda" else "💻"
    
    st.markdown(f"""
    <div style="margin-bottom: 15px;">
        <span style="font-weight: 600; color: #6200EA;">Processing Device:</span> 
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
                <span style="font-weight: 600; color: #6200EA;">GPU:</span> {gpu_name}
            </div>
            """, unsafe_allow_html=True)
            
            # Check if pynvml is available
            try:
                from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown
                nvmlInit()
                handle = nvmlDeviceGetHandleByIndex(0)
                info = nvmlDeviceGetMemoryInfo(handle)
                
                # Calculate percentage used
                percent_used = (info.used / info.total) * 100
                
                st.markdown(f"""
                <div style="margin-bottom: 15px;">
                    <span style="font-weight: 600; color: #6200EA;">GPU Memory:</span>
                    <div style="margin-top: 5px; height: 10px; background-color: #E0E0E0; border-radius: 5px;">
                        <div style="width: {percent_used}%; height: 10px; background: linear-gradient(90deg, #6200EA, #00BFA5); 
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
                    <i>Install pynvml for detailed GPU memory statistics</i>
                </div>
                """, unsafe_allow_html=True)
                # Fallback to torch stats
                allocated = torch.cuda.memory_allocated(0)/1024**2
                cached = torch.cuda.memory_cached(0)/1024**2
                st.markdown(f"""
                <div style="margin-bottom: 15px;">
                    <span style="font-weight: 600; color: #6200EA;">GPU Memory (torch):</span>
                    <div style="margin-top: 8px;">
                        <span>Allocated: {allocated:.2f} MB</span><br>
                        <span>Cached: {cached:.2f} MB</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f"""
            <div style="margin-bottom: 15px; color: #D50000;">
                <span>Could not retrieve GPU info: {str(e)}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Models info
    st.markdown("""
    <div style="margin-top: 25px; margin-bottom: 15px;">
        <span style="font-weight: 600; color: #6200EA; font-size: 1.1rem;">Models</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="margin-bottom: 10px;">
        <span style="font-weight: 600; color: #6200EA;">Embedding:</span>
        <div style="background-color: rgba(98, 0, 234, 0.05); padding: 8px; border-radius: 5px; margin-top: 5px;">
            <code>{EMBEDDING_MODEL_NAME}</code>
        </div>
    </div>
    
    <div style="margin-bottom: 10px;">
        <span style="font-weight: 600; color: #6200EA;">Language Model:</span>
        <div style="background-color: rgba(98, 0, 234, 0.05); padding: 8px; border-radius: 5px; margin-top: 5px;">
            <code>{LLM_MODEL_NAME}</code>
        </div>
    </div>
    
    <div style="margin-bottom: 20px;">
        <span style="font-weight: 600; color: #6200EA;">FAISS Index:</span>
        <div style="background-color: rgba(98, 0, 234, 0.05); padding: 8px; border-radius: 5px; margin-top: 5px; 
                   overflow-wrap: break-word; word-wrap: break-word;">
            <code>{faiss_index_path}</code>
        </div>
    </div>
    
    <div style="margin-top: 25px; margin-bottom: 15px;">
        <span style="font-weight: 600; color: #6200EA; font-size: 1.1rem;">System Status</span>
    </div>
    
    <div style="margin-bottom: 15px;">
        <span style="font-weight: 600; color: #6200EA;">Ready Status:</span> 
        <span style="background-color: {'rgba(0, 200, 83, 0.1)' if st.session_state.qa_chain else 'rgba(255, 214, 0, 0.1)'}; 
               padding: 3px 10px; border-radius: 50px; font-size: 0.9rem; 
               color: {'#00C853' if st.session_state.qa_chain else '#FF6D00'};">
            {'✅ Ready' if st.session_state.qa_chain else '⏳ Waiting'}
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Help section
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