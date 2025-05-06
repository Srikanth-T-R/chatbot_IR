import streamlit as st
import os
import torch
# Import sentence_transformers explicitly as we will load it directly
import sentence_transformers
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, logging as hf_logging
import logging # Import standard logging

# Suppress Hugging Face logging messages that aren't warnings or errors
hf_logging.set_verbosity_warning()
logging.basicConfig(level=logging.WARNING) # Set root logger level to warning

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
        --background-color: #f4f4f4;     /* Light grey background - Changed from red for better visibility */
        --text-color: #212121;           /* Dark grey */
        --light-accent: #E1F5FE;         /* Light blue accent */
        --dark-accent: #311B92;          /* Darker purple */
        --success-color: #00C853;        /* Green success */
        --warning-color: #FFD600;        /* Yellow warning */
        --error-color: #D50000;          /* Red error */
    }

    /* General container styling */
    /* Remove reportview-container - deprecated. Style main block container instead */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1200px;
        margin: 0 auto;
        background-color: var(--background-color); /* Apply background color */
        color: var(--text-color);
    }

    /* Custom card styling */
    .custom-card {
        background-color: #ffffff; /* Changed to white for contrast on grey background */
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
        padding-top: 2rem; /* Added padding */
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
        color: white !important; /* Changed to white for contrast */
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

     /* Disabled button styling */
    .stButton > button:disabled {
        background: #cccccc !important; /* Grey background for disabled */
        color: #666666 !important;     /* Dark grey text */
        cursor: not-allowed !important;
        box-shadow: none !important;
        transform: none !important;
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
        background-color: #ffffff; /* Changed to white */
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
        padding: 1rem 1.2rem !important; /* Added padding */
    }

     .streamlit-expanderHeader > div > div > p { /* Target the text inside expander header */
        font-weight: 600 !important;
        color: var(--primary-color) !important;
    }


    .streamlit-expanderContent {
        border: none !important;
        border-top: none !important;
        background-color: #ffffff !important; /* Changed to white */
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
    .stAlert { /* Use stAlert as info/success/warning/error boxes are now based on it */
        border-radius: 12px !important;
        padding: 20px !important;
    }

    /* Sidebar header */
    .sidebar .sidebar-content h2 { /* Adjusted from h1 */
        font-size: 1.5rem !important;
        margin-top: 0 !important;
        margin-bottom: 1rem !important;
        text-align: center; /* Center align sidebar header */
    }
     .sidebar .sidebar-content h3 { /* Style h3 in sidebar */
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: var(--primary-color);
        margin-top: 1rem !important;
        margin-bottom: 0.5rem !important;
    }


    /* Footer styling */
    .footer {
        background-color: #e0e0e0; /* Light grey background */
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 30px;
        border-top: 3px solid var(--light-accent);
        color: #555; /* Darker grey text */
    }
    .footer p {
        margin-bottom: 5px;
        color: #555; /* Ensure text color is consistent */
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
        background-color: #333; /* Dark background */
        color: #fff; /* White text */
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
        font-size: 0.85rem; /* Smaller font */
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }

    /* Custom tabs if needed */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #ffffff; /* White background */
        border-radius: 10px;
        padding: 5px;
         margin-bottom: 1rem; /* Space below tabs */
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 5px;
        padding: 10px 20px;
        margin-right: 5px;
         color: var(--text-color); /* Default tab text color */
         font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color) !important;
        color: white !important; /* White text for selected tab */
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] > div { /* Target text inside selected tab */
         color: white !important;
    }
     .stTabs [data-baseweb="tab"]:hover { /* Hover effect for tabs */
        background-color: rgba(98, 0, 234, 0.1); /* Light purple hover */
         color: var(--primary-color) !important;
    }
     .stTabs [aria-selected="true"]:hover { /* No change on hover for selected */
         background-color: var(--primary-color) !important;
         color: white !important;
    }

     /* General text and link color adjustment */
     body {
        color: var(--text-color);
    }
    a {
        color: var(--primary-color);
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
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
if 'used_device' not in st.session_state:
    st.session_state.used_device = "Not Loaded Yet" # Track the device actually used
if 'loading_in_progress' not in st.session_state:
    st.session_state.loading_in_progress = False # Flag to indicate if loading button was clicked

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
elif st.session_state.loading_in_progress:
     st.markdown("""
    <div class="status-indicator status-waiting">
        ⏳ Loading System - Please wait...
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="status-indicator status-waiting">
        ⏳ System Awaiting Configuration - Please load required models
    </div>
    """, unsafe_allow_html=True)


# --- Configuration Section ---
with st.expander("💻 System Configuration", expanded=True): # Start expanded for easier debugging
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

    # Add a checkbox to force CPU
    # Using session state for the checkbox value to make it persistent across reruns
    if 'force_cpu_checkbox' not in st.session_state:
        st.session_state.force_cpu_checkbox = False
    st.session_state.force_cpu_checkbox = st.checkbox(
        "Force CPU for all models",
        value=st.session_state.force_cpu_checkbox,
        help="Check this to load models on CPU even if a GPU is available. Useful for debugging or if GPU VRAM is insufficient.",
        disabled=st.session_state.loading_in_progress # Disable while loading
    )
    force_cpu = st.session_state.force_cpu_checkbox

    col1, col2 = st.columns([3, 1])
    with col2:
        load_system_button = st.button(
            "Load System",
            use_container_width=True,
            key="load_system_button",
            disabled=st.session_state.qa_chain is not None or st.session_state.loading_in_progress # Disable if already loaded or loading
        )

    # --- Loading Logic triggered by button click ---
    # Use a flag to manage the loading process across reruns
    if load_system_button:
        # Set flag and reset state to indicate loading is starting
        st.session_state.loading_in_progress = True
        st.session_state.qa_chain = None
        st.session_state.model_loaded = False
        st.session_state.llm_pipeline = None
        st.session_state.used_device = "Loading..."
        st.rerun() # Rerun immediately to show loading state

    if st.session_state.loading_in_progress and st.session_state.qa_chain is None:
        # This block runs on the rerun triggered by the button click
        with st.spinner(f"Initializing knowledge system on {st.session_state.used_device}..."): # Show spinner during the whole process
            try:
                # Determine the device - respect force_cpu checkbox value from session state
                determined_device = "cpu"
                if not st.session_state.force_cpu_checkbox and torch.cuda.is_available():
                    determined_device = "cuda"
                    try:
                         gpu_name = torch.cuda.get_device_name(0)
                         st.info(f"GPU available: {gpu_name}. Attempting to use CUDA.")
                    except Exception:
                         st.info("GPU available. Attempting to use CUDA.")
                else:
                    st.info("Loading models on CPU (GPU not available or Force CPU selected).")

                st.session_state.used_device = determined_device # Update session state with the chosen device

                # --- Load Embedding Model Manually ---
                st.info(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' onto {determined_device}...")
                try:
                    # Use sentence_transformers directly to control device loading
                    embedding_model = sentence_transformers.SentenceTransformer(EMBEDDING_MODEL_NAME, device=determined_device)
                    # Wrap the loaded model with HuggingFaceEmbeddings
                    embeddings = HuggingFaceEmbeddings(client=embedding_model)
                    st.success(f"Embedding model loaded successfully on {determined_device}.")
                except ImportError:
                     st.error("The 'sentence_transformers' library is not installed. Please add 'sentence-transformers' to your requirements.txt file.")
                     raise # Re-raise to fail gracefully
                except Exception as e:
                     st.error(f"Failed to load embedding model on {determined_device}: {e}")
                     st.exception(e) # Show traceback for embedding error
                     raise # Re-raise to fail gracefully


                # Load existing FAISS index
                st.info(f"Loading FAISS index from `{faiss_index_path}`...")
                if not os.path.exists(faiss_index_path) or not os.path.exists(os.path.join(faiss_index_path, "index.faiss")):
                     st.error(f"FAISS index not found at `{faiss_index_path}`. Please ensure the directory and 'index.faiss' file exist.")
                     raise FileNotFoundError(f"FAISS index not found at {faiss_index_path}")

                try:
                    db = FAISS.load_local(
                        faiss_index_path,
                        embeddings, # Use the correctly loaded embeddings object
                        allow_dangerous_deserialization=True # Required for loading pickle files
                    )
                    st.success(f"Successfully loaded FAISS index from `{faiss_index_path}`")
                except Exception as e:
                    st.error(f"Failed to load FAISS index from `{faiss_index_path}`: {e}")
                    st.exception(e)
                    raise # Re-raise

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

                # --- Load the LLM ---
                st.info(f"Loading language model '{LLM_MODEL_NAME}' on {determined_device}...")

                bnb_config = None # Default to no quantization
                if determined_device == "cuda":
                    try:
                        import bitsandbytes # Check if library is available
                        # Check compute capability for bfloat16
                        major, minor = torch.cuda.get_device_capability(0)
                        if major >= 8: # Devices with compute capability 8.0+ support bfloat16 natively
                            bnb_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16
                            )
                            st.info("GPU supports bfloat16 (Compute Capability >= 8.0). Loading with 4-bit quantization.")
                        else:
                            bnb_config = BitsAndBytesConfig( # Still try 4bit without bfloat16
                                load_in_4bit=True,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4",
                                # bnb_4bit_compute_dtype will default based on GPU or be None
                            )
                            st.warning("GPU does not natively support bfloat16 (compute capability < 8.0). Loading with 4-bit quantization (compute_dtype might differ).")
                    except ImportError:
                        st.warning("BitsAndBytes library not found or CUDA setup is incompatible. Cannot use 4-bit quantization. Attempting full precision load.")
                        bnb_config = None # Disable quantization if import fails
                    except Exception as e:
                        st.warning(f"Could not setup BitsAndBytes config: {e}. Attempting full precision load.")
                        bnb_config = None # Disable quantization on other errors


                # Load the model
                try:
                    if bnb_config is None and determined_device == "cuda":
                        st.warning("GPU detected but 4-bit quantization is not enabled or compatible. Loading model in full precision on GPU (requires significant VRAM).")
                        model = AutoModelForCausalLM.from_pretrained(
                            LLM_MODEL_NAME,
                            trust_remote_code=False,
                            # device_map="auto" is recommended for managing tensors across devices/quantization
                            # but for single GPU FP load, explicit .to(determined_device) works
                        ).to(determined_device) # Explicitly move to device

                    elif bnb_config is None and determined_device == "cpu":
                        st.warning("GPU not available or Force CPU selected and 4-bit quantization disabled. Loading model in full precision on CPU (will be very slow and require substantial CPU RAM).")
                        model = AutoModelForCausalLM.from_pretrained(
                            LLM_MODEL_NAME,
                            trust_remote_code=False,
                            # Defaults to CPU
                        )
                    elif bnb_config is not None: # bnb_config is not None (meaning on GPU with 4-bit)
                        st.info("Loading model with BitsAndBytes 4-bit quantization...")
                        model = AutoModelForCausalLM.from_pretrained(
                            LLM_MODEL_NAME,
                            quantization_config=bnb_config,
                            device_map="auto", # Use device_map="auto" with quantization
                            trust_remote_code=False,
                            # If device_map='auto', the model's modules are placed on available devices.
                            # The pipeline will then pick up the model's device placements.
                        )
                    else:
                         # Fallback/Shouldn't happen
                         st.error("Could not determine a valid model loading strategy based on device and quantization options.")
                         raise ValueError("Invalid loading strategy")

                    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
                    # Set pad token id if missing
                    if tokenizer.pad_token_id is None:
                        if tokenizer.eos_token_id is not None:
                            tokenizer.pad_token_id = tokenizer.eos_token_id
                            st.info(f"Tokenizer pad_token_id set to eos_token_id: {tokenizer.pad_token_id}")
                        else:
                            # Fallback if neither is set - common for some models
                            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                            # Resize model embeddings ONLY if adding a new token. Check if the token was actually added.
                            if tokenizer.pad_token_id is None: # It should be set after add_special_tokens
                                 model.resize_token_embeddings(len(tokenizer)) # Resize model embeddings
                                 tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]') # Get the new pad_token_id
                                 st.warning("Tokenizer missing pad_token, added a new PAD token and resized model embeddings.")
                            else:
                                 st.info(f"Tokenizer pad_token added: {tokenizer.pad_token}. ID: {tokenizer.pad_token_id}")

                    # Ensure model's pad_token_id is set for generation config if tokenizer has one
                    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
                         model.config.pad_token_id = tokenizer.pad_token_id
                         st.info(f"Set model's pad_token_id to tokenizer's pad_token_id: {tokenizer.pad_token_id}")


                    pipe = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.95,
                        # Ensure correct tokens are passed to pipeline generation args
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        # Passing device explicitly can sometimes help, but device_map="auto"
                        # on the model is usually sufficient for pipeline when using quantization.
                        # device=(0 if determined_device == "cuda" else -1)
                    )

                    llm = HuggingFacePipeline(pipeline=pipe)
                    st.session_state.llm_pipeline = pipe # Store the pipeline
                    st.session_state.model_loaded = True
                    st.success(f"Language model loaded successfully on {determined_device} and ready for inference.")

                except Exception as e:
                    st.error(f"Failed to load language model on {determined_device}: {e}")
                    st.exception(e) # Show traceback for LLM error
                    # Add a more specific OOM check hint if possible
                    if "CUDA out of memory" in str(e) or "hipErrorOutOfMemory" in str(e):
                         st.error("This looks like a CUDA/GPU Out of Memory error. The model or embeddings might be too large for the available VRAM. Try using the 'Force CPU' option (if you have enough CPU RAM) or running on a machine with more VRAM.")
                    if "bitsandbytes" in str(e):
                         st.error("BitsAndBytes error during quantization. Ensure `bitsandbytes` is installed correctly for your CUDA version and GPU.")
                    raise # Re-raise to fail gracefully


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
"""
                QA_PROMPT = PromptTemplate.from_template(qa_prompt_template)

                # Create QA chain
                st.session_state.qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=False,
                    chain_type_kwargs={"prompt": QA_PROMPT}
                )

                st.success("Knowledge system fully initialized and ready!")


            except Exception as e:
                # Catch any exceptions that weren't re-raised earlier and display them
                st.error(f"An error occurred during system loading: {e}")
                st.exception(e) # Display full traceback for any remaining errors
                # Reset state on error
                st.session_state.qa_chain = None
                st.session_state.model_loaded = False
                st.session_state.llm_pipeline = None
                st.session_state.used_device = "Failed" # Indicate loading failed

            finally:
                 # Always set loading flag to False after the process finishes (success or failure)
                 st.session_state.loading_in_progress = False
                 st.rerun() # Rerun one last time to update button/status states


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
            # Only enable the generate button if the QA chain is loaded
            generate_button = st.button("Generate Response", use_container_width=True, disabled=st.session_state.qa_chain is None)

        if generate_button:
            if query:
                try:
                    with st.spinner("Analyzing knowledge base and formulating response..."):
                        # This call returns the full text output including the prompt template parts
                        # LangChain v0.1.0+ will return {'query': ..., 'result': ..., 'source_documents': ...}
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
                            # Attempt to clean potential chat formatting if prefix wasn't found
                            clean_patterns = ["### System:", "### Context:", "### Question:"]
                            cleaned_answer = raw_output
                            for pattern in clean_patterns:
                                cleaned_answer = cleaned_answer.replace(pattern, "").strip()
                            answer = cleaned_answer # Use the cleaned version as fallback
                            if raw_output != answer:
                                 st.warning("Could not find the exact response format. Attempted to clean output.")
                            # else: raw_output had no obvious prefixes

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
    <div style="background-color: #f0f0f0; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); color: #333;">
    """, unsafe_allow_html=True)

    # Display the actual device used based on session state
    current_device_status = st.session_state.used_device
    device_icon = "⚡" if current_device_status == "cuda" else ("💻" if current_device_status == "cpu" else "...")
    device_color = '#00C853' if current_device_status == 'cuda' else ('#FF6D00' if current_device_status == 'cpu' else '#D50000')
    device_bg_color = 'rgba(0, 200, 83, 0.1)' if current_device_status == 'cuda' else ('rgba(255, 214, 0, 0.1)' if current_device_status == 'cpu' else 'rgba(213, 0, 0, 0.1)')


    st.markdown(f"""
    <div style="margin-bottom: 15px;">
        <span style="font-weight: 600; color: #6200EA;">Processing Device:</span>
        <span style="background-color: {device_bg_color};
               padding: 3px 10px; border-radius: 50px; font-size: 0.9rem; color: {device_color};">
            {device_icon} {current_device_status.upper()}
        </span>
         {'<span style="font-size: 0.8rem; color: #FF6D00;">(Forced)</span>' if st.session_state.force_cpu_checkbox and current_device_status == 'cpu' else ''}
    </div>
    """, unsafe_allow_html=True)

    # Display Force CPU setting status
    st.markdown(f"""
    <div style="margin-bottom: 15px;">
        <span style="font-weight: 600; color: #6200EA;">Force CPU Active:</span>
        <span style="background-color: {'rgba(0, 200, 83, 0.1)' if st.session_state.force_cpu_checkbox else 'rgba(213, 0, 0, 0.1)'};
               padding: 3px 10px; border-radius: 50px; font-size: 0.9rem; color: {'#00C853' if st.session_state.force_cpu_checkbox else '#D50000'};">
            {'✅ Yes' if st.session_state.force_cpu_checkbox else '❌ No'}
        </span>
    </div>
    """, unsafe_allow_html=True)


    # GPU memory info - only display if CUDA device was successfully loaded
    if current_device_status == "cuda":
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
                # Fallback to torch stats if pynvml is not available but CUDA is
                try:
                    allocated = torch.cuda.memory_allocated(0)/1024**2
                    cached = torch.cuda.memory_reserved(0)/1024**2 # Use reserved for better estimate
                    st.markdown(f"""
                    <div style="margin-bottom: 15px;">
                        <span style="font-weight: 600; color: #6200EA;">GPU Memory (torch):</span>
                        <div style="margin-top: 8px;">
                            <span>Allocated: {allocated:.2f} MB</span><br>
                            <span>Reserved: {cached:.2f} MB</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as torch_mem_e:
                    st.markdown(f"""
                    <div style="margin-bottom: 15px; color: #D50000; font-size: 0.9rem;">
                       <i>Could not retrieve torch GPU memory info: {str(torch_mem_e)}</i>
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
        <h3 style="color: #6200EA;">Models</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-bottom: 10px;">
        <span style="font-weight: 600; color: #6200EA;">Embedding:</span>
        <div style="background-color: rgba(98, 0, 234, 0.05); padding: 8px; border-radius: 5px; margin-top: 5px; word-break: break-all;">
            <code>{EMBEDDING_MODEL_NAME}</code>
        </div>
    </div>

    <div style="margin-bottom: 10px;">
        <span style="font-weight: 600; color: #6200EA;">Language Model:</span>
        <div style="background-color: rgba(98, 0, 234, 0.05); padding: 8px; border-radius: 5px; margin-top: 5px; word-break: break-all;">
            <code>{LLM_MODEL_NAME}</code>
        </div>
    </div>

    <div style="margin-bottom: 20px;">
        <span style="font-weight: 600; color: #6200EA;">FAISS Index:</span>
        <div style="background-color: rgba(98, 0, 234, 0.05); padding: 8px; border-radius: 5px; margin-top: 5px;
                   overflow-wrap: break-word; word-wrap: break-word; word-break: break-all;">
            <code>{faiss_index_path}</code>
        </div>
    </div>

    <div style="margin-top: 25px; margin-bottom: 15px;">
         <h3 style="color: #6200EA;">Application Status</h3>
    </div>

    <div style="margin-bottom: 15px;">
        <span style="font-weight: 600; color: #6200EA;">Ready Status:</span>
        <span style="background-color: {'rgba(0, 200, 83, 0.1)' if st.session_state.qa_chain else ('rgba(255, 214, 0, 0.1)' if st.session_state.loading_in_progress else 'rgba(213, 0, 0, 0.1)')};
               padding: 3px 10px; border-radius: 50px; font-size: 0.9rem;
               color: {'#00C853' if st.session_state.qa_chain else ('#FF6D00' if st.session_state.loading_in_progress else '#D50000')};">
            {'✅ Ready' if st.session_state.qa_chain else ('⏳ Loading...' if st.session_state.loading_in_progress else '❌ Not Loaded')}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Help section
    st.markdown("""
    <div style="margin-top: 30px; background-color: #E1F5FE; padding: 20px; border-radius: 10px;">
        <h3 style="color: #0277BD; margin-top: 0;">Quick Tips</h3>
        <ul style="padding-left: 20px; margin-bottom: 0; font-size: 0.95rem; line-height: 1.6;">
            <li>Ask specific questions for more accurate answers</li>
            <li>Include key terms relevant to your research</li>
            <li>For complex topics, break down into multiple queries</li>
            <li>System works best with focused, clear questions</li>
             <li>If loading fails, try checking 'Force CPU' in Configuration (may be slow).</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    # Close the info card div
    st.markdown("</div>", unsafe_allow_html=True)


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
