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
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# Suppress Hugging Face logging messages that aren't warnings or errors
hf_logging.set_verbosity_warning()
logging.basicConfig(level=logging.WARNING) # Set root logger level to warning

# ... (rest of your imports and initial setup like set_page_config, CSS, etc.) ...

# --- Initialize session state variables if they don't exist ---
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'llm_pipeline' not in st.session_state:
    st.session_state.llm_pipeline = None
if 'used_device' not in st.session_state:
    st.session_state.used_device = "Not Loaded Yet" # Track the device actually used

# --- Header Area with Logo and Title ---
# ... (Your header code) ...

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
        help="Check this to load models on CPU even if a GPU is available. Useful for debugging or if GPU VRAM is insufficient."
    )
    force_cpu = st.session_state.force_cpu_checkbox

    col1, col2 = st.columns([3, 1])
    with col2:
        load_system_button = st.button("Load System", use_container_width=True, key="load_system_button")

    if load_system_button:
        # Reset status on button click to indicate loading is starting
        st.session_state.qa_chain = None
        st.session_state.model_loaded = False
        st.session_state.llm_pipeline = None
        st.session_state.used_device = "Loading..."
        st.rerun() # Rerun to show waiting status immediately

# Logic that runs *after* the button is clicked and state is updated
if st.session_state.used_device == "Loading...": # Check if loading was initiated
    with st.spinner("Initializing knowledge system..."):
        try:
            # Determine the device - respect force_cpu checkbox
            determined_device = "cpu"
            if not force_cpu and torch.cuda.is_available():
                determined_device = "cuda"
                st.info(f"GPU available: {torch.cuda.get_device_name(0)}. Attempting to use CUDA.")
            else:
                st.info("Loading models on CPU (GPU not available or Force CPU selected).")

            st.session_state.used_device = determined_device # Update session state with the chosen device

            # --- Load Embedding Model Manually ---
            st.info(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' onto {determined_device}...")
            try:
                # Use sentence_transformers directly to control device loading
                # Ensure correct imports are at the top: `import sentence_transformers`
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

            db = FAISS.load_local(
                faiss_index_path,
                embeddings, # Use the correctly loaded embeddings object
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

            # Load the LLM if it hasn't been loaded yet (or if we are reloading on a different device)
            # This condition should be 'True' if we initiated loading
            if not st.session_state.model_loaded or st.session_state.used_device != determined_device:
                 if st.session_state.model_loaded:
                      st.warning(f"Switching device from {st.session_state.used_device} to {determined_device}. Reloading LLM...")
                 else:
                      st.info(f"Loading language model '{LLM_MODEL_NAME}' on {determined_device}...")

                 with st.spinner(f"Loading language model '{LLM_MODEL_NAME}' on {determined_device} (this may take a few minutes)..."):

                    # LLM Loading logic - now uses the 'determined_device' variable consistently
                    bnb_config = None # Default to no quantization
                    if determined_device == "cuda":
                        try:
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
                            # Check if bitsandbytes is actually available and works
                            import bitsandbytes
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
                                # but for single GPU FP load, explicit .to(device) might be simpler or necessary
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
                            # Explicitly setting the main device for the pipeline is sometimes needed
                            # depending on how pipeline handles device_map="auto"
                            # pipeline_device = 0 if determined_device == "cuda" else -1


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
                                # Resize model embeddings ONLY if adding a new token
                                # This line should only be here if add_special_tokens added '[PAD]'
                                model.resize_token_embeddings(len(tokenizer))
                                # Retrieve the new pad_token_id after adding it
                                tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')
                                st.warning("Tokenizer missing pad_token, added a new PAD token and resized model embeddings.")
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
                            # Passing device explicitly can sometimes help,
                            # but device_map="auto" on the model is usually sufficient for pipeline
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
                             st.error("This looks like a CUDA/GPU Out of Memory error. The model or embeddings might be too large for the available VRAM. Try using the 'Force CPU' option or running on a machine with more VRAM.")
                        if "bitsandbytes" in str(e):
                             st.error("BitsAndBytes error during quantization. Ensure `bitsandbytes` is installed correctly for your CUDA version and GPU.")
                        raise # Re-raise to fail gracefully

            else:
                # Model was already loaded on the correct device
                llm = HuggingFacePipeline(pipeline=st.session_state.llm_pipeline)
                st.info(f"Language model was already loaded on {st.session_state.used_device}.")

            # Define QA prompt (assuming this is correct)
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

            st.session_state.used_device = determined_device # Final confirmation of device used
            st.success("Knowledge system fully initialized and ready!")
            st.rerun() # Rerun to update status indicators

        except Exception as e:
            # Catch any exceptions that weren't re-raised earlier and display them
            st.error(f"An unexpected error occurred during system loading: {e}")
            st.exception(e) # Display full traceback for any remaining errors
            # Reset state on error
            st.session_state.qa_chain = None
            st.session_state.model_loaded = False
            st.session_state.llm_pipeline = None
            st.session_state.used_device = "Failed"
            # No rerun here, let the error message stay visible

# ... (rest of your code for tabs, sidebar, footer) ...

# Update sidebar device display using the session state variable
# Find the existing sidebar section for device display
# Replace or modify this part in your sidebar block:
with st.sidebar:
     # ... (your existing sidebar markdown up to the device display) ...
     st.markdown(f"""
     <div style="margin-bottom: 15px;">
         <span style="font-weight: 600; color: #6200EA;">Processing Device:</span>
         <span style="background-color: {'rgba(0, 200, 83, 0.1)' if st.session_state.used_device == 'cuda' else ('rgba(255, 214, 0, 0.1)' if st.session_state.used_device == 'cpu' else 'rgba(213, 0, 0, 0.1)')};
                padding: 3px 10px; border-radius: 50px; font-size: 0.9rem; color: {'#00C853' if st.session_state.used_device == 'cuda' else ('#FF6D00' if st.session_state.used_device == 'cpu' else '#D50000')};">
             {'⚡ CUDA' if st.session_state.used_device == 'cuda' else ('💻 CPU' if st.session_state.used_device == 'cpu' else '❌ Failed')}
         </span>
         {'<span style="font-size: 0.8rem; color: #FFD600;">(Forced)</span>' if force_cpu and st.session_state.used_device == 'cpu' else ''}
     </div>
     """, unsafe_allow_html=True)

     # Add sidebar status for Force CPU setting
     st.markdown(f"""
     <div style="margin-bottom: 15px;">
         <span style="font-weight: 600; color: #6200EA;">Force CPU Active:</span>
         <span style="background-color: {'rgba(0, 200, 83, 0.1)' if force_cpu else 'rgba(213, 0, 0, 0.1)'};
                padding: 3px 10px; border-radius: 50px; font-size: 0.9rem; color: {'#00C853' if force_cpu else '#D50000'};">
             {'✅ Yes' if force_cpu else '❌ No'}
         </span>
     </div>
     """, unsafe_allow_html=True)

     # ... (rest of your sidebar code like GPU memory info, models info, etc.) ...
