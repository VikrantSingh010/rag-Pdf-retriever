import streamlit as st
import torch
import os
import time
import hashlib
import re
import pandas as pd
import base64
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
import tempfile
import json

# LangChain imports
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.schema import Document
from dotenv import load_dotenv
st.set_option("server.fileWatcherType", "none")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


load_dotenv()


CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)
CHROMA_DIR = Path("./chroma_db")
CHROMA_DIR.mkdir(exist_ok=True)
SESSION_DIR = Path("./sessions")
SESSION_DIR.mkdir(exist_ok=True)
TEMP_DIR = Path("./temp_files")
TEMP_DIR.mkdir(exist_ok=True)


def get_device():
    """Determine the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"  # For Apple Silicon
    else:
        return "cpu"

def create_session_id(files: List) -> str:
    """Create a unique session ID from uploaded files."""
    hash_obj = hashlib.sha256()
    for file in files:
        hash_obj.update(file.getvalue())
    return hash_obj.hexdigest()[:12]

def sanitize_text(text: str) -> str:

    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E\n\t]', '', text)
    return text.strip()

def save_session_data(session_id: str, data: Dict) -> None:
    """Save session data to disk."""
    session_file = SESSION_DIR / f"{session_id}.json"
    try:
        with open(session_file, 'w') as f:
            json.dump(data, f)
        logger.info(f"Session data saved: {session_id}")
    except Exception as e:
        logger.error(f"Error saving session data: {e}")

def load_session_data(session_id: str) -> Optional[Dict]:
    """Load session data from disk."""
    session_file = SESSION_DIR / f"{session_id}.json"
    if session_file.exists():
        try:
            with open(session_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading session data: {e}")
    return None

def process_pdf(file_data: bytes, filename: str, ocr_enabled: bool = False) -> List[Document]:
    """Process a single PDF file and return documents."""
    temp_path = TEMP_DIR / f"temp_{filename}"
    
    try:

        with open(temp_path, "wb") as f:
            f.write(file_data)

        loader = PyPDFLoader(str(temp_path))
        documents = loader.load()
        
        # TODO:
        for doc in documents:
            doc.page_content = sanitize_text(doc.page_content)
            doc.metadata = {
                "source": filename,
                "page": doc.metadata.get("page", 0),
                "total_pages": len(documents)
            }
            
        return documents
    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")
        raise
    finally:
        if temp_path.exists():
            os.remove(temp_path)

def get_pdf_metadata(filename: str, num_pages: int) -> Dict:
    return {
        "filename": filename,
        "pages": num_pages,
        "processed_time": time.strftime("%Y-%m-%d %H:%M:%S")
    }

def get_chat_history(session_id: str) -> BaseChatMessageHistory:
    """Retrieve or create chat history for session."""
    if session_id not in st.session_state.conversations:
        st.session_state.conversations[session_id] = ChatMessageHistory()
    return st.session_state.conversations[session_id]

def clear_chat_history(session_id: str) -> None:
    """Clear chat history for a session."""
    # Clear UI messages
    st.session_state.messages = []
    
    # Clear LangChain conversation history
    if session_id in st.session_state.conversations:
        st.session_state.conversations[session_id] = ChatMessageHistory()
        logger.info(f"Chat history cleared for session: {session_id}")

def display_pdf_preview(file_bytes: bytes) -> None:
    """Display a PDF preview in the Streamlit app."""
    base64_pdf = base64.b64encode(file_bytes).decode('utf-8')
    pdf_display = f"""
        <iframe 
            src="data:application/pdf;base64,{base64_pdf}" 
            width="100%" 
            height="500px" 
            type="application/pdf">
        </iframe>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)

def create_text_chunks(documents: List[Document], chunk_method: str, 
                      chunk_size: int, chunk_overlap: int) -> List[Document]:
    """Create text chunks based on selected method."""
    if chunk_method == "Recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]
        )
    else: 
        splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    return splitter.split_documents(documents)

def initialize_vectorstore(chunks: List[Document], embeddings, session_id: str) -> Chroma:
    collection_name = f"collection_{session_id}"
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "cosine"}
    )

def parse_sources_from_response(text: str) -> List[str]:
    sources = re.findall(r'\[Document(\d+)\]', text)
    return [int(src) for src in sources]


def render_sidebar() -> Tuple[str, str, Dict]:
    st.sidebar.title("⚙️ System Configuration")

    with st.sidebar.expander("🔑 API Keys", expanded=True):
        api_key_type = st.radio("Select API Provider:", ["Groq", "OpenAI"])
        
        if api_key_type == "Groq":
            api_key = st.text_input("Groq API Key:", type="password")
            st.session_state.using_groq = True
            st.session_state.using_openai = False
        else:
            api_key = st.text_input("OpenAI API Key:", type="password") 
            st.session_state.using_groq = False
            st.session_state.using_openai = True
    
    # Model Configuration
    with st.sidebar.expander("🤖 Model Configuration", expanded=True):
        if st.session_state.using_groq:
            model_options = [
                "gemma2-9b-it", 
                "llama3-8b-8192", 
                "deepseek-r1-distill-llama-70b", 
                "llama-3.1-8b-instant"
            ]
            default_index = 1
        else:
            model_options = [
                "gpt-3.5-turbo",
                "gpt-4o",
                "gpt-4.1"
            ]
            default_index = 0
            
        model_name = st.selectbox(
            "LLM Model:",
            model_options,
            index=default_index,
            help="Choose the AI model based on your needs"
        )
        
        temperature = st.slider(
            "Temperature:", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.3, 
            step=0.1,
            help="Lower values = more deterministic, higher = more creative"
        )
    
    # Document Processing
    with st.sidebar.expander("📄 Document Processing", expanded=False):
        chunk_method = st.radio(
            "Chunking Method:",
            ["Recursive", "Character"],
            index=0,
            help="Recursive uses intelligent splitting, Character is simpler"
        )
        
        chunk_size = st.slider(
            "Chunk Size (tokens):", 
            min_value=256, 
            max_value=4096, 
            value=1500, 
            step=128
        )
        
        chunk_overlap = st.slider(
            "Chunk Overlap (tokens):", 
            min_value=0, 
            max_value=512, 
            value=200, 
            step=32
        )
        
        enable_ocr = st.toggle(
            "Enable OCR for Scanned PDFs",
            value=False,
            help="Use OCR to extract text from scanned documents (slower)"
        )
    
    # Retrieval Settings
    with st.sidebar.expander("🔍 Retrieval Settings", expanded=False):
        top_k = st.slider(
            "Number of Documents to Retrieve (k):", 
            min_value=1, 
            max_value=10, 
            value=4, 
            step=1
        )
        
        embedding_model = st.selectbox(
            "Embedding Model:",
            ["all-MiniLM-L6-v2", "e5-small-v2"],
            index=0
        )
    
    # Save session settings
    processing_settings = {
        "chunk_method": chunk_method,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "enable_ocr": enable_ocr,
        "top_k": top_k,
        "model": model_name,
        "temperature": temperature,
        "embedding_model": embedding_model
    }
    
    return api_key, model_name, processing_settings

def render_about():
    """Render about section."""
    st.sidebar.divider()
    with st.sidebar.expander("ℹ️ About This App"):
        st.markdown("""
        **PDF RAG System**
        
        This application allows you to:
        - Upload multiple PDFs of any size
        - Process and analyze document content
        - Ask questions about the documents
        - Get AI-powered answers with source citations
        
        Built with Streamlit, LangChain, and Vector Databases.
        """)

def render_upload_section():
    st.subheader("📤 Upload Documents")
    
    # Session ID input/restoration
    col1, col2 = st.columns([3, 1])
    with col1:
        saved_session_id = st.text_input(
            "Restore Session ID (optional):",
            help="Enter a previous session ID to restore your documents"
        )
    with col2:
        if saved_session_id and st.button("🔄 Restore"):
            session_data = load_session_data(saved_session_id)
            if session_data:
                st.session_state.session_id = saved_session_id
                st.session_state.processed_docs = session_data.get("processed_docs", [])
                st.session_state.has_vectorstore = True
                st.success(f"Session restored: {saved_session_id}")
                st.rerun()
            else:
                st.error("Session not found or invalid")

    uploaded_files = st.file_uploader(
        "Upload PDF Documents:", 
        type="pdf", 
        accept_multiple_files=True,
        help="Upload multiple PDFs (research papers, manuals, etc.)"
    )
    
    # Process button with progress tracking
    if uploaded_files:
        process_col1, process_col2 = st.columns([3, 1])
        with process_col1:
            settings_display = f"""
            • Chunk Size: {st.session_state.processing_settings['chunk_size']} tokens
            • Chunk Overlap: {st.session_state.processing_settings['chunk_overlap']} tokens
            • Method: {st.session_state.processing_settings['chunk_method']}
            """
            st.markdown(settings_display)
        with process_col2:
            if st.button("🔍 Process Documents", type="primary"):
                return process_documents(uploaded_files)
    
    # Display previously processed documents if any
    if hasattr(st.session_state, 'processed_docs') and st.session_state.processed_docs:
        st.success(f"Session ID: {st.session_state.session_id}")
        render_processed_documents()
    
    return False

def render_processed_documents():
    """Display processed documents information."""
    st.subheader("📚 Processed Documents")
    
    # Create a DataFrame for better display
    if hasattr(st.session_state, 'processed_docs') and st.session_state.processed_docs:
        doc_data = []
        for doc in st.session_state.processed_docs:
            doc_data.append({
                "Filename": doc["filename"],
                "Pages": doc["pages"],
                "Processed": doc["processed_time"]
            })
        
        df = pd.DataFrame(doc_data)
        st.dataframe(df, use_container_width=True)
        
        # Copy session ID button
        st.button(
            f"📋 Copy Session ID: {st.session_state.session_id}",
            help="Click to copy your session ID to clipboard",
            on_click=lambda: st.write(
                f"""<script>navigator.clipboard.writeText('{st.session_state.session_id}');</script>""", 
                unsafe_allow_html=True
            )
        )
    else:
        st.info("No documents processed yet")

def render_chat_interface():

    st.divider()
    st.subheader("🤖 Ask Questions About Your Documents")
    
    # Initialize chat history in the UI
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Add clear chat button
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🗑️ Clear Chat", type="secondary"):
            clear_chat_history(st.session_state.session_id)
            st.rerun()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                st.caption(f"Sources: {', '.join(message['sources'])}")
    
    # Chat input
    user_input = st.chat_input(
        "Ask a question about your documents...",
        disabled=not hasattr(st.session_state, 'has_vectorstore') or not st.session_state.has_vectorstore
    )
    
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_placeholder = st.empty()
                source_placeholder = st.empty()
                
                try:
                    response = query_documents(user_input)
                    response_placeholder.markdown(response["answer"])
                    
                    # Display sources if available
                    if response.get("context"):
                        sources = [f"Document {i+1}" for i in range(len(response["context"]))]
                        source_placeholder.caption(f"Sources: {', '.join(sources)}")
                        
                        # Add sources to expandable sections
                        for i, doc in enumerate(response.get("context", [])):
                            with st.expander(f"📄 Source {i+1}: {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', '?')})"):
                                st.markdown(doc.page_content)
                                
                        # Save response with sources to history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response["answer"],
                            "sources": sources
                        })
                    else:
                        # Save response without sources
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response["answer"]
                        })
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    response_placeholder.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
                    logger.error(f"Query error: {e}")

def process_documents(uploaded_files):
    session_id = create_session_id(uploaded_files)
    st.session_state.session_id = session_id

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
  
        status_text.text("Step 1/3: Extracting text from PDFs...")
        all_docs = []
        processed_docs_meta = []
   
        with ThreadPoolExecutor() as executor:
            futures = []
            for file in uploaded_files:
                futures.append(
                    executor.submit(
                        process_pdf, 
                        file.getvalue(), 
                        file.name, 
                        st.session_state.processing_settings["enable_ocr"]
                    )
                )

            for i, future in enumerate(futures):
                try:
                    docs = future.result()
                    if docs:
                        all_docs.extend(docs)
         
                        file_name = uploaded_files[i].name
                        num_pages = max([d.metadata.get("page", 0) for d in docs]) + 1
                        processed_docs_meta.append(
                            get_pdf_metadata(file_name, num_pages)
                        )
                    progress_bar.progress((i + 1) / len(futures) * 0.4)
                except Exception as e:
                    st.error(f"Failed to process {uploaded_files[i].name}: {e}")
        
        if not all_docs:
            st.error("No text could be extracted from the uploaded files.")
            return False
       
        st.session_state.processed_docs = processed_docs_meta
       
        status_text.text("Step 2/3: Creating text chunks...")
        chunk_settings = st.session_state.processing_settings
        chunks = create_text_chunks(
            all_docs, 
            chunk_settings["chunk_method"],
            chunk_settings["chunk_size"],
            chunk_settings["chunk_overlap"]
        )
        progress_bar.progress(0.7)  # 70% complete
        
        # Step 3: Build vector store
        status_text.text("Step 3/3: Building vector database...")
        
        # Initialize embeddings
        device = get_device()
        embeddings = HuggingFaceEmbeddings(
            model_name=chunk_settings["embedding_model"],
            model_kwargs={"device": device}
        )
        
        # Create vector store
        initialize_vectorstore(chunks, embeddings, session_id)
        progress_bar.progress(1.0)
        
        # Save session data for future restoration
        session_data = {
            "processed_docs": processed_docs_meta,
            "settings": st.session_state.processing_settings,
            "creation_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        save_session_data(session_id, session_data)
        
        # Update session state
        st.session_state.has_vectorstore = True
        
        # Success message
        status_text.success(f"✅ Processing complete! {len(chunks)} chunks created from {len(all_docs)} pages.")
        st.session_state.session_ready = True
        return True
        
    except Exception as e:
        status_text.error(f"Error during processing: {str(e)}")
        logger.error(f"Document processing error: {e}")
        return False

def initialize_rag_chain():
    """Initialize the RAG chain for document querying."""
    if not hasattr(st.session_state, 'rag_chain') or st.session_state.rag_chain is None:
        try:
            # Get session settings
            settings = st.session_state.processing_settings
            session_id = st.session_state.session_id
            
            # Initialize LLM based on provider
            if st.session_state.using_groq and st.session_state.groq_api_key:
                llm = ChatGroq(
                    groq_api_key=st.session_state.groq_api_key, 
                    model_name=settings["model"],
                    temperature=settings["temperature"]
                )
            elif st.session_state.using_openai and st.session_state.openai_api_key:
                llm = ChatOpenAI(
                    api_key=st.session_state.openai_api_key,
                    model_name=settings["model"],
                    temperature=settings["temperature"]
                )
            else:
                raise ValueError("No valid API key provided")
            
            # Initialize embeddings
            device = get_device()
            embeddings = HuggingFaceEmbeddings(
                model_name=settings["embedding_model"],
                model_kwargs={"device": device}
            )
            
            # Load vector store
            collection_name = f"collection_{session_id}"
            vectorstore = Chroma(
                persist_directory=str(CHROMA_DIR),
                embedding_function=embeddings,
                collection_name=collection_name
            )
            
            retriever = vectorstore.as_retriever(
                search_kwargs={
                    "k": max(settings["top_k"] * 2, 8),
                    "k": max(settings["top_k"] * 3, 15) 
                }
            )
           
            context_prompt = ChatPromptTemplate.from_messages([
                ("system", """Analyze the conversation history and the user's question to create an optimized search query.
                This is a multi-document search system with information spread across different PDFs.
                
                Focus on:
                1. Key entities, topics, and technical terms from the question
                2. Specific document references if mentioned
                3. Creating a standalone search query that will retrieve the most relevant information
                4. IMPORTANT: Keep your query broad enough to search across multiple documents
                5. IMPORTANT: Do not restrict the search to only previously discussed documents
                
                Output ONLY the optimized search query without explanation."""),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            
            history_aware_retriever = create_history_aware_retriever(
                llm, retriever, context_prompt
            )
            
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert document analyst with deep knowledge in multiple domains.
                
                Answer the question based ONLY on the provided context, which may come from multiple different PDF documents.
                Follow these guidelines:
                
                1. Provide direct, concise answers from the context
                2. Cite your sources using [Document1], [Document2], etc. notation
                3. When information comes from different documents, clearly indicate which parts come from which sources
                4. If the context doesn't contain the answer, say "I cannot find information about this in the provided documents."
                5. Maintain technical accuracy - use terms precisely as they appear in the documents
                6. For complex questions, use numbered lists or bullet points for clarity
                7. When citing numerical data, quote the exact figures from the documents
                8. IMPORTANT: Give equal consideration to all document sources, not just the ones discussed in previous questions
                
                Context:
                {context}"""),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
         
            answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, answer_chain)
            
            # Make conversation-aware
            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_chat_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )
            
            # Store in session state
            st.session_state.rag_chain = conversational_rag_chain
            logger.info("RAG chain initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RAG chain: {e}")
            raise

def process_documents(uploaded_files):
    """Process uploaded documents and build the vector store."""
    session_id = create_session_id(uploaded_files)
    st.session_state.session_id = session_id
    
    # Configure progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Process PDFs
        status_text.text("Step 1/3: Extracting text from PDFs...")
        all_docs = []
        processed_docs_meta = []
        
        # Process files in parallel for better performance
        with ThreadPoolExecutor() as executor:
            futures = []
            for file in uploaded_files:
                futures.append(
                    executor.submit(
                        process_pdf, 
                        file.getvalue(), 
                        file.name, 
                        st.session_state.processing_settings["enable_ocr"]
                    )
                )
            
            # Collect results as they complete
            for i, future in enumerate(futures):
                try:
                    docs = future.result()
                    if docs:
                        # Improve metadata to track document sources better
                        for doc in docs:
                            doc.metadata["doc_id"] = i + 1
                            doc.metadata["doc_name"] = uploaded_files[i].name
                            doc.metadata["total_docs"] = len(uploaded_files)
                        
                        all_docs.extend(docs)
                        # Extract metadata from the first document
                        file_name = uploaded_files[i].name
                        num_pages = max([d.metadata.get("page", 0) for d in docs]) + 1
                        processed_docs_meta.append(
                            get_pdf_metadata(file_name, num_pages)
                        )
                    progress_bar.progress((i + 1) / len(futures) * 0.4)  # First 40% of progress
                except Exception as e:
                    st.error(f"Failed to process {uploaded_files[i].name}: {e}")
        
        if not all_docs:
            st.error("No text could be extracted from the uploaded files.")
            return False
        
        # Save document metadata to session state
        st.session_state.processed_docs = processed_docs_meta
        
        # Step 2: Create text chunks
        status_text.text("Step 2/3: Creating text chunks...")
        chunk_settings = st.session_state.processing_settings
        
        chunk_size = min(chunk_settings["chunk_size"], 1000)  
        chunk_overlap = max(chunk_settings["chunk_overlap"], int(chunk_size * 0.2))  
        
        chunks = create_text_chunks(
            all_docs, 
            chunk_settings["chunk_method"],
            chunk_size,
            chunk_overlap
        )
        
   
        for chunk in chunks:
            if "doc_name" not in chunk.metadata:
                chunk.metadata["doc_name"] = chunk.metadata.get("source", "Unknown")
        
        progress_bar.progress(0.7) 
        
    
        status_text.text("Step 3/3: Building vector database...")
     
        device = get_device()
        embeddings = HuggingFaceEmbeddings(
            model_name=chunk_settings["embedding_model"],
            model_kwargs={"device": device}
        )
     
        initialize_vectorstore(chunks, embeddings, session_id)
        progress_bar.progress(1.0)
        
        # Save session data for future restoration
        session_data = {
            "processed_docs": processed_docs_meta,
            "settings": st.session_state.processing_settings,
            "creation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_documents": len(uploaded_files),
            "num_chunks": len(chunks)
        }
        save_session_data(session_id, session_data)
        
        st.session_state.has_vectorstore = True
        
        status_text.success(f"✅ Processing complete! {len(chunks)} chunks created from {len(all_docs)} pages across {len(uploaded_files)} documents.")
        st.session_state.session_ready = True
        return True
        
    except Exception as e:
        status_text.error(f"Error during processing: {str(e)}")
        logger.error(f"Document processing error: {e}")
        return False

def render_processed_documents():

    st.subheader("📚 Processed Documents")
 
    if hasattr(st.session_state, 'processed_docs') and st.session_state.processed_docs:
        doc_data = []
        for i, doc in enumerate(st.session_state.processed_docs):
            doc_data.append({
                "Doc ID": i + 1, 
                "Filename": doc["filename"],
                "Pages": doc["pages"],
                "Processed": doc["processed_time"]
            })
        
        df = pd.DataFrame(doc_data)
        st.dataframe(df, use_container_width=True)
   
        st.button(
            f"📋 Copy Session ID: {st.session_state.session_id}",
            help="Click to copy your session ID to clipboard",
            on_click=lambda: st.write(
                f"""<script>navigator.clipboard.writeText('{st.session_state.session_id}');</script>""", 
                unsafe_allow_html=True
            )
        )
    else:
        st.info("No documents processed yet")

def query_documents(question: str) -> Dict:

    if not hasattr(st.session_state, 'rag_chain') or st.session_state.rag_chain is None:
        initialize_rag_chain()
    
    if hasattr(st.session_state, 'reset_needed') and st.session_state.reset_needed:
        st.session_state.rag_chain = None
        initialize_rag_chain()
        st.session_state.reset_needed = False
   
    try:
        response = st.session_state.rag_chain.invoke(
            {"input": question},
            config={
                "session_id": st.session_state.session_id,
                "configurable": {
                    "retrieval": {
                        "search_type": "similarity", 
                    }
                }
            }
        )

        if "cannot find information" in response.get("answer", "").lower():
            st.session_state.reset_needed = True
        
        return response
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise
def main():

    st.set_page_config(
        page_title="Enhanced PDF RAG System",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("📚 Enhanced PDF Analysis System")
    st.markdown("""
    Upload PDF documents, process them with AI, and get intelligent answers to your questions.
    """)
   
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
    if "session_ready" not in st.session_state:
        st.session_state.session_ready = False
    if "has_vectorstore" not in st.session_state:
        st.session_state.has_vectorstore = False
    
    # Render sidebar
    api_key, model_name, processing_settings = render_sidebar()
    st.session_state.processing_settings = processing_settings
    
    # Store API keys
    if st.session_state.using_groq:
        st.session_state.groq_api_key = api_key
    else:
        st.session_state.openai_api_key = api_key
    
    # About section
    render_about()
    
    # Main content area with tabs
    tab1, tab2 = st.tabs(["📄 Documents", "💬 Chat"])
    
    with tab1:
        documents_processed = render_upload_section()
        if documents_processed:
            st.rerun() 
    
    with tab2:
        if hasattr(st.session_state, 'has_vectorstore') and st.session_state.has_vectorstore:
            render_chat_interface()
        else:
            st.info("👈 Please upload and process documents first")

if __name__ == "__main__":
    main()
