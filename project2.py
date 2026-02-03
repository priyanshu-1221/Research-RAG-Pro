import sys
import types
import os
import re
import streamlit as st

# 1. Windows Compatibility Fix
if sys.platform == "win32":
    m = types.ModuleType("pwd")
    sys.modules["pwd"] = m

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma

# --- Page Config ---
st.set_page_config(page_title="PaperMind Turbo", layout="wide")

# --- Optimized Model Loading ---
@st.cache_resource
def load_models():
    # Force LLM to utilize multi-threading and GPU offloading
    llm = OllamaLLM(
        model="llama3.2", 
        temperature=0,
        num_thread=8,  # Optimized for i5 cores
        num_gpu=1      # Encourages usage of Intel Iris Xe
    ) 
    # Switched to Nomic for 3x faster indexing than MXBAI
    embeddings = OllamaEmbeddings(model="nomic-embed-text") 
    return llm, embeddings

llm, embeddings = load_models()

# --- Fast Indexing Logic ---
@st.cache_resource
def process_pdf_turbo(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    for page in pages:
        page.page_content = " ".join(page.page_content.split())

    # Larger chunks = significantly faster embedding time
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, 
        chunk_overlap=150
    )
    chunks = splitter.split_documents(pages)
    
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        collection_name="turbo_res"
    )
    # k=3 reduces the amount of text the LLM must process for each answer
    return vectorstore.as_retriever(search_kwargs={"k": 3}), chunks

# --- Sidebar with Session Control ---
if "messages" not in st.session_state: st.session_state.messages = []
if "retriever" not in st.session_state: st.session_state.retriever = None

with st.sidebar:
    st.header("⚙️ Settings")
    st.info("Mode: GPU Accelerated (Iris Xe)\nEmbedding: Nomic-Turbo")
    
    st.divider()
    
    st.subheader("🧹 Session Control")
    with st.container(border=True):
        if st.button("🆕 Start New Chat"):
            st.session_state.messages = []
            st.session_state.retriever = None
            st.rerun()

        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.toast("Chat cleared!")
            st.rerun()
            
        st.write("---")
        
        if st.session_state.messages:
            history = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages])
            st.download_button("📥 Download Report", history, file_name="research_report.txt")

# --- Main UI ---
st.title("🔬 PaperMind AI: Turbo Edition")

if not st.session_state.retriever:
    uploaded_file = st.file_uploader("Upload Scientific Paper", type="pdf")
    if uploaded_file:
        temp_path = "temp_turbo.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("🚀 Fast Index"):
            with st.spinner("Indexing using Intel Iris Xe..."):
                retriever, chunks = process_pdf_turbo(temp_path)
                st.session_state.retriever = retriever
                st.success(f"Indexed {len(chunks)} segments.")
                st.rerun()
else:
    if st.button("📝 Generate Quick Summary"):
        with st.spinner("Synthesizing..."):
            docs = st.session_state.retriever.get_relevant_documents("abstract methodology results")
            context = "\n".join([d.page_content for d in docs])
            res = llm.invoke(f"Briefly summarize this paper's objective and main finding: {context}")
            st.session_state.messages.append({"role": "assistant", "content": f"**SUMMARY:** {res}"})
            st.rerun()

# --- Chat Interface ---
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if st.session_state.retriever:
    if query := st.chat_input("Ask a technical question..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"): st.markdown(query)

        with st.chat_message("assistant"):
            docs = st.session_state.retriever.get_relevant_documents(query)
            context_text = "\n".join([d.page_content for d in docs])
            with st.spinner(""):
                response = llm.invoke(f"Context: {context_text}\n\nQuestion: {query}")
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})