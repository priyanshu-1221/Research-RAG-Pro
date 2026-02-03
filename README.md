# 🔬 PaperMind AI: Intelligent Research Assistant

PaperMind AI is a local RAG (Retrieval-Augmented Generation) application designed for deep analysis of scientific papers and technical documents. It enables users to have interactive dialogues with complex PDFs while keeping all data private and local.

## 🚀 Key Features
- **Semantic Vector Search:** Transforms PDF text into high-dimensional embeddings for precise information retrieval.
- **Local LLM Integration:** Powered by Llama 3.2 via Ollama for private, offline data processing.
- **Fast Indexing Pipeline:** Optimized document chunking and embedding strategy for rapid setup.
- **Automated Summarization:** Built-in logic to synthesize objectives, methodologies, and results instantly.
- **Exportable Insights:** Generate and download comprehensive text reports from your analysis sessions.

## 🛠️ Technical Stack
- **Framework:** Streamlit (UI), LangChain (Orchestration)
- **Model:** Llama 3.2 (3B)
- **Embeddings:** Nomic-Embed-Text
- **Vector Database:** ChromaDB
- **Document Processing:** PyPDF & Recursive Character Text Splitting

## 💻 Setup Instructions

1. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
