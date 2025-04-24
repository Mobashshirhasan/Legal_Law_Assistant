# Legal_Law RAG System

A Retrieval-Augmented Generation (RAG) system for legal_law documents, designed to provide accurate legal information based on a corpus of law documents.

![License](https://img.shields.io/github/license/yourusername/saudi-legal-rag-system)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)

 ## 🔍 Overview

This system uses LangChain, OpenAI embeddings, and FAISS vector database to implement a RAG pipeline that can:

- Process legal documents in various formats (PDF, DOCX, TXT)
- Index documents for semantic search
- Detect legal vs. non-legal queries
- Retrieve relevant document sections
- Generate accurate responses with source citations
- Maintain a conversation cache for improved performance

## ✨ Features

- 📄 Multi-format document processing (PDF, DOCX, TXT)
- 🔎 Semantic search with FAISS vector database
- 🧠 LLM-powered query classification
- 💾 Response caching for improved performance
- 🔄 On-demand index reloading
- ✅ Source citation with document filenames
- 📊 Detailed logging and error handling
- 📝 HTML-formatted responses

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- OpenAI API key

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/legal_law-rag-system.git
cd legal_law-rag-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set your OpenAI API key:
```bash
export OPENAI_API_KEY=your_api_key_here
```

4. Configure document folder:
Edit the `DOCUMENTS_FOLDER` variable in `legal_rag.py` to point to your legal documents.

## 🚀 Usage

Run the main script:

```bash
python legal_rag.py
```

### Commands:
- Type your legal question to get an answer
- Type `reload` to refresh the document index
- Type `exit` to quit the application

## 📂 Document Preparation

The system works best with well-structured legal documents. Place your documents in the configured `DOCUMENTS_FOLDER`:

- PDF files should be text-based (not scanned images)
- DOCX files should be properly formatted
- TXT files should use UTF-8 encoding

## ⚙️ Configuration

Key configuration variables in `legal_rag.py`:

```python
DOCUMENTS_FOLDER = r"path/to/your/legal/documents"
FAISS_INDEX_PATH = "faiss_index"
LOG_FILE = "legal_rag_log.txt"
MAX_FILE_SIZE_MB = 50
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVAL_TOP_K = 5
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o"
CACHE_EXPIRY = 3600  # Cache expiry in seconds
```

## 🧩 System Architecture

The system follows a modular architecture:

1. **Document Loading**: Loads and processes documents from the specified folder
2. **Chunking**: Splits documents into manageable chunks for embedding
3. **Vector Store**: Creates and manages FAISS index for semantic search
4. **LangGraph Nodes**:
   - `classify_query`: Determines if the query is legal-related
   - `retrieve_docs`: Retrieves relevant document chunks
   - `generate_response`: Creates the final answer with source citations


## 🔗 Related Resources

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [FAISS Documentation](https://faiss.ai/index.html)
- [OpenAI Documentation](https://platform.openai.com/docs/introduction)
