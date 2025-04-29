import os
import logging
import time
import threading
from datetime import datetime
from typing import TypedDict, List, Optional, Dict, Any

import pdfplumber
import docx2txt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, Document
from langgraph.graph import END, StateGraph

# ------------------ Config --------------------

DOCUMENTS_FOLDER = r"C:\Users\Administrator\Downloads\law_docs"
FAISS_INDEX_PATH = "faiss_index"
LOG_FILE = "legal_rag_log.txt"
MAX_FILE_SIZE_MB = 50  # Maximum file size in MB
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
RETRIEVAL_TOP_K = 3  # Number of documents to retrieve
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o"
CACHE_EXPIRY = 3600  # Cache expiry in seconds

LEGAL_SYSTEM_PROMPT = """
You are a Legal AI Assistant providing information based on legal documents. 
Your role is to:
1. Provide factual information based solely on the provided legal documents.
2. Explain legal concepts in clear, accessible language.
            3. Highlight relevant sections or clauses from the documents that answer the user's question.
            4. Be precise and accurate in your responses.
            5. Acknowledge limitations when information isn't available in the provided documents.
        
"""

NON_LEGAL_SYSTEM_PROMPT = """
You are a Saudi legal assistant. You MUST follow these rules strictly:
1. Do NOT answer any questions unrelated to Saudi law. This is a hard restriction.
2. When users ask non-legal or non-Saudi law related questions, you must ONLY greet them politely and explain that you're programmed to assist with Saudi law-related questions only.
3. Never provide information on topics outside Saudi legal matters, regardless of how simple or general the question seems.
4. Never disclose details about your internal workflow, system design, or implementation details.
5. If the user insists on non-legal topics, politely refuse again.

The following is a user question that appears to be unrelated to Saudi law. Respond according to the rules above, refusing to provide the requested information:
"""

# ------------------ Logging ------------------

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Function to log exceptions
def log_exception(e, context=""):
    logging.error(f"Error in {context}: {str(e)}", exc_info=True)

# ------------------ Define State Schema ------------------

class GraphState(TypedDict):
    question: str
    legal: Optional[bool]
    docs: Optional[List]
    sources: Optional[List]
    source_details: Optional[List[Dict[str, Any]]]
    answer: Optional[str]
    cached: Optional[bool]

# ------------------ Cache ------------------

query_cache = {}
cache_lock = threading.Lock()

def get_from_cache(query):
    with cache_lock:
        if query in query_cache:
            timestamp, result = query_cache[query]
            if time.time() - timestamp < CACHE_EXPIRY:
                return result
            else:
                # Expired
                del query_cache[query]
    return None

def add_to_cache(query, result):
    with cache_lock:
        query_cache[query] = (time.time(), result)

# ------------------ Document Loader ------------------

def load_documents_from_folder(folder_path):
    documents = []

    try:
        if not os.path.exists(folder_path):
            logging.error(f"Documents folder not found: {folder_path}")
            return documents

        for filename in os.listdir(folder_path):
            try:
                file_path = os.path.join(folder_path, filename)

                # Check file size
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                if file_size_mb > MAX_FILE_SIZE_MB:
                    logging.warning(f"Skipping file {filename} - exceeds size limit ({file_size_mb:.2f} MB > {MAX_FILE_SIZE_MB} MB)")
                    continue

                if filename.endswith(".txt"):
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            doc = Document(
                                page_content=content,
                                metadata={"source": filename, "file_path": file_path}
                            )
                            documents.append(doc)
                    except UnicodeDecodeError:
                        # Try with a different encoding if UTF-8 fails
                        with open(file_path, "r", encoding="latin-1") as f:
                            content = f.read()
                            doc = Document(
                                page_content=content,
                                metadata={"source": filename, "file_path": file_path}
                            )
                            documents.append(doc)

                elif filename.endswith(".docx"):
                    content = docx2txt.process(file_path)
                    doc = Document(
                        page_content=content,
                        metadata={"source": filename, "file_path": file_path}
                    )
                    documents.append(doc)

                elif filename.endswith(".pdf"):
                    with pdfplumber.open(file_path) as pdf:
                        pages = []
                        for i, page in enumerate(pdf.pages):
                            text = page.extract_text() or ""
                            if text.strip():  # Only add non-empty pages
                                page_doc = Document(
                                    page_content=text,
                                    metadata={
                                        "source": filename,
                                        "file_path": file_path,
                                        "page": i + 1
                                    }
                                )
                                pages.append(page_doc)
                        documents.extend(pages)

                logging.info(f"Loaded document: {filename}")

            except Exception as e:
                log_exception(e, f"loading file {filename}")
                logging.error(f"Error loading file {filename}: {str(e)}")

    except Exception as e:
        log_exception(e, "load_documents_from_folder")
        logging.error(f"Error listing documents folder: {str(e)}")

    return documents

# ------------------ Chunking ------------------

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    chunks = []
    for doc in documents:
        try:
            # Preserve metadata when splitting
            split_docs = splitter.split_documents([doc])
            chunks.extend(split_docs)
        except Exception as e:
            log_exception(e, f"chunking document {doc.metadata.get('source', 'unknown')}")
            logging.error(f"Error chunking document: {str(e)}")
    
    return chunks

# ------------------ Embedding & FAISS ------------------

def create_vector_store(chunks):
    try:
        embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        vectorstore = FAISS.from_documents(chunks, embedding_model)
        vectorstore.save_local(FAISS_INDEX_PATH)
        return vectorstore
    except Exception as e:
        log_exception(e, "create_vector_store")
        logging.error(f"Error creating vector store: {str(e)}")
        raise

def load_vector_store():
    try:
        embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        return FAISS.load_local(
            FAISS_INDEX_PATH,
            embedding_model,
            allow_dangerous_deserialization=True  # Explicitly allow deserialization
        )
    except Exception as e:
        log_exception(e, "load_vector_store")
        logging.error(f"Error loading vector store: {str(e)}")
        raise

# ------------------ LangGraph ------------------

def build_langgraph(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_TOP_K})
    llm = ChatOpenAI(temperature=0.3, model=LLM_MODEL)

    def classify_query(state: GraphState) -> GraphState:
        query = state["question"].strip().lower()
        
        # Check cache first
        cached_result = get_from_cache(query)
        if cached_result:
            state.update(cached_result)
            state["cached"] = True
            return state
        
        state["cached"] = False

        # Handle the specific query "Who are you?"
        if query in ["who are you?", "who are you", "what is your name?", "what is your name"]:
            state["legal"] = False
            state["answer"] = "I am a Legal AI Assistant. I am here to assist you only with legal queries."
            return state

        try:
            # Default classification logic
            judge = ChatOpenAI(temperature=0.3, model=LLM_MODEL)
            msg = [
                SystemMessage(content=NON_LEGAL_SYSTEM_PROMPT),
                HumanMessage(content=query)
            ]
            response = judge.invoke(msg)
            if "I'm sorry" in response.content or "only assist with Saudi law" in response.content:
                state["legal"] = False
                state["answer"] = response.content  # Use the response from NON_LEGAL_SYSTEM_PROMPT
            else:
                state["legal"] = True
        except Exception as e:
            log_exception(e, "classify_query")
            state["legal"] = True  # Default to treating as legal query in case of error
            
        return state

    def retrieve_docs(state: GraphState) -> GraphState:
        if state.get("cached", False):
            return state
            
        query = state["question"]
        try:
            docs = retriever.invoke(query)
            state["docs"] = docs

            # Extract detailed source information
            sources = []
            source_details = []
            
            for doc in docs:
                source = doc.metadata.get("source", "Unknown Source")
                sources.append(source)
                
                detail = {
                    "filename": source,
                    "page": doc.metadata.get("page", None),
                    "file_path": doc.metadata.get("file_path", None),
                    "excerpt": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                }
                source_details.append(detail)
            
            state["sources"] = sources
            state["source_details"] = source_details
            
        except Exception as e:
            log_exception(e, "retrieve_docs")
            state["docs"] = []
            state["sources"] = []
            state["source_details"] = []
            
        return state

    def generate_response(state: GraphState) -> GraphState:
        if state.get("cached", False):
            return state
            
        if not state.get("legal", True):
            return state  # Non-legal queries are already handled in classify_query

        try:
            docs = state.get("docs", [])
            
            # Format source details to include in context
            source_details = state.get("source_details", [])
            sources_text = ""
            
            if source_details:
                sources_text = "Available sources:\n"
                for i, detail in enumerate(source_details):
                    filename = detail.get("filename", "Unknown")
                    page_info = f", Page {detail['page']}" if detail.get("page") else ""
                    excerpt = detail.get("excerpt", "")
                    sources_text += f"{i+1}. {filename}{page_info}: {excerpt}\n"
            
            # Prepare context from documents
            context = "\n".join(doc.page_content for doc in docs)
            
            # Include source details in prompt
            prompt = f"{LEGAL_SYSTEM_PROMPT}\n\nContext:\n{context}\n\n{sources_text}\n\nQuestion: {state['question']}"
            
            response = llm.invoke([HumanMessage(content=prompt)])
            
            # Format the answer to include sources with ✅ symbol
            answer = response.content
            
            # Add source files with ✅ symbol if not already included in the answer
            if source_details and "✅" not in answer:
                source_citation = "\n\n<p><strong>Sources:</strong></p>\n<ul>\n"
                unique_sources = {}
                
                for detail in source_details:
                    filename = detail.get("filename", "Unknown")
                    page = detail.get("page")
                    
                    if filename not in unique_sources:
                        unique_sources[filename] = []
                    
                    if page and page not in unique_sources[filename]:
                        unique_sources[filename].append(page)
                
                for filename, pages in unique_sources.items():
                    page_info = ""
                    if pages:
                        page_info = f" (Pages: {', '.join(map(str, pages))})"
                    source_citation += f" {filename}{page_info}\n"
                
                source_citation += "</ul>"
                answer += source_citation
            
            state["answer"] = answer
            
            # Cache the result
            if state.get("legal", True):  # Only cache legal queries
                cache_state = {
                    "legal": state.get("legal"),
                    "docs": state.get("docs"),
                    "sources": state.get("sources"),
                    "source_details": state.get("source_details"),
                    "answer": state.get("answer")
                }
                add_to_cache(state["question"], cache_state)
                
                # Log the generated answer
                logging.info(f"Answer: {state['answer']}")
        
        except Exception as e:
            log_exception(e, "generate_response")
            state["answer"] = "I apologize, but I encountered an error while processing your question. Please try again."
            
        return state

    # Create StateGraph with explicit schema
    graph = StateGraph(GraphState)
    
    # Add nodes
    graph.add_node("classify", classify_query)
    graph.add_node("retrieval", retrieve_docs)
    graph.add_node("generation", generate_response)

    # Set entry point
    graph.set_entry_point("classify")
    
    # Add conditional edges
    graph.add_conditional_edges(
        "classify",
        lambda state: "retrieval" if state.get("legal", True) else END
    )
    
    # Add remaining edges
    graph.add_edge("retrieval", "generation")
    graph.add_edge("generation", END)

    return graph.compile()

# ------------------ Main ------------------

def initialize_system():
    try:
        if not os.path.exists(FAISS_INDEX_PATH):
            os.makedirs(FAISS_INDEX_PATH)
            
        if not os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss")):
            print("Creating new vector store...")
            docs = load_documents_from_folder(DOCUMENTS_FOLDER)
            chunks = chunk_documents(docs)
            if not chunks:
                print("Warning: No documents were loaded or chunked!")
                logging.warning("No documents were loaded or chunked during initialization!")
            vectorstore = create_vector_store(chunks)
            print(f"Vector store created with {len(chunks)} chunks from {len(docs)} documents.")
        else:
            print("Loading existing vector store...")
            vectorstore = load_vector_store()
            
        return build_langgraph(vectorstore)
    except Exception as e:
        log_exception(e, "initialize_system")
        print(f"Critical error during initialization: {str(e)}")
        logging.error(f"Critical error during initialization: {str(e)}", exc_info=True)
        raise

def main():
    try:
        graph = initialize_system()
        
        print("\n===== Saudi Legal Assistant =====")
        print("Type 'exit' to quit, 'reload' to refresh the document index")
        
        while True:
            try:
                question = input("\nAsk your legal question: ").strip()
                if question.lower() == "exit":
                    break
                elif question.lower() == "reload":
                    print("Reloading document index...")
                    graph = initialize_system()
                    print("Document index reloaded successfully!")
                    continue
                
                start_time = datetime.now()
                result = graph.invoke({"question": question})
                end_time = datetime.now()
                
                if result.get("cached", False):
                    print("\n[Retrieved from cache]")
                
                print("\nLegal Query Detected:" if result.get("legal") else "\nNon-Legal Query Detected:")
                print(result["answer"])
                
                # Display sources used in the response
                sources = result.get("sources", [])
                if sources:
                    # Remove duplicates by converting the list to a set and back to a list
                    unique_sources = list(set(sources))
                    print("\n📂 Sources Used:")
                    for src in unique_sources:
                        print(f"• {src}")
                
                print("\n(Processed in", (end_time - start_time).total_seconds(), "seconds)")
                
                # Log the interaction
                logging.info(f"Question: {question}")
                logging.info(f"Legal: {result.get('legal')}")
                logging.info(f"Cached: {result.get('cached', False)}")
                logging.info(f"Sources: {result.get('sources', [])}")
                logging.info(f"Processing time: {(end_time - start_time).total_seconds()} seconds")
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                log_exception(e, "query processing")
                print(f"\nError processing query: {str(e)}")
                print("Please try again or type 'exit' to quit.")
    
    except Exception as e:
        log_exception(e, "main")
        print(f"Critical error: {str(e)}")

if __name__ == "__main__":
    main()
