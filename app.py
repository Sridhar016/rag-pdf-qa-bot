import os
import streamlit as st  # Web app framework for creating the UI
from PyPDF2 import PdfReader  # Library to read and extract text from PDF files
from langchain.embeddings import HuggingFaceBgeEmbeddings  # For converting text to vector embeddings
from langchain_community.vectorstores import FAISS  # Vector database for storing and searching embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Splits large text into smaller chunks
from langchain.chains import RetrievalQA  # Chain that retrieves relevant docs and generates answers
from langchain_community.llms import Ollama  # Local LLM integration (Llama model)
from langchain.chains.question_answering import load_qa_chain  # Pre-built QA chain for document analysis
from langchain.schema import Document  # Document schema for LangChain

def extract_text_from_pdf(pdf_path): #This Function Extract all text content from a PDF file.
    
    # Create a PDF reader object
    reader = PdfReader(pdf_path)
    text = ""
    
    # Loop through each page in the PDF and extract text
    for page in reader.pages:
        text += page.extract_text()
    
    return text

def create_faiss_vector_store(text, path="faiss_index"): #This function splits text into chunks, converts them to embeddings, and saves them locally.
    
    # Split the text into smaller chunks for better retrieval
    # chunk_size: Maximum characters per chunk
    # chunk_overlap: Number of characters to overlap between chunks (maintains context)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    
    # Convert text chunks into Document objects (required by LangChain)
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    # Initialize embeddings model (converts text to numerical vectors)
    # Using a lightweight, fast model suitable for semantic search
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create FAISS vector store from documents and their embeddings
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    
    # Save the vector store locally for future use
    vector_store.save_local(path)

def load_faiss_vectore_store(path="faiss_index"): #This function Load a previously saved FAISS vector store from local storage.

    # Initialize the same embeddings model used during creation
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load the vector store from local storage
    # allow_dangerous_deserialization=True: Required for loading FAISS indices
    vector_store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    
    return vector_store

def build_qa_chain(vector_store_path="faiss_index"):
    '''
    Build a complete Question-Answering chain that can retrieve relevant documents
    and generate answers using a local LLM.
    '''
    # Load the vector store containing document embeddings
    vector_store = load_faiss_vectore_store(vector_store_path)
    
    # Create a retriever that can find relevant document chunks based on similarity
    retriever = vector_store.as_retriever()
    
    # Initialize the local LLM (Llama 3.2 model via Ollama)
    llm = Ollama(model="llama3.2")
    
    # Create a QA chain that processes retrieved documents
    # "stuff" strategy: Concatenates all retrieved docs and sends to LLM at once
    qa_chain = load_qa_chain(llm, chain_type="stuff")
    
    # Combine retriever and QA chain into a complete RetrievalQA system
    qa_chain = RetrievalQA(retriever=retriever, combine_documents_chain=qa_chain)
    
    return qa_chain

# Streamlit Web Application UI
st.header("Chat with PDFs")

with st.sidebar:
    st.title("Menu:")
    
    # File uploader widget - accepts only PDF files
    uploaded_file = st.file_uploader("Upload your PDF file and click Submit & Process Button", type="pdf")
    
    # Process button to handle the uploaded PDF
    if st.button("Submit & Process"):
        if uploaded_file is not None:
            # Create directory for storing uploaded files
            os.makedirs("uploaded", exist_ok=True)
            
            # Save uploaded file to local storage
            pdf_path = os.path.join("uploaded", uploaded_file.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Extract text from the uploaded PDF
            with st.spinner("Extracting text from PDF..."):
                text = extract_text_from_pdf(pdf_path)
            
            # Check if text extraction was successful
            if not text.strip():
                st.error("Failed to extract text from PDF. Try a different file.")
            else:
                # Create vector embeddings and store them in FAISS
                with st.spinner("Creating FAISS vector store..."):
                    create_faiss_vector_store(text)
                
                # Build the complete QA chain
                st.info("Initializing chatbot...")
                qa_chain = build_qa_chain()
                
                # Store QA chain in session state for persistence across interactions
                st.session_state['qa_chain'] = qa_chain
                
                st.success("Chatbot is ready!")

# Main chat interface
# Check if QA chain is available (PDF has been processed)
if 'qa_chain' in st.session_state:
    # Input field for user questions
    question = st.text_input("Ask a question about the uploaded PDF:")
    
    if question:
        # Query the document and get answer
        with st.spinner("Querying the document..."):
            # Use the QA chain to find relevant content and generate answer
            answer = st.session_state.qa_chain.run(question)
        
        # Display the answer
        st.success(f"Answer: {answer}")
else:
    # Show instruction message when no PDF is processed
    st.info("Upload and process a PDF to start chatting.")
