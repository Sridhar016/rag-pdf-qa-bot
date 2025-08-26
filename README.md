# RAG PDF Chat Application

A powerful Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and have interactive conversations with their content using AI. Built with Streamlit for the web interface and powered by LangChain, FAISS vector database, and Ollama's Llama 3.2 model.

## Features

- **PDF Text Extraction**: Seamlessly extract text content from uploaded PDF files
- **Intelligent Document Chunking**: Split large documents into manageable chunks with overlap for context preservation
- **Vector Embeddings**: Convert text chunks into high-dimensional vectors using HuggingFace BGE embeddings
- **FAISS Vector Database**: Efficient similarity search and retrieval of relevant document sections
- **AI-Powered Q&A**: Interactive chat interface powered by Ollama's Llama 3.2 model
- **Persistent Storage**: Save and load vector stores for reuse across sessions
- **User-Friendly Interface**: Clean and intuitive Streamlit web interface

## Technology Stack

- **Frontend**: Streamlit
- **PDF Processing**: PyPDF2
- **Text Processing**: LangChain
- **Embeddings**: HuggingFace BGE (sentence-transformers/all-MiniLM-L6-v2)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Language Model**: Ollama Llama 3.2
- **Text Splitting**: RecursiveCharacterTextSplitter

## Prerequisites

Before running this application, ensure you have:

1. **Python 3.8 or higher** installed
2. **Ollama** installed and running with the Llama 3.2 model
   ```bash
      # Install Ollama (visit https://ollama.com for installation instructions)
      # Pull the Llama 3.2 model
      ollama pull llama3.2
   ```

## Installation

1. **Clone the repository**
   ```bash
      git clone <repository-url>
      cd rag-pdf-chat
   ```

2. **Create a virtual environment**
   ```bash
      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
      pip install -r requirements.txt
   ```

## Dependencies

The application requires the following Python packages:

```
streamlit
PyPDF2
langchain
sentence-transformers
faiss-cpu
Ollama
langchain-community
```

Create a `requirements.txt` file with the above dependencies for easy installation.

## Usage

1. **Start the Streamlit application**
   ```bash
      streamlit run app.py
   ```

2. **Open your web browser** and navigate to the provided local URL (typically `http://localhost:8501`)

3. **Upload a PDF file** using the file uploader in the sidebar

4. **Click "Submit & Process"** to:
      - Extract text from the PDF
      - Create embeddings and build the vector store
      - Initialize the QA chain

5. **Start chatting** by typing questions about your PDF content in the text input field

## Project Structure

```
rag-pdf-chat/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── uploaded/             # Directory for uploaded PDF files
├── faiss_index/          # FAISS vector store files
└── README.md             # Project documentation
```

## How It Works

1. **PDF Processing**: The application extracts text from uploaded PDF files using PyPDF2
2. **Text Chunking**: Large documents are split into smaller, overlapping chunks (1000 characters with 200 character overlap)
3. **Embedding Generation**: Each chunk is converted to vector embeddings using HuggingFace's sentence transformer model
4. **Vector Storage**: Embeddings are stored in a FAISS vector database for efficient retrieval
5. **Question Answering**: When a user asks a question:
      - The question is embedded using the same model
      - Similar document chunks are retrieved from FAISS
      - Retrieved context is sent to Llama 3.2 for answer generation

## Configuration

### Embedding Model
The application uses `sentence-transformers/all-MiniLM-L6-v2` for generating embeddings. You can modify this in the code:

```python
embeddings = HuggingFaceBgeEmbeddings(model_name="your-preferred-model")
```

### Text Chunking Parameters
Adjust chunk size and overlap in the `create_faiss_vector_store` function:

```python
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
```

### Language Model
The application uses Ollama's Llama 3.2. Ensure the model is available:

```python
llm = Ollama(model="llama3.2")
```

## Troubleshooting

### Common Issues

1. **"Failed to extract text from PDF"**
      - Ensure the PDF contains extractable text (not scanned images)
      - Try a different PDF file

2. **Ollama Connection Issues**
      - Verify Ollama is running: `ollama serve`
      - Confirm Llama 3.2 model is installed: `ollama list`

3. **Memory Issues with Large PDFs**
      - Reduce chunk size in the text splitter
      - Process smaller PDF files

## Acknowledgments

- [LangChain](https://langchain.com/) for the RAG framework
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search
- [Ollama](https://ollama.ai/) for local LLM inference
- [Streamlit](https://streamlit.io/) for the web interface
- [HuggingFace](https://huggingface.co/) for the embedding models
