from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import (
    PDF_FOLDER,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    VECTORSTORE_PATH
)

# Load PDFs
loader = PyPDFDirectoryLoader(PDF_FOLDER)
documents = loader.load()

print(f"Loaded {len(documents)} pages")

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"}
)

print("Embedding model loaded")

# Create vector database
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=VECTORSTORE_PATH
)

print("Vector database created successfully!")