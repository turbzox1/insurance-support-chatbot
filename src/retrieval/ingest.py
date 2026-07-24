from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.config.config import (
    PDF_FOLDER,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    VECTORSTORE_PATH
)

# Load PDFs
from src.retrieval.document_loader import (
    load_documents
)

documents = load_documents()


print(f"Loaded {len(documents)} pages")

from collections import Counter

sources = Counter()

for doc in documents:
    sources[doc.metadata["source"]] += 1

print("\nPages loaded from each PDF:\n")

for source, count in sources.items():
    print(f"{source} : {count} pages")

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")

print("\nChunks from Insurance Ombudsman PDF:\n")

for chunk in chunks:
    if "OMBUDSMAN" in chunk.metadata["source"].upper():
        print("=" * 80)
        print(chunk.metadata)
        print(chunk.page_content[:700])

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"}
)

print("Embedding model loaded")

import shutil
import os

if os.path.exists(VECTORSTORE_PATH):
    shutil.rmtree(VECTORSTORE_PATH)
    
# Create vector database
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=VECTORSTORE_PATH
)

print("Vector database created successfully!")