import os
import shutil

from document_loader import load_documents

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_chroma import Chroma

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    VECTORSTORE_PATH
)


class KnowledgeManager:

    def __init__(self):

        self.embedding_model = (
            HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"}
            )
        )

    def rebuild_knowledge_base(self):

        print(
            "\nRebuilding Knowledge Base..."
        )

        # Load all supported documents
        documents = load_documents()

        print(
            f"Loaded {len(documents)} pages"
        )

        # Split into chunks
        splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
        )

        chunks = splitter.split_documents(
            documents
        )

        print(
            f"Created {len(chunks)} chunks"
        )

        # Remove old vector database
        try:

            if os.path.exists(
                VECTORSTORE_PATH
            ):

                shutil.rmtree(
                    VECTORSTORE_PATH
                )

        except PermissionError:

            print(
                "Close chatbot/app before rebuilding vectorstore."
            )

            return

        # Create new vector database
        Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_model,
            persist_directory=VECTORSTORE_PATH
        )

        print(
            "Knowledge Base Updated!"
        )


if __name__ == "__main__":

    manager = KnowledgeManager()

    manager.rebuild_knowledge_base()