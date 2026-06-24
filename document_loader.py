from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    TextLoader,
    DirectoryLoader
)

from langchain_community.document_loaders import (
    UnstructuredWordDocumentLoader
)


def load_documents():

    documents = []

    # PDFs
    pdf_loader = PyPDFDirectoryLoader(
        "data/pdfs"
    )

    documents.extend(
        pdf_loader.load()
    )

    # DOCX
    docx_loader = DirectoryLoader(
        "data/docx",
        glob="*.docx",
        loader_cls=UnstructuredWordDocumentLoader
    )

    documents.extend(
        docx_loader.load()
    )

    # TXT
    txt_loader = DirectoryLoader(
        "data/txt",
        glob="*.txt",
        loader_cls=TextLoader
    )

    documents.extend(
        txt_loader.load()
    )

    return documents