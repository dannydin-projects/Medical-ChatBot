from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings


# Load all PDF from a folder
def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob='*.pdf',
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

# FIlter out unnecessary details from docs
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs

# Splitting text into small chunks to fit in llm's context window
def chunking_docs(docs):
    text_splits = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50,)
    text_chunks = text_splits.split_documents(docs)
    return text_chunks

# Text(Chunked Docs) -> Vectors
def download_embeddings():
    '''
    Download the HuggingFace embedding model and return the embeddings object.
    '''
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings
embeddings = download_embeddings()
