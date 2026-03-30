from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP
import tempfile
import os

def load_and_split_pdf(uploaded_file) -> list:
    """
    Takes a Streamlit uploaded file,
    saves it temporarily, loads it,
    splits into chunks and returns them.
    """
    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".pdf"
    ) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Load PDF pages
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(pages)

    os.unlink(tmp_path)  # clean up temp file
    return chunks
