from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import EMBEDDING_MODEL

def create_vector_store(chunks: list):
    if not chunks:
        raise ValueError("No text chunks found. Check your PDF.")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    return FAISS.from_documents(chunks, embeddings)

def get_relevant_chunks(vector_store, question: str, k: int = 4) -> list:
    return vector_store.similarity_search(question, k=k)
