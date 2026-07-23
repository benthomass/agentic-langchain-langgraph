import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PDF_PATH = Path(__file__).parent / "HuckFinn.pdf"

if __name__ == "__main__":
    # Only real change vs txt: use PyPDFLoader instead of UnstructuredLoader
    loader = PyPDFLoader(str(PDF_PATH))
    document = loader.load()

    print("Splitting...")
    # Recursive splitter falls back to character splits; keep chunks under
    # mxbai-embed-large's ~512-token context (CharacterTextSplitter won't).
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    print("Ingesting...")
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("Finished")
