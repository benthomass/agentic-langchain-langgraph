from dotenv import load_dotenv

load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import os

CHROMA_PATH = r"C:\Users\BenThomas\courses\ia\langchain-langgraph\agentic-langchain-langgraph\agentic-langgraph\04-agentic-rag\chroma"

urls = [
    "https://www.bfi.org.uk/lists/humphrey-bogart-10-essential-films",
    "https://www.britannica.com/art/film-noir",
    "https://www.filmmakersacademy.com/blog-film-noir-lighting/",
]

docs = [
    UnstructuredLoader(
        web_url=url, chunking_strategy="basic", max_characters=10000
    ).load()
    for url in urls
]
docs_list = [item for sublist in docs for item in sublist]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=20)
docs_splits = text_splitter.split_documents(docs_list)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma.from_documents(
    docs_splits,
    embedding=embeddings,
    collection_name="film-noir",
    persist_directory=CHROMA_PATH,
)

retriever = Chroma(
    collection_name="film-noir",
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings,
).as_retriever()
