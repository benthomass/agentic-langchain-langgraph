import os
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == "__main__":
    print("Ingesting...")
    loader = UnstructuredLoader(
        file_path="/Users/BenThomas/courses/ia/langchain-langgraph/langchain-course/04-rag/mediumblog1.txt",
        chunking_strategy="basic",
        max_characters=1000,
        encoding="UTF-8",
    )
    document = loader.load()

    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    # Initialize the local Ollama embedding model
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    print("Ingesting...")
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("Finished")
