import os
import time  # Added to handle our pauses
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Changed: Bring in the Google embeddings instead of Ollama
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PDF_PATH = Path(__file__).parent / "HuckFinn.pdf"

if __name__ == "__main__":
    loader = PyPDFLoader(str(PDF_PATH))
    document = loader.load()

    print("Splitting...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(document)
    print(f"Created {len(texts)} chunks")

    print("Initializing Google Embeddings...")
    # Changed: Set up Google with 768 dimensions to match your Pinecone index
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        output_dimensionality=768,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    # Grab 90 chunks at a time (just under the 100 limit to be safe)
    batch_size = 90
    print(f"Ingesting into Pinecone in batches of {batch_size}...")

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        PineconeVectorStore.from_documents(
            batch, embeddings, index_name=os.environ["INDEX_NAME"]
        )

        print(f"Done chunks {i} to {i + len(batch)} of {len(texts)}")

        # Force a 60-second wait so Google's rate limit completely resets
        time.sleep(60)

    print("Finished!")
