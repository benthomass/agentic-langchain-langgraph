from dotenv import load_dotenv
from graph.graph import app

load_dotenv()
if __name__ == "__main__":
    print("Hello Advanced RAG")
    result = app.invoke({"question": "Why is film noir so attractive?"})
    print(result)
