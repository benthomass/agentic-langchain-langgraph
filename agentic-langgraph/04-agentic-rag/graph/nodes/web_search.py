import sys
from pathlib import Path

# Allow running/debugging this nested file directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from typing import Any, Dict
from langchain_core.documents import Document
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

from graph.state import GraphState

web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    print("--- Running web search ---")
    question = state["question"]
    documents = state["documents"]

    tavily_results = web_search_tool.invoke({"query": question})
    joined_tavily_results = "\n".join([r["content"] for r in tavily_results["results"]])
    web_results = Document(page_content=joined_tavily_results)

    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}


if __name__ == "__main__":
    web_search(state={"question": "Bogart", "documents": None})
