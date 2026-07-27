from typing import Any, Dict, List
from graph.state import GraphState
from ingestion import retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    print(f"Retrieving documents for question: {state['question']}")
    question = state["question"]

    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}
