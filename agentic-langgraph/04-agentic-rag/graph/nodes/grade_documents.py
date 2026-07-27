from typing import Any, Dict
from graph.state import GraphState
from graph.chains.retrieval_grader import retrieval_grader_chain, GradeDocuments


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the documents retrieved from the vector store are
    relevant to the user question. If they are not, we will set a flag
    to run a web search.

    args:
        state (dict): the current state of the graph

    returns:
        state (dict): filteres put irrelevant documents and updated web_search flag
    """

    print("--- Checking document relevance ---")
    question = state["question"]
    documents = state["documents"]

    filtered_documents = []
    web_search = False

    for doc in documents:
        score = retrieval_grader_chain.invoke(
            {"question": question, "document": doc.page_content}
        )
    if score == "yes":
        print(f"Document is relevant: {doc.page_content}")
        filtered_documents.append(doc)
    else:
        print(f"Document is irrelevant: {doc.page_content}")
        web_search = True

    return {
        "documents": filtered_documents,
        "question": question,
        "web_search": web_search,
    }
