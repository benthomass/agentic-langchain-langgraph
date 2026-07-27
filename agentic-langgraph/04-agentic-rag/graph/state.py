from typing import TypedDict, List


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation response
        web_search: wether to add search results to the generation (bool)
        documents: list of documents retrieved from the vector store
    """

    question: str
    generation: str
    web_search: bool
    documents: List[str]
