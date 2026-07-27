from dotenv import load_dotenv
from pprint import pprint


load_dotenv()

from ingestion import retriever
from graph.chains.retrieval_grader import retrieval_grader_chain, GradeDocuments
from graph.state import GraphState
from graph.chains.generation import generation_chain


def test_retrieval_grader_answer_yes() -> None:
    question = "Bogart"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    res: GradeDocuments = retrieval_grader_chain.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == "yes"


def test_retrieval_grader_answer_no() -> None:
    question = "Bogart"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    res: GradeDocuments = retrieval_grader_chain.invoke(
        {"question": "Quantum physics", "document": doc_txt}
    )

    assert res.binary_score == "no"


def test_generation_chain() -> None:
    question = "Bogart"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)
