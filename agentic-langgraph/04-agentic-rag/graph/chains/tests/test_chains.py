from dotenv import load_dotenv
from pprint import pprint
import os

load_dotenv()

from ingestion import retriever
from graph.chains.retrieval_grader import retrieval_grader_chain, GradeDocuments
from graph.state import GraphState
from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.hallucination_grader import GradeHallucination
from graph.chains.router import router, RouteQuery


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


def test_hallucination_grader_yes() -> None:
    question = "Bogart"
    docs = retriever.invoke(question)

    generation = generation_chain.invoke({"context": docs, "question": question})
    res: GradeHallucination = hallucination_grader.invoke(
        {"facts": docs, "question": question, "answer": generation}
    )
    assert res.is_hallucinated == "yes"


def test_hallucination_grader_no() -> None:
    question = "Bogart"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    res: GradeHallucination = hallucination_grader.invoke(
        {"facts": docs, "question": question, "answer": "Quantum physics is fun"}
    )
    assert res.is_hallucinated == "no"


def test_router() -> None:
    question = "Bogart"
    res: RouteQuery = router.invoke({"question": question})
    assert res.datasource == "vectorstore"


def test_router_web_search() -> None:
    question = "Quantum physics"
    res: RouteQuery = router.invoke({"question": question})
    assert res.datasource == "web_search"
