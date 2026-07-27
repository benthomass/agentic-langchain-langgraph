from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from graph.consts import RETRIEVE, GRADE_DOCUMENTS, WEB_SEARCH, GENERATE
from graph.nodes import retrieve, grade_documents, web_search, generate
from graph.state import GraphState
from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import router, RouteQuery

load_dotenv()


def decide_to_generate(state):
    print("--- ASSES GRADED DOCUMENTS ---")

    if state["web_search"]:
        print("--- NOT ALL DOCUMENTS WERE RELEVANT ---")
        return WEB_SEARCH
    else:
        print("--- ALL DOCUMENTS WERE RELEVANT ---")
        return GENERATE


def grade_generation_grounded_in_docs_and_question(state: GraphState) -> str:
    print("--- GRADE GENERATION GROUNDED IN DOCS AND QUESTION ---")
    question = state["question"]
    docs = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"facts": docs, "question": question, "answer": generation}
    )

    if score.is_hallucinated == "yes":
        print("---Decision: generation is grounded in docs and question ---")
        print("--- Grade generation vs question---")
        answer_score = answer_grader.invoke(
            {"question": question, "answer": generation}
        )
        if answer_score.binary_score:
            print("---Decision: generation addresses the question ---")
            return "useful"
        else:
            print("---Decision: generation does not address the question ---")
            return "not useful"
    else:
        print("---Decision: generation is not grounded in docs and question ---")
        return "not supported"


def route_question(state: GraphState) -> str:
    print("--- ROUTING QUESTION ---")
    question = state["question"]
    source: RouteQuery = router.invoke({"question": question})
    if source.datasource == "vectorstore":
        return RETRIEVE
    elif source.datasource == "web_search":
        return WEB_SEARCH
    else:
        raise ValueError(f"Invalid datasource: {source.datasource}")


workflow = StateGraph(GraphState)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(WEB_SEARCH, web_search)
workflow.add_node(GENERATE, generate)

workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)

workflow.set_conditional_entry_point(
    route_question,
    path_map={RETRIEVE: RETRIEVE, WEB_SEARCH: WEB_SEARCH},
)

workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    path_map={WEB_SEARCH: WEB_SEARCH, GENERATE: GENERATE},
)
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_docs_and_question,
    path_map={"useful": END, "not useful": WEB_SEARCH, "not supported": GENERATE},
)

workflow.add_edge(WEB_SEARCH, GENERATE)

app = workflow.compile()

print(app.get_graph().draw_mermaid_png())
