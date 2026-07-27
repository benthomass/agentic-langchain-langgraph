from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from graph.consts import RETRIEVE, GRADE_DOCUMENTS, WEB_SEARCH, GENERATE
from graph.nodes import retrieve, grade_documents, web_search, generate
from graph.state import GraphState

load_dotenv()


def decide_to_generate(state):
    print("--- ASSES GRADED DOCUMENTS ---")

    if state["web_search"]:
        print("--- NOT ALL DOCUMENTS WERE RELEVANT ---")
        return WEB_SEARCH
    else:
        print("--- ALL DOCUMENTS WERE RELEVANT ---")
        return GENERATE


workflow = StateGraph(GraphState)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(WEB_SEARCH, web_search)
workflow.add_node(GENERATE, generate)

workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)


workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    path_map={WEB_SEARCH: WEB_SEARCH, GENERATE: GENERATE},
)

workflow.add_edge(WEB_SEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()

print(app.get_graph().draw_mermaid_png())
