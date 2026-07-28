from dotenv import load_dotenv

load_dotenv()

import operator
from pathlib import Path
from typing import Annotated, Any, Sequence
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    aggregate: Annotated[list, operator.add]
    which: str


class ReturnNodeValue:
    def __init__(self, node_secret: str):
        self.value = node_secret

    def __call__(self, state: State) -> State:
        print(f"Adding {self.value} to {state['aggregate']}")
        return {"aggregate": [self.value]}


def route_bc_or_cd(state: State) -> Sequence[str]:
    if state["which"] == "bc":
        return ["b", "c"]
    return ["c", "d"]


intermediate_nodes = ["b", "c", "d"]

graph_path = Path(__file__).parent.parent / "graph-async.png"

builder = StateGraph(State)
builder.add_node("a", ReturnNodeValue("I'm in A"))
builder.add_edge(START, "a")

builder.add_node("b", ReturnNodeValue("I'm in B"))
builder.add_node("c", ReturnNodeValue("I'm in C"))
builder.add_node("d", ReturnNodeValue("I'm in D"))
builder.add_node("e", ReturnNodeValue("I'm in E"))

builder.add_conditional_edges("a", route_bc_or_cd, intermediate_nodes)

for node in intermediate_nodes:
    builder.add_edge(node, "e")

builder.add_edge("e", END)
graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path=str(graph_path))


if __name__ == "__main__":
    print("Hello, async world!")
    graph.invoke(
        {"aggregate": [], "which": "bc"}, {"configurable": {"thread_id": "foo"}}
    )
