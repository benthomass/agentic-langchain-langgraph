from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from react import tools, llm

load_dotenv()

SYSATEM_MESSAGE = """
You are a helpful assistant that can use the following tools to answer questions:
"""


def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning node.
    """
    tool_mode = ToolNode(tools)
    agent = create_react_agent(llm, tool_mode)
    return agent.invoke(state)
