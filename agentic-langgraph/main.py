from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState

from nodes import tool_node, run_agent_reasoning

load_dotenv()


def main():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY")
    )
    print("Hello from langgraph!")


if __name__ == "__main__":
    main()
