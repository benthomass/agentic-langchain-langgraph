from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_ollama import ChatOllama


load_dotenv()


@tool
def triple(num: float) -> float:
    """
    param num: a number to triple
    returns: the triple of the input number
    """
    print(f"Tripling {num}")
    return float(num) * 3


tools = [TavilySearch(max_results=1), triple]

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite", temperature=0, thinking_level="minimal"
).bind_tools(tools)

# llm = ChatOllama(model="llama3.1", temperature=0).bind_tools(tools)
