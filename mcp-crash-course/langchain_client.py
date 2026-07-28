import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

from mcp import ClientSession, StdioServerParameters
from mcp import stdio_client
from langchain_mcp_adapters.client import load_mcp_tools
from langchain.agents import create_agent

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite", temperature=0, api_key=os.getenv("GOOGLE_API_KEY")
)

stdio_server_params = StdioServerParameters(
    command="python",
    args=[
        r"C:\Users\BenThomas\courses\ia\langchain-langgraph\agentic-langchain-langgraph\mcp-crash-course\servers\math_server.py"
    ],
)


async def main():
    asyncio.run(main())
