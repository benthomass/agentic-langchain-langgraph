import asyncio
import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters, stdio_client

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    api_key=os.getenv("GOOGLE_API_KEY"),
)

stdio_server_params = StdioServerParameters(
    command=sys.executable,
    args=[
        r"C:\Users\BenThomas\courses\ia\langchain-langgraph\agentic-langchain-langgraph\mcp-crash-course\servers\math_server.py"
    ],
)


async def main() -> None:
    async with stdio_client(stdio_server_params) as (read_stream, write_stream):
        async with ClientSession(
            read_stream=read_stream, write_stream=write_stream
        ) as session:
            await session.initialize()
            print("Session initialized")

            tools = await load_mcp_tools(session)
            print(f"Loaded {len(tools)} tool(s)")

            agent = create_react_agent(llm, tools)
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content="What is 2 + 2?")]}
            )
            print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
