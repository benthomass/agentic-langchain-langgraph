from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from typing import List
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

# from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_ollama import ChatOllama


@tool
def search_online(query: str) -> str:
    """
    Search online for information related to the query.

    Args:
        query: Search terms to look for
    """
    return f"Found information related to '{query}' online."


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for the agent's response with answer and sources"""

    answer: str = Field(description="The answer provided by the agent")
    sources: List[Source] = Field(
        description="A list of sources used to generate the answer"
    )


llm = ChatOllama(model="qwen3", temperature=0)
# llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)
tools = [search_online]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-session-3!")
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="""Search for three job postings for a physicist with a minor 
                    in mathematics on linkedin and list the company names, titles an descriptions?"""
                )
            ]
        }
    )
    print(f"Agent response: {result}")


if __name__ == "__main__":
    main()
