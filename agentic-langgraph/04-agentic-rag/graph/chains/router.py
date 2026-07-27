from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()


class RouteQuery(BaseModel):
    """
    Route a user query to the most relevant datasource.
    """

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or vectorstore.",
    )


llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite", temperature=0, api_key=os.getenv("GOOGLE_API_KEY")
)
structured_llm = llm.with_structured_output(RouteQuery)

system = """
You are a router that routes a user query to the most relevant datasource.
Given a user question choose to route it to web search or vectorstore.
The vectorstore contains information about the film noir and Humphrey Bogart.
Use the vectorstore to answer question about film noir and Humphrey Bogart
Use web_search to answer question about other topics."""
router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Question: {question}"),
    ]
)
router = router_prompt | structured_llm
