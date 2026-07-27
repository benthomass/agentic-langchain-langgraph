from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Literal
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite", temperature=0, api_key=os.getenv("GOOGLE_API_KEY")
)


class GradeHallucination(BaseModel):
    is_hallucinated: Literal["yes", "no"] = Field(
        description="Whether the answer is grounded in the facts, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeHallucination)

system = """
You are a helpful assistant that grades the hallucination of an answer.
You will be given a question and an answer.
You need to grade the answer based on the question and the answer.
If the answer is grounded in the facts, return 'yes', otherwise return 'no'.
"""

hallucination_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: {facts}\nQuestion: {question}\nAnswer: {answer}"),
    ]
)

hallucination_grader: RunnableSequence = (
    hallucination_grader_prompt | structured_llm_grader
)
