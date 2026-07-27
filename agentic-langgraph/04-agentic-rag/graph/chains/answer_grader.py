from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field  # works
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()


class GradeAnswer(BaseModel):
    binary_score: bool = Field(
        description="Answer adresses the question, 'yes' or 'no'"
    )


llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite", temperature=0, api_key=os.getenv("GOOGLE_API_KEY")
)
structured_llm = llm.with_structured_output(GradeAnswer)

system = """
You are a grader assesing wether an answer addresses the question.
Give a binary score, 'yes' or 'no'. 'yes' means the answer addresses 
the question, 'no' means it does not.
"""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Question: {question}\nAnswer: {answer}"),
    ]
)

answer_grader: RunnableSequence = answer_prompt | structured_llm
