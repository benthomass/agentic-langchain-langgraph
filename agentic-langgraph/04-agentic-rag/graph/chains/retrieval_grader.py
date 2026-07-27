from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Literal

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: Literal["yes", "no"] = Field(
        description="Documents are relevant to the question if the score is 'yes', otherwise 'no'."
    )

    @field_validator("binary_score", mode="before")
    @classmethod
    def normalize_binary_score(cls, v: str) -> str:
        return str(v).strip().lower()


structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """
You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keywords or phrases that are relevant to the question, score it as relevant.
If the document does not contain any relevant keywords or phrases, score it as irrelevant.
Give a binary score "yes" or "no" to idicate relevance.
"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("user", "{question}"),
        ("user", "{document}"),
    ]
)

retrieval_grader_chain = grade_prompt | structured_llm_grader
