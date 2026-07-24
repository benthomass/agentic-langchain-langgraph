from typing import List
from pydantic import BaseModel, Field


class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing")
    superfluous: str = Field(description="Critique of what is superfluous")


class AnswerQuestion(BaseModel):
    """Answer the question"""

    answer: str = Field(description="around 250 word detailed answer to the question")
    reflection: Reflection = Field(description="Reflection on the initial answer")
    search_query: List[str] = Field(
        description="1-3 search queries to adress the critique of the current answer"
    )


class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question."""

    references: List[str] = Field(
        description="Citations motivating your updated answer."
    )
