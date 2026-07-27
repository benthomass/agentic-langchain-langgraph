from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os


load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite", temperature=0, api_key=os.getenv("GOOGLE_API_KEY")
)

prompt = ChatPromptTemplate.from_template(
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question}\nContext: {context}\nAnswer:"
)

generation_chain = prompt | llm | StrOutputParser()
