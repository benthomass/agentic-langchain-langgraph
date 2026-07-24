import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from operator import itemgetter

load_dotenv()
print("Initializing components")

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
llm = ChatOllama(model="qwen2.5:3b")

vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)  # limit relevant docs taken to 3

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
    
    {context}
    
    Question: {question}
    
    Provide a detailed answer:"""
)


def retrieval_chain_without_lcel(query: str):
    """Simple retrieval chain without LangChain Expressions Language

    Limitations:
    - Manual step by step exectutio
    - No built in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error prone
    """

    # Step 1: Retrieve relevant documents
    docs = retriever.invoke(query)

    # Step 2: Format documents
    context = format_docs(docs)

    # Step 3: Combine context with query and invoke LLM
    messages = prompt_template.format_messages(context=context, question=query)

    response = llm.invoke(messages)
    return response.content


def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL.
    Returns a chain that can be invoked with a query string.

    Benefits:
    - Declarative and composable: easy to chain operations with pipe operator |
    - Built-in streaming: chain.stream() works out of the box
    - Built-in  async: chain.ainvoke() and chain.astream() are available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    print("Retrieving...")
    query = "What is Pinecone in machine learning?"

    # ===========================
    # Option 0:Wthout RAG
    # ===========================
    print("\n" + "=" * 70)
    print("Implementation 0: Raw LLM Invocation (No RAG)")
    print("=" * 70 + "\n")
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("Answer:")
    print(result_raw.content)

    # ===========================
    # Option 1: Use implementation without LCEL
    # ===========================

    print("\n" + "=" * 70)
    print("Implementation 1: Retrieval Chain without LCEL")
    print("=" * 70 + "\n")
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("Answer:")
    print(result_without_lcel)

    # ===========================
    # Option 2: Use implementation with LCEL
    # ===========================

    print("\n" + "=" * 70)
    print("Implementation 2: Retrieval Chain with LCEL")
    print("=" * 70 + "\n")
    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("Answer:")
    print(result_with_lcel)
