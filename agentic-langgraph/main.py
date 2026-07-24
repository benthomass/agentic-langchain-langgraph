from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


def main():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY")
    )
    print("Hello from langgraph!")


if __name__ == "__main__":
    main()
