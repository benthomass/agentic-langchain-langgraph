from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable
from langchain_google_genai import ChatGoogleGenerativeAI

MAX_ITERATIONS = 5
MODEL = "qwen3:1.7b"


@tool
def get_product_price(product_name: str) -> float:
    """
    Look up the price of a product in the catalogue.

    Args:
        product_name: The base name of the product ONLY (e.g., 'laptop').
                      NEVER include discount tiers, adjectives, or colors in this name.
    """

    print(f"Looking up price for product: {product_name}")
    prices = {
        "laptop": 999.0,
        "smartphone": 699.0,
        "headphones": 199.0,
    }
    print(f"Found price for {product_name}: {prices.get(product_name, 0)}")
    return prices.get(product_name, 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to a price.
    ALLWAYS use this tool to apply discounts before calculating shipping costs."""
    print(f"Applying discount for tier: {discount_tier} on price: {price}")
    discount_percentages = {"bronze": 5, "silver": 10, "gold": 12}
    discount = discount_percentages.get(discount_tier, 0)
    print(f"Discount applied: {discount}%")
    return price * (1 - discount / 100)


@tool
def calculate_shipping(discounted_price: float, location: str) -> float:
    """Calculate the shipping cost based on the discounted price and location.
    You should never calculate the shipping cost manually, always use this tool to get the final price
    of the product including discount and shipping cost.
    ALWAYS use this tool immediately after applying the discount."""
    prices_dict = {"USA": 5, "Canada": 10, "Europe": 15, "Other": 20}
    total_cost = discounted_price + prices_dict.get(location, 0)
    print(f"Total cost calculated: {total_cost}")
    return total_cost


# ----- Agent Loop -----


@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount, calculate_shipping]
    tools_dict = {tool.name: tool for tool in tools}
    # lm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant that can look up product prices and apply discounts."
                "You have access to a product catalog tool, a discount application tool, and a shipping calculation tool."
                "STRICT RULES - you must follow these exactly:"
                "1. NEVER guess or assume any product price, discount or shipping cost. You must use the tools provided to get accurate information."
                "2. NEVER make up any product names, discount tiers, or shipping locations. Only use the ones provided in the tools."
                "3. ALWAYS use the tools to get the product price, apply the discount, and calculate the shipping cost."
                "4. NEVER provide any information that is not obtained from the tools."
                "5. NEVER calculate or estimate prices, discounts, or shipping costs manually. You must use the tools for all calculations."
                "6. If the user does not specify a valid product name, discount tier, or shipping location, you must inform them and ask for clarification."
                "7. If the user inputs a country that is not USA, Canada, or Europe, you must classify the country as 'Other' "
                "and use the shipping cost for 'Other' in the shipping calculation."
                "8. ALWAYS provide the final answer in a clear and concise manner, including the product name, discount tier, and total cost with shipping."
                "9. ALWAYS follow the order of operations: first get the product price, then apply the discount, and finally calculate the shipping cost."
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(MAX_ITERATIONS):
        print(f"\n--- Iteration {iteration + 1} ---")
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"\nFinal answer: {ai_message.content}")
            return ai_message.content

        # Process only for the FIRST tool call - force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        tool_to_use = tools_dict.get(tool_name)
        print(f" {tool_name} called with arguments: {tool_args}")
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found in tools_dict.")

        observation = tool_to_use.invoke(tool_args)

        print(f"Observation from tool {tool_name}: {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )


if __name__ == "__main__":
    print("Welcome to the Agent Loop Tool Calling Example!")
    result = run_agent(
        "What is the price of a laptop with a silver discount in Europe?"
    )
    print(f"Result: {result}")
