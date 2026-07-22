import inspect

from dotenv import load_dotenv
import ollama
import regex as re

load_dotenv()  # Load environment variables from .env file

from langsmith import traceable

MAX_ITERATIONS = 5
MODEL = "qwen3:1.7b"


@traceable(run_type="tool")
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


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to a price.
    ALLWAYS use this tool to apply discounts before calculating shipping costs."""
    print(f"Applying discount for tier: {discount_tier} on price: {price}")
    discount_percentages = {"bronze": 5, "silver": 10, "gold": 12}
    discount = discount_percentages.get(discount_tier, 0)
    print(f"Discount applied: {discount}%")
    return price * (1 - discount / 100)


@traceable(run_type="tool")
def calculate_shipping(discounted_price: float, location: str) -> float:
    """Calculate the shipping cost based on the discounted price and location.
    You should never calculate the shipping cost manually, always use this tool to get the final price
    of the product including discount and shipping cost.
    ALWAYS use this tool immediately after applying the discount."""
    prices_dict = {"USA": 5, "Canada": 10, "Europe": 15, "Other": 20}
    total_cost = discounted_price + prices_dict.get(location, 0)
    print(f"Total cost calculated: {total_cost}")
    return total_cost


tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
    "calculate_shipping": calculate_shipping,
}


def get_tool_descriptions(tools_dict):
    descriptions = []
    for tool_name, tool_func in tools_dict.items():
        original_function = getattr(tool_func, "__wrapped__", tool_func)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(original_function)
        descriptions.append(f"{tool_name}{signature} -- {docstring}")

    return "\n".join(descriptions)


tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())
react_prompt = f"""
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

                Answer the following questions as best you can. You have access to the following tools:

                {tool_descriptions}

                Use the following format:

                Question: the input question you must answer
                Thought: you should always think about what to do
                Action: the action to take, should be one of [{tool_names}]
                Action Input: the input to the action
                Observation: the result of the action
                ... (this Thought/Action/Action Input/Observation can repeat N times)
                Thought: I now know the final answer
                Final Answer: the final answer to the original input question

                Begin!

                Question: {{question}}
                Thought
                """


@traceable(name="Ollama Chat Traced", run_type="llm")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model=model, messages=messages, options=options)


# ----- Agent Loop -----


@traceable(name="LangChain Agent Loop")
def run_agent(question: str):

    print(f"Question: {question}")
    print("=" * 60)

    prompt = react_prompt.format(question=question)
    scratchpad = ""

    for iteration in range(MAX_ITERATIONS):
        print(f"\n--- Iteration {iteration + 1} ---")
        full_prompt = prompt + scratchpad

        response = ollama_chat_traced(
            model=MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            options={"stop": ["\nObservation"], "temperature": 0},
        )
        output = response.message.content
        print(f"LLM Output: \n{output}")

        print(f"   [Parsing] Looking for Final Answer in LLM output...")
        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print("\n" + "=" * 60)
            print(f"Final Answer: {final_answer}")
            return final_answer

        print(f"   [Parsing] Looking for Action and Action Input in LLM output...")
        # Process only for the FIRST tool call - force one tool per iteration

        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print(
                "   [Parsing] ERROR: Could not parse Action/Action Input from LLM output"
            )
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(f"   [Tool Selected] {tool_name} with args {tool_input_raw}")
        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=, 1")[-1].strip("''") for x in raw_args]

        print(f"   [Tool Executing] {tool_name}{{args}}...")
        if tool_name not in tools:
            observation = f"Error Tool: {tool_name} not found. Available tools: {list[str](tools.keys())}"
        else:
            observation = str(tools[tool_name](*args))

        print(f"   [Tool Result] {observation}")

        scratchpad += f"{output}:\nObservation: {observation}:\nThought:"


if __name__ == "__main__":
    print("Welcome to the Agent Loop Tool Calling Example!")
    result = run_agent(
        "What is the price of a laptop with a silver discount in Europe?"
    )
    print(f"Result: {result}")
