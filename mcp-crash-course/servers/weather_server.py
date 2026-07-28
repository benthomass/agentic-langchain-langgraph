from typing import List
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")


@mcp.tool()
async def get_weather(location: str) -> str:
    """Get the weather for a given location"""
    return f"It's always hot in my room"


if __name__ == "__main__":
    mcp.run(transport="sse")
