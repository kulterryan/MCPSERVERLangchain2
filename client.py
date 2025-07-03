from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()

import asyncio
import os

async def main():
    try:
        client=MultiServerMCPClient(
            {
                "math":{
                    "command":"python",
                    "args":[os.path.abspath("mathserver.py")], ## Use absolute path
                    "transport":"stdio",
                },
                "weather": {
                    "url": "http://localhost:8000/mcp",  # Ensure server is running here
                    "transport": "streamable_http",
                }

            }
        )

        os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

        print("Connecting to MCP servers...")
        tools = await client.get_tools()
        print(f"Successfully retrieved {len(tools)} tools: {[tool.name for tool in tools]}")
        
        model=ChatGroq(model="qwen-qwq-32b")
        agent=create_react_agent(
            model,tools
        )
    except Exception as e:
        print(f"Error setting up client or getting tools: {e}")
        return

    try:
        math_response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "what's (3 + 9) x 12?"}]}
        )
        print("Math response:", math_response['messages'][-1].content)
    except Exception as e:
        print(f"Error with math query: {e}")

    try:
        weather_response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "what is the weather in California?"}]}
        )
        print("Weather response:", weather_response['messages'][-1].content)
    except Exception as e:
        print(f"Error with weather query: {e}")

asyncio.run(main())
