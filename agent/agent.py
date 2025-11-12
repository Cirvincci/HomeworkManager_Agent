import os
import asyncio
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseServerParams
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from typing import Any, Dict
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Use model from environment or default to deepseek
model_type = os.getenv('MODEL', 'deepseek/deepseek-chat')


toolset = MCPToolset(
    connection_params=SseServerParams(
        url="http://localhost:50001/sse",
    ),
)

# Create agent
root_agent = Agent(
    name="homework_manager_agent",
    model=LiteLlm(model=model_type),
    instruction="You are an intelligent assistant capable of using external tools via MCP.",
    tools=[toolset]
)