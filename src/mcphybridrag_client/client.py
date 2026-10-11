
"""Ollama-powered MCP client."""

import asyncio
import json
from pathlib import Path

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from ollama import AsyncClient
import os
from dotenv import load_dotenv

load_dotenv()
MODEL = os.getenv("MODEL")
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "tev1:0.8b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS"))

server_dir_value = os.getenv("SERVER_DIR")
if not server_dir_value:
    raise ValueError("SERVER_DIR is not set in .env")
SERVER_DIR = Path(server_dir_value)

ollama = AsyncClient()


def convert_mcp_tools(tools):
    """Convert MCP tool definitions into Ollama tool schemas."""
    converted = []

    for tool in tools:
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            }
        )

    return converted


def get_text(result) -> str:
    """Extract text from an MCP tool result."""
    parts = [
        item.text
        for item in result.content
        if getattr(item, "type", None) == "text"
    ]

    text = "\n".join(parts) or json.dumps(
        getattr(result, "structuredContent", None) or {}
    )

    if result.isError:
        return f"Tool error: {text}"

    return text


async def select_tool(question: str, tools) -> str | None:
    """Ask the router model which tool fits the question.

    Returns a tool name, or None when no tool is needed.
    tev1 is a classifier: it picks one option from a list via
    Ollama's /v1/systemone endpoint (not a normal chat).
    """
    criteria = {
        tool.name: (tool.description or tool.name).strip().splitlines()[0]
        for tool in tools
    }
    criteria["none"] = "No tool is needed; answer directly."

    payload = {
        "model": ROUTER_MODEL,
        "state": f"User question: {question}",
        "questions": {
            "tool": {
                "type": "choice",
                "instructions": "Which tool is needed to answer the user question?",
                "criteria": criteria,
            },
            "sentiment": {
                "type": "choice",
                "instructions": "How positive is the user's message?",
                "criteria": {
                    "negative": "The message is unhappy or angry.",
                    "neutral": "The message is neither positive nor negative.",
                    "positive": "The message is happy or friendly.",
                },
            },
        },
    }

    async with httpx.AsyncClient(timeout=60) as http:
        response = await http.post(f"{OLLAMA_HOST}/v1/systemone", json=payload)
        response.raise_for_status()
    data = response.json()
    answer = data["answers"]["tool"]
    print(json.dumps(data, indent=2))

    return None if answer["choice"] == "none" else answer["choice"]


async def ask_mcp(session: ClientSession, question: str) -> str:
    """Router model picks the tool; the main LLM does everything else."""

    discovered = await session.list_tools()

    try:
        selected = await select_tool(question, discovered.tools)
        allowed = [tool for tool in discovered.tools if tool.name == selected]
    except Exception as exc:
        print(f"\n[Router failed: {exc}] Giving all tools to {MODEL}.")
        allowed = discovered.tools

    ollama_tools = convert_mcp_tools(allowed)
    allowed_names = {tool.name for tool in allowed}

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful local assistant. "
                "Use the available tools when calculations or the "
                "current date/time are needed. Never invent tool results. "
                "After receiving tool results, answer the user clearly."
            ),
        },
        {"role": "user", "content": question},
    ]

    for round_number in range(MAX_TOOL_ROUNDS):
        response = await ollama.chat(
            model=MODEL,
            messages=messages,
            tools=ollama_tools or None,
        )

        message = response.message
        messages.append(message)

        tool_calls = message.tool_calls or []

        # No tool requested: this is the final model answer.
        if not tool_calls:
            return message.content or "(The model returned no text.)"

        for call in tool_calls:
            name = call.function.name
            arguments = call.function.arguments

            print(f"\n[LLM requested tool] {name}")
            print(f"[Arguments] {arguments}")

            # Only execute the tool(s) the router allowed.
            if name not in allowed_names:
                tool_result = f"Error: tool {name!r} was not selected"
            else:
                try:
                    result = await session.call_tool(
                        name,
                        arguments=arguments,
                    )
                    tool_result = get_text(result)
                except Exception as exc:
                    tool_result = f"Tool execution failed: {exc}"

            print(f"[Tool result] {tool_result}")

            # Ollama expects tool results as messages in the conversation.
            messages.append(
                {
                    "role": "tool",
                    "tool_name": name,
                    "content": tool_result,
                }
            )

    return (
        "Stopped because the maximum number of tool-calling rounds "
        "was reached. Please try a simpler question."
    )


async def main():
    if not SERVER_DIR.exists():
        raise FileNotFoundError(f"MCP server project not found: {SERVER_DIR}")

    # Run the server inside its own project/venv via uv.
    server_parameters = StdioServerParameters(
        command="uv",
        args=["run", "--directory", str(SERVER_DIR), "mcphybridrag-server"],
    )

    print(f"Using Ollama model: {MODEL} (tool router: {ROUTER_MODEL})")
    print("Starting MCP server over stdio...")

    # The client launches the server and manages its lifecycle.
    async with stdio_client(server_parameters) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Connected to MCP server.")
            print("Ask a question, or type 'exit' to quit.")

            while True:
                try:
                    question = input("\nYou: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    break

                if question.lower() in {"exit", "quit"}:
                    print("Goodbye!")
                    break

                if not question:
                    continue

                try:
                    answer = await ask_mcp(session, question)
                    print(f"\nAssistant: {answer}")
                except Exception as exc:
                    print(f"\nError: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
