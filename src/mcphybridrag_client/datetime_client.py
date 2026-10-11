"""Ollama-powered MCP client that can ONLY use allowlisted tools (e.g. date/time)."""

import asyncio
import json
import os

from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from ollama import AsyncClient

load_dotenv()
MODEL = os.getenv("MODEL")
MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS"))

# URL of the running MCP server (Streamable HTTP transport).
SERVER_URL = os.getenv("SERVER_URL", "http://127.0.0.1:8000/mcp")

# Comma-separated tool names this client is allowed to use.
# Get the exact names by running list_tools.py first.
ALLOWED_TOOLS = {
    name.strip()
    for name in os.getenv("DATETIME_ALLOWED_TOOLS", "").split(",")
    if name.strip()
}
if not ALLOWED_TOOLS:
    raise ValueError("DATETIME_ALLOWED_TOOLS is not set in .env")

ollama = AsyncClient()


def convert_mcp_tools(tools):
    """Convert MCP tool definitions into Ollama tool schemas."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
            },
        }
        for tool in tools
    ]


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
    return f"Tool error: {text}" if result.isError else text


async def get_allowed_tools(session: ClientSession):
    """Discover all server tools, keep only the allowlisted ones."""
    discovered = await session.list_tools()
    allowed = [t for t in discovered.tools if t.name in ALLOWED_TOOLS]

    missing = ALLOWED_TOOLS - {t.name for t in allowed}
    if missing:
        print(f"Warning: these allowed tools were not found on the server: {missing}")
    if not allowed:
        raise RuntimeError("None of the allowed tools exist on the server.")

    print("\nThis client can use:")
    for tool in allowed:
        print(f"  - {tool.name}: {tool.description or ''}")
    hidden = len(discovered.tools) - len(allowed)
    print(f"({hidden} other server tool(s) hidden from the model)")

    return allowed


async def ask_mcp(session: ClientSession, allowed_tools, question: str) -> str:
    """Let the LLM use only the allowed MCP tools."""
    ollama_tools = convert_mcp_tools(allowed_tools)
    allowed_names = {t.name for t in allowed_tools}

    messages = [
        {
            "role": "system",
            "content": (
                "You are a date and time assistant. You can only answer "
                "questions about the current date and time, using the "
                "provided tool. Never guess the date or time; always call "
                "the tool. If the question is about anything else, politely "
                "say you can only help with the current date and time."
            ),
        },
        {"role": "user", "content": question},
    ]

    for _ in range(MAX_TOOL_ROUNDS):
        response = await ollama.chat(
            model=MODEL,
            messages=messages,
            tools=ollama_tools,
        )
        message = response.message
        messages.append(message)

        tool_calls = message.tool_calls or []
        if not tool_calls:
            return message.content or "(The model returned no text.)"

        for call in tool_calls:
            name = call.function.name
            arguments = call.function.arguments
            print(f"\n[LLM requested tool] {name}")
            print(f"[Arguments] {arguments}")

            # Second safety layer: even if the model invents a tool name,
            # we never execute anything outside the allowlist.
            if name not in allowed_names:
                tool_result = f"Error: tool {name!r} is not allowed in this client"
            else:
                try:
                    result = await session.call_tool(name, arguments=arguments)
                    tool_result = get_text(result)
                except Exception as exc:
                    tool_result = f"Tool execution failed: {exc}"

            print(f"[Tool result] {tool_result}")
            messages.append(
                {"role": "tool", "tool_name": name, "content": tool_result}
            )

    return "Stopped because the maximum number of tool-calling rounds was reached."


async def main():
    print(f"Using Ollama model: {MODEL}")
    print(f"Connecting to MCP server at {SERVER_URL} ...")

    async with streamable_http_client(SERVER_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            allowed_tools = await get_allowed_tools(session)
            print("\nAsk about the current date/time, or type 'exit' to quit.")

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
                    answer = await ask_mcp(session, allowed_tools, question)
                    print(f"\nAssistant: {answer}")
                except Exception as exc:
                    print(f"\nError: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
