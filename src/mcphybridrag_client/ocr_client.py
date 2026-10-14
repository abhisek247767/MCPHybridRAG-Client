
"""MCP client with Gemini OCR: attach an image/PDF, then ask questions about it.

Run: uv run src/mcphybridrag_client/ocr_client.py
"""

import asyncio
import json
import mimetypes
from datetime import date
from pathlib import Path

import httpx
from google import genai
from google.genai import types
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from ollama import AsyncClient
import os
from dotenv import load_dotenv

load_dotenv()
MODEL = os.getenv("MODEL")
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "tev1:0.8b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS"))

SERVER_URL = os.getenv("SERVER_URL", "http://127.0.0.1:8000/mcp")

# OCR runs here in the client with Gemini; the server never sees the image.
# Needs GEMINI_API_KEY in .env (https://aistudio.google.com/apikey).
OCR_MODEL = os.getenv("OCR_MODEL", "gemini-flash-latest")
OCR_PROMPT = (
    "Extract all text from this document exactly as written. "
    "Keep the reading order. Write tables as Markdown tables. "
    "Do not summarize, translate, or add anything."
)
SUPPORTED_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/heic",
    "image/heif",
    "application/pdf",
}
MAX_IMAGE_BYTES = 15 * 1024 * 1024  # stay under Gemini's inline request limit

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


def load_file(path_text: str) -> tuple[Path, str, bytes]:
    """Read an image/PDF file and check it is something Gemini can OCR."""
    path = Path(path_text).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type not in SUPPORTED_MIME_TYPES:
        raise ValueError(
            f"Unsupported file type {mime_type!r}. "
            f"Use one of: {', '.join(sorted(SUPPORTED_MIME_TYPES))}"
        )

    data = path.read_bytes()
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"File is {len(data) / 1024 / 1024:.1f} MB; "
            f"the limit is {MAX_IMAGE_BYTES // 1024 // 1024} MB."
        )

    return path, mime_type, data


async def extract_text(path_text: str) -> dict:
    """OCR an image/PDF with Gemini and return the extracted text."""
    path, mime_type, data = load_file(path_text)

    # Created here so the client still starts without a key (no /image used).
    gemini = genai.Client()  # reads GEMINI_API_KEY from the environment
    response = await gemini.aio.models.generate_content(
        model=OCR_MODEL,
        contents=[types.Part.from_bytes(data=data, mime_type=mime_type), OCR_PROMPT],
        config=types.GenerateContentConfig(temperature=0),
    )

    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("OCR returned no text.")

    return {"name": path.name, "text": text}


async def select_tool(question: str, tools, has_document: bool = False) -> str | None:
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
    if has_document:
        # tev1 can't see the document, so tell it that reading a value off the
        # document is the "none" case. Without this, "what is the total?"
        # gets routed to calculate.
        criteria["none"] = (
            "Look up a value already written on the attached document "
            "(total, amount due, client, dates, terms). No tool needed."
        )

    state = f"User question: {question}"
    if has_document:
        state += "\nThe user attached a document; its text is already available."

    payload = {
        "model": ROUTER_MODEL,
        "state": state,
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


def document_message(document: dict) -> dict:
    """System message that gives the LLM the OCR text."""
    return {
        "role": "system",
        "content": (
            f"The user attached a document ({document['name']}). "
            "Answer from this extracted text only; if something is "
            "not in it, say so.\n\n"
            f"<document>\n{document['text']}\n</document>"
        ),
    }


async def fill_arguments(
    tool, question: str, document: dict | None, error: str | None = None
) -> dict:
    """Have the LLM fill the tool's arguments as JSON matching its schema.

    Small models often ignore tools and just reply with text. Asking for
    JSON in the exact schema (Ollama structured output) is far more reliable,
    and works even for models without tool support.
    """
    if not tool.inputSchema.get("properties"):
        return {}  # tool takes no arguments

    messages = [
        {
            "role": "system",
            "content": (
                f"Fill in the arguments for the tool {tool.name!r}.\n"
                f"Tool description:\n{tool.description or ''}\n\n"
                f"Today's date is {date.today().isoformat()}.\n"
                "Reply with JSON only. Copy values exactly from the user's "
                "question or the attached document. Copy dates from the "
                "document exactly as printed (e.g. 05/01/2026); the server "
                "converts them. Only dates you work out yourself, like "
                "'this month', should be written as YYYY-MM-DD. Write "
                "amounts as plain numbers (no currency symbol). Use null "
                "for values that are not available."
            ),
        },
        {"role": "user", "content": question},
    ]
    if document:
        messages.insert(1, document_message(document))
    if error:
        messages.append(
            {
                "role": "user",
                "content": f"The previous arguments were rejected: {error}\nFix them.",
            }
        )

    response = await ollama.chat(
        model=MODEL,
        messages=messages,
        format=tool.inputSchema,
        options={"temperature": 0, "num_ctx": 8192},
    )
    arguments = json.loads(response.message.content or "{}")
    # Drop nulls so the server uses its own defaults for optional arguments.
    return {key: value for key, value in arguments.items() if value is not None}


async def run_selected_tool(
    session: ClientSession, tool, question: str, document: dict | None
) -> str:
    """Always call the tool the router picked, then answer from its result."""
    error = None
    for attempt in range(2):  # one retry if the server rejects the arguments
        arguments = await fill_arguments(tool, question, document, error)
        print(f"\n[Client calling tool] {tool.name}")
        print(f"[Arguments] {arguments}")

        result = await session.call_tool(tool.name, arguments=arguments)
        tool_result = get_text(result)
        print(f"[Tool result] {tool_result}")

        if not result.isError:
            break
        error = tool_result

    messages = [
        {
            "role": "system",
            "content": (
                "Answer the user's question using the tool result below. "
                "The tool result is real data from the database: every "
                "invoice in it is saved, and null only means that field was "
                "not recorded. Mention every relevant invoice. Never add "
                "amounts in different currencies together. "
                "If the tool failed, say so plainly and explain why. "
                "Never claim something happened unless the tool result shows it."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Result from the {tool.name} tool:\n{tool_result}"
            ),
        },
    ]
    response = await ollama.chat(model=MODEL, messages=messages)
    return (response.message.content or "").strip() or tool_result


async def ask_mcp(
    session: ClientSession, question: str, document: dict | None = None
) -> str:
    """Router model picks the tool; the main LLM does everything else."""

    discovered = await session.list_tools()

    try:
        selected = await select_tool(
            question, discovered.tools, has_document=document is not None
        )
        allowed = [tool for tool in discovered.tools if tool.name == selected]
    except Exception as exc:
        print(f"\n[Router failed: {exc}] Giving all tools to {MODEL}.")
        allowed = discovered.tools
    else:
        # Router picked exactly one tool: call it ourselves, don't hope the
        # LLM decides to. ("none" -> allowed is empty -> plain answer below.)
        if allowed:
            return await run_selected_tool(session, allowed[0], question, document)

    ollama_tools = convert_mcp_tools(allowed)
    allowed_names = {tool.name for tool in allowed}

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful local assistant. "
                "Use the available tools when calculations or the "
                "current date/time are needed. Never invent tool results. "
                "After receiving tool results, answer the user clearly.\n"
                # Needed to turn "this month" / "next week" into real dates.
                f"Today's date is {date.today().isoformat()}. "
                "Always write dates as YYYY-MM-DD in tool arguments."
            ),
        },
        {"role": "user", "content": question},
    ]

    # Give the LLM the OCR text so it can answer questions about the document.
    if document:
        messages.insert(1, document_message(document))

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
    print(f"Using Ollama model: {MODEL} (tool router: {ROUTER_MODEL})")
    print(f"Connecting to MCP server at {SERVER_URL} ...")

    async with streamable_http_client(SERVER_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Connected to MCP server.")
            print("Ask a question, or type 'exit' to quit.")
            print("Attach an image/PDF with '/image <path>', remove it with '/clear'.")

            document = None

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

                if question.startswith("/image"):
                    print(f"Extracting text with {OCR_MODEL} ...")
                    try:
                        document = await extract_text(
                            question.removeprefix("/image").strip()
                        )
                        print(
                            f"Attached {document['name']} "
                            f"({len(document['text'])} characters extracted)."
                        )
                    except Exception as exc:
                        print(f"Could not attach file: {exc}")
                    continue

                if question == "/clear":
                    document = None
                    print("Attachment removed.")
                    continue

                try:
                    answer = await ask_mcp(session, question, document)
                    print(f"\nAssistant: {answer}")
                except Exception as exc:
                    print(f"\nError: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
