You: "How much does Example Corp owe in total?"
 │
 ① tev1 (router)          select_tool()
 │   question + tool descriptions  →  "total_due_by_client"
 │
 ② LLM call #1             fill_arguments()
 │   question + tool docstring + JSON schema
 │   →  {"client_name": "Example Corp"}          ← question becomes function parameters
 │
 ③ Tool call (MCP)         session.call_tool()
 │   client ──HTTP──▶ server: total_due_by_client(client_name="Example Corp")
 │   server runs SQL  →  {"totals": [{"currency": "USD", "total": 150.25}, ...]}
 │
 │   (if the server rejects the arguments → back to ② with the error message, at most once)
 │
 ④ LLM call #2             run_selected_tool()
 │   question + tool result  →  "Example Corp owes USD 150.25 and INR 5000."
 ▼
Assistant: ...
