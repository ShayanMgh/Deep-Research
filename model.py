# llama3_agent.py
import json, os
import ollama
from search_tool import web_search_digest
import gradio as gr

MODEL = "llama3.2"
SYSTEM = (
    "You are a meticulous research assistant.\n"
    "When the user asks a factual question, decide if you need to call "
    "`do_search` to collect fresh web context. "
    "After receiving the tool result, produce a concise 2–3-paragraph summary "
    "(≤300 words). If the answer is unrelated to current events you can reply directly."
)

# ----- 1) declare the tool schema -----
search_tool_schema = {
    "name": "do_search",
    "description": "Search the public web and return raw snippets to help answer recent-facts questions.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": { "type": "string", "description": "The search query" }
        },
        "required": ["query"],
    },
}
TOOLS = [{"type": "function", "function": search_tool_schema}]

# ----- 2) helper that executes tool calls -----
def maybe_run_tool(message):
    """
    If the assistant asked to call our `do_search` tool, run it and build
    the tool-response message needed by Ollama.
    """
    tool_calls = message.get("tool_calls", [])

    if not tool_calls:
        return None                      # model did not request a tool

    call = tool_calls[0]                 # ↩️ only supporting one call
    # ---- extract arguments no matter how they're wrapped ----
    if "function" in call:               # most recent schema
        fn_name = call["function"]["name"]
        arguments = call["function"]["arguments"]
    else:                                # very old schema
        fn_name = call["name"]
        arguments = call["arguments"]

    if fn_name != "do_search":
        return None                      # not our tool

    query = arguments.get("query", "")
    digest = web_search_digest(query)

    # Some Ollama builds include an id, some don't
    tool_reply = {
        "role": "tool",
        "name": "do_search",
        "content": digest,
    }
    if "id" in call:
        tool_reply["tool_call_id"] = str(call["id"])

    return tool_reply

# ----- 3) chat loop used by Gradio -----
def chat(user_message, history):
    messages = (
        [{"role": "system", "content": SYSTEM}] +
        history +
        [{"role": "user", "content": user_message}]
    )

    # first pass (LLM may request the search tool)
    response = ollama.chat(model=MODEL, messages=messages, tools=TOOLS)
    tool_reply = maybe_run_tool(response["message"])

    # if tool was called, append its result and ask the LLM again
    if tool_reply:
        messages.append(response["message"])
        messages.append(tool_reply)
        response = ollama.chat(model=MODEL, messages=messages)   # second pass

    return response["message"]["content"]

# ----- 4) launch simple Gradio UI -----
gr.ChatInterface(fn=chat, type="messages", title="Llama-3 Research Bot").launch()
