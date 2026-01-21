"""Tool schema converters for different LLM providers."""

from typing import Any


def anthropic_to_gemini(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert Anthropic tool format to Gemini function declarations.

    Anthropic format:
    {
        "name": "tool_name",
        "description": "...",
        "input_schema": {
            "type": "object",
            "properties": {...},
            "required": [...]
        }
    }

    Gemini format:
    {
        "name": "tool_name",
        "description": "...",
        "parameters": {
            "type": "object",
            "properties": {...},
            "required": [...]
        }
    }
    """
    gemini_tools = []
    for tool in tools:
        gemini_tool = {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {
                "type": "object",
                "properties": {},
            }),
        }
        gemini_tools.append(gemini_tool)
    return gemini_tools


def anthropic_to_groq(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert Anthropic tool format to Groq/OpenAI function format.

    Anthropic format:
    {
        "name": "tool_name",
        "description": "...",
        "input_schema": {...}
    }

    Groq/OpenAI format:
    {
        "type": "function",
        "function": {
            "name": "tool_name",
            "description": "...",
            "parameters": {...}
        }
    }
    """
    groq_tools = []
    for tool in tools:
        groq_tool = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {
                    "type": "object",
                    "properties": {},
                }),
            },
        }
        groq_tools.append(groq_tool)
    return groq_tools


def convert_messages_to_gemini(
    messages: list[dict[str, Any]],
    system_prompt: str,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Convert standard messages to Gemini format.

    Standard format (Anthropic-style):
    [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": [{"type": "tool_result", ...}]}
    ]

    Gemini format:
    [
        {"role": "user", "parts": [{"text": "..."}]},
        {"role": "model", "parts": [{"text": "..."}]},
    ]

    Returns:
        Tuple of (system_instruction, converted_messages)
    """
    gemini_messages = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Map roles
        gemini_role = "model" if role == "assistant" else "user"

        # Handle content
        if isinstance(content, str):
            parts = [{"text": content}]
        elif isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append({"text": item.get("text", "")})
                    elif item.get("type") == "tool_use":
                        # Function call from assistant
                        parts.append({
                            "functionCall": {
                                "name": item.get("name", ""),
                                "args": item.get("input", {}),
                            }
                        })
                    elif item.get("type") == "tool_result":
                        # Tool result from user
                        result_content = item.get("content", "")
                        if isinstance(result_content, str):
                            result_text = result_content
                        else:
                            result_text = str(result_content)
                        parts.append({
                            "functionResponse": {
                                "name": item.get("tool_use_id", "unknown"),
                                "response": {"result": result_text},
                            }
                        })
                else:
                    parts.append({"text": str(item)})
            if not parts:
                parts = [{"text": ""}]
        else:
            parts = [{"text": str(content)}]

        gemini_messages.append({
            "role": gemini_role,
            "parts": parts,
        })

    return system_prompt, gemini_messages


def convert_messages_to_groq(
    messages: list[dict[str, Any]],
    system_prompt: str,
) -> list[dict[str, Any]]:
    """
    Convert standard messages to Groq/OpenAI format.

    Standard format (Anthropic-style):
    [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": [{"type": "tool_result", ...}]}
    ]

    Groq/OpenAI format:
    [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "...", "tool_calls": [...]},
        {"role": "tool", "tool_call_id": "...", "content": "..."},
    ]
    """
    groq_messages = []

    # Add system message first
    if system_prompt:
        groq_messages.append({
            "role": "system",
            "content": system_prompt,
        })

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            groq_messages.append({
                "role": role,
                "content": content,
            })
        elif isinstance(content, list):
            # Complex content with potential tool calls/results
            text_parts = []
            tool_calls = []
            tool_results = []

            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type", "")
                    if item_type == "text":
                        text_parts.append(item.get("text", ""))
                    elif item_type == "tool_use":
                        # Assistant made a tool call
                        import json
                        tool_calls.append({
                            "id": item.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": item.get("name", ""),
                                "arguments": json.dumps(item.get("input", {})),
                            },
                        })
                    elif item_type == "tool_result":
                        # Tool result
                        result_content = item.get("content", "")
                        if not isinstance(result_content, str):
                            result_content = str(result_content)
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": item.get("tool_use_id", ""),
                            "content": result_content,
                        })

            # Add assistant message with tool calls
            if role == "assistant":
                assistant_msg = {"role": "assistant"}
                if text_parts:
                    assistant_msg["content"] = " ".join(text_parts)
                else:
                    assistant_msg["content"] = None
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                groq_messages.append(assistant_msg)

            # Add tool results
            for result in tool_results:
                groq_messages.append(result)

            # If user message with only text
            if role == "user" and text_parts and not tool_results:
                groq_messages.append({
                    "role": "user",
                    "content": " ".join(text_parts),
                })

    return groq_messages
