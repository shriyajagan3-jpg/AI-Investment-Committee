"""
llm.py - a thin wrapper around the language model.

Every agent talks to the model through `LLM.chat()`. Keeping this interface
tiny means the rest of the code doesn't care which provider is behind it, and
tests can swap in a fake model that needs no API key.
"""

import json
import os


class LLM:
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Returns {"content": str | None,
                    "tool_calls": [{"id", "name", "arguments": dict}],
                    "message": <assistant message to append to history>}"""
        raise NotImplementedError


class OpenAILLM(LLM):
    def __init__(self, model: str | None = None, temperature: float = 0.3):
        from openai import OpenAI  # imported here so tests don't need the package configured
        self.client = OpenAI()     # reads OPENAI_API_KEY from the environment
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature

    def chat(self, messages, tools=None):
        kwargs = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if tools:
            kwargs["tools"] = tools
        msg = self.client.chat.completions.create(**kwargs).choices[0].message

        calls = [{"id": c.id, "name": c.function.name, "arguments": json.loads(c.function.arguments or "{}")}
                 for c in (msg.tool_calls or [])]
        history_msg = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            history_msg["tool_calls"] = [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in msg.tool_calls]
        return {"content": msg.content, "tool_calls": calls, "message": history_msg}
