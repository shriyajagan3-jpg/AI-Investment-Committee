"""
llm.py - a thin wrapper around the language model.

Every agent talks to the model through `LLM.chat()`. Keeping this interface
tiny means the rest of the code doesn't care which provider is behind it, and
tests can swap in a fake model that needs no API key.

Providers (picked from environment variables):
  * GEMINI_API_KEY set  -> Google Gemini (has a free tier), via its
                           OpenAI-compatible endpoint
  * OPENAI_API_KEY set  -> OpenAI
LLM_MODEL overrides the default model name for either provider.
"""

import json
import os

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class LLM:
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Returns {"content": str | None,
                    "tool_calls": [{"id", "name", "arguments": dict}],
                    "message": <assistant message to append to history>}"""
        raise NotImplementedError


class OpenAICompatibleLLM(LLM):
    def __init__(self, model: str | None = None, temperature: float = 0.3):
        from openai import OpenAI  # imported here so tests don't need the package configured

        if os.environ.get("GEMINI_API_KEY"):
            self.client = OpenAI(api_key=os.environ["GEMINI_API_KEY"], base_url=GEMINI_BASE_URL,
                                 max_retries=6)  # free tier has per-minute limits; the SDK backs off and retries
            default_model = "gemini-flash-latest"
        elif os.environ.get("OPENAI_API_KEY"):
            self.client = OpenAI(max_retries=4)
            default_model = "gpt-4o-mini"
        else:
            raise RuntimeError("Set GEMINI_API_KEY (free) or OPENAI_API_KEY. See README.")
        self.model = model or os.environ.get("LLM_MODEL", default_model)
        self.temperature = temperature

    def chat(self, messages, tools=None):
        kwargs = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if tools:
            kwargs["tools"] = tools
        msg = self.client.chat.completions.create(**kwargs).choices[0].message

        calls = [{"id": c.id, "name": c.function.name, "arguments": json.loads(c.function.arguments or "{}")}
                 for c in (msg.tool_calls or [])]
        # Send the assistant message back exactly as received (including any provider-specific
        # extra fields, e.g. Gemini's thought signatures) so multi-step tool calling keeps working.
        history_msg = msg.model_dump(exclude_none=True)
        history_msg.setdefault("content", msg.content)
        return {"content": msg.content, "tool_calls": calls, "message": history_msg}


# Backwards-compatible name used by app.py
OpenAILLM = OpenAICompatibleLLM
