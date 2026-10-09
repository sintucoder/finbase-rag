"""The only file that talks to the LLM provider.

It uses the `openai` package, which also works with other providers that expose
an OpenAI-compatible endpoint (Gemini, Groq, OpenRouter, a local Ollama, ...).
Switching provider = changing LLM_API_KEY / LLM_BASE_URL / LLM_MODEL in .env.
To use a provider with a different SDK, only this file needs to change.
"""
from openai import OpenAI

from src import config


class LLMClient:
    def __init__(self, api_key: str = config.LLM_API_KEY, base_url: str | None = config.LLM_BASE_URL,
                 model: str = config.LLM_MODEL):
        if not api_key:
            raise RuntimeError(
                "LLM_API_KEY is not set. Copy .env.example to .env and add your API key "
                "(see the README, section 'Environment variables')."
            )
        if not model:
            raise RuntimeError(
                "LLM_MODEL is not set. Put a current model id from your provider in .env "
                "(model ids change over time, so there is no built-in default)."
            )
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0, max_retries=2)

    def chat(self, messages: list[dict], temperature: float = 0.0,
             max_tokens: int = config.LLM_MAX_TOKENS) -> str:
        """Send chat messages, return the reply text. Temperature 0 = as deterministic as possible."""
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()
