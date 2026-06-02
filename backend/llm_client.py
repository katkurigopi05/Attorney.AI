"""
Attorney.AI — Unified LLM Client
Supports Ollama (local, free) and OpenAI (optional, if key is set).

Priority:
  1. If OPENAI_API_KEY is set → use OpenAI
  2. Otherwise             → use Ollama (http://localhost:11434)

Ollama exposes an OpenAI-compatible API at /v1, so we use the
openai Python library pointing at localhost — zero code change needed.

Recommended free Ollama models (pull one before running):
  ollama pull llama3.1          # 8B, best quality (~5GB)
  ollama pull mistral           # 7B, fast (~4GB)
  ollama pull phi3:mini         # 3.8B, very lightweight (~2.3GB)
  ollama pull qwen2.5:7b        # 7B, strong reasoning (~5GB)
"""
from functools import lru_cache
from typing import Optional

from loguru import logger
from openai import AsyncOpenAI

from config import settings


@lru_cache(maxsize=1)
def get_llm_client() -> AsyncOpenAI:
    """
    Return the appropriate LLM client.
    - Ollama by default (free, local)
    - OpenAI if OPENAI_API_KEY is configured
    """
    if settings.openai_api_key and settings.openai_api_key.startswith("sk-"):
        logger.info(f"LLM backend: OpenAI ({settings.openai_chat_model})")
        return AsyncOpenAI(api_key=settings.openai_api_key)
    else:
        logger.info(f"LLM backend: Ollama @ {settings.ollama_base_url} ({settings.ollama_model})")
        return AsyncOpenAI(
            base_url=f"{settings.ollama_base_url}/v1",
            api_key="ollama",  # Ollama doesn't need a real key
        )


def get_chat_model() -> str:
    """Return the chat model name to use."""
    if settings.openai_api_key and settings.openai_api_key.startswith("sk-"):
        return settings.openai_chat_model
    return settings.ollama_model


async def chat_complete(
    messages: list,
    temperature: float = 0.1,
    max_tokens: int = 1500,
    json_mode: bool = False,
) -> str:
    """
    Unified chat completion call.
    Automatically routes to Ollama or OpenAI.
    Returns the response content string.
    """
    client = get_llm_client()
    model = get_chat_model()

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # JSON mode: OpenAI supports response_format; Ollama supports it in newer versions
    if json_mode:
        try:
            kwargs["response_format"] = {"type": "json_object"}
        except Exception:
            pass  # Older Ollama versions ignore this gracefully

    resp = await client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content.strip()
