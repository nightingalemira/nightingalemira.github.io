"""
llm_adapter.py
--------------
Unified LLM client factory supporting three providers:
  - ollama      : local Ollama server (OpenAI-compatible)
  - openrouter  : OpenRouter API (large free models)
  - kilo        : KiloCode API (OpenAI-compatible)

Usage:
    client, cfg = get_client()          # uses LLM_CONFIG_FILE
    response    = chat(sys, usr, client=client, cfg=cfg)
"""

import os
import yaml
from openai import OpenAI

LLM_CONFIG_FILE = "llm_config.yaml"

# Default (largest free) models per provider
DEFAULT_MODELS = {
    "ollama":      "llama3.3:70b",
    "openrouter":  "nvidia/nemotron-3-super-120b-a12b:free",
    "kilo":        "kilo/free-large",
}

DEFAULT_BASE_URLS = {
    "ollama":     os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    "openrouter": "https://openrouter.ai/api/v1",
    "kilo":       "https://api.kilo.codes/v1",
}


def load_llm_config() -> dict:
    if not os.path.exists(LLM_CONFIG_FILE):
        return {}
    with open(LLM_CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_client(provider: str = None):
    """
    Returns (OpenAI client, cfg dict).
    cfg contains: model, temperature, max_tokens, provider.
    """
    file_cfg  = load_llm_config()
    provider  = provider or file_cfg.get("provider", "openrouter")

    model = file_cfg.get("model") or DEFAULT_MODELS.get(provider, "gpt-4o-mini")

    api_key = None
    base_url = DEFAULT_BASE_URLS.get(provider)

    if provider == "ollama":
        api_key  = os.getenv("OLLAMA_API_KEY", "ollama")
        base_url = os.getenv("OLLAMA_BASE_URL", base_url)
    elif provider == "openrouter":
        api_key  = os.getenv("OPENROUTER_API_KEY", "")
        base_url = os.getenv("OPENROUTER_BASE_URL", base_url)
    elif provider == "kilo":
        api_key  = os.getenv("KILO_API_KEY", "")
        base_url = os.getenv("KILO_BASE_URL", base_url)
    else:
        raise ValueError(f"Unknown LLM provider: '{provider}'")

    client = OpenAI(api_key=api_key or "no-key", base_url=base_url)
    cfg = {
        "provider":   provider,
        "model":      model,
        "temperature": float(file_cfg.get("temperature", 0.3)),
        "max_tokens":  int(file_cfg.get("max_tokens", 1024)),
    }
    return client, cfg


def chat(system_prompt: str, user_prompt: str, client=None, cfg: dict = None) -> str:
    """Send a single chat turn and return the assistant's text."""
    if client is None or cfg is None:
        client, cfg = get_client()
    response = client.chat.completions.create(
        model=cfg["model"],
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )
    return response.choices[0].message.content
