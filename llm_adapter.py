"""
llm_adapter.py
--------------
Unified LLM client factory supporting Ollama, OpenRouter, and KiloCode.
All providers expose an OpenAI-compatible /v1/chat/completions endpoint.

Provider selection and model defaults are driven by llm_config.yaml.
Default free models:
  - openrouter : nvidia/nemotron-3-super-120b-a12b:free
  - ollama     : qwen3:latest (resolves to largest variant the host can serve;
                 on 16 GB GPU → qwen3:30b-a3b MoE; on 8 GB GPU → qwen3:8b;
                 run `ollama pull qwen3:latest` once before first use)
  - kilo       : moonshotai/kmoonshot-v1-8k (check Kilo free tier)
"""

import os
import yaml
from openai import OpenAI

CONFIG_FILE = "llm_config.yaml"

# ---------------------------------------------------------------------------
# Default models per provider (largest available on free tier, April 2026)
# ---------------------------------------------------------------------------
DEFAULT_MODELS = {
    "openrouter": "nvidia/nemotron-3-super-120b-a12b:free",
    "ollama":     "qwen3:latest",
    "kilo": "kilo/auto-free",
}

DEFAULT_BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama":     "http://localhost:11434/v1",
    "kilo": "kilo/auto-free",
}

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_llm_config() -> dict:
    """Load llm_config.yaml, falling back to env vars and defaults."""
    cfg = {}
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    # Env var takes precedence over yaml so CI secrets can override without editing config
    provider = (os.getenv("LLM_PROVIDER") or cfg.get("provider") or "openrouter").lower().strip()

    model = cfg.get("model") or os.getenv("LLM_MODEL") or DEFAULT_MODELS.get(provider, "")
    # Provider-specific base URL env vars (e.g. OLLAMA_BASE_URL) take precedence over generic LLM_BASE_URL
    provider_url_env = {
        "ollama": "OLLAMA_BASE_URL",
    }
    base_url = (
        cfg.get("base_url")
        or os.getenv(provider_url_env.get(provider, ""), "")
        or os.getenv("LLM_BASE_URL", "")
        or DEFAULT_BASE_URLS.get(provider, "")
    )
    temperature = float(cfg.get("temperature", 0.3))
    max_tokens = int(cfg.get("max_tokens", 1024))

    # API key: config file → env var per provider → generic LLM_API_KEY
    api_key_env_map = {
        "openrouter": "OPENROUTER_API_KEY",
        "ollama":     "OLLAMA_API_KEY",   # usually not needed; ollama accepts any string
        "kilo": "kilo/auto-free",
    }
    api_key = (
        cfg.get("api_key")
        or os.getenv(api_key_env_map.get(provider, ""))
        or os.getenv("LLM_API_KEY")
        or "ollama"  # Ollama local doesn't validate the key
    )

    return {
        "provider":    provider,
        "model":       model,
        "base_url":    base_url,
        "api_key":     api_key,
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def get_client() -> tuple:
    """
    Returns (OpenAI client, config dict).
    The config dict contains model, temperature, max_tokens for use in calls.
    """
    cfg = load_llm_config()
    provider = cfg["provider"]

    extra_headers = {}
    if provider == "openrouter":
        # OpenRouter recommends these headers for tracking / free-tier priority
        extra_headers = {
            "HTTP-Referer": "https://github.com/TeamDailyPaper",
            "X-Title":      "TeamDailyPaper",
        }

    client = OpenAI(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        default_headers=extra_headers if extra_headers else None,
    )

    print(f"[LLM] Provider: {provider} | Model: {cfg['model']} | Base URL: {cfg['base_url']}")
    return client, cfg


# ---------------------------------------------------------------------------
# Convenience call wrapper
# ---------------------------------------------------------------------------

def chat(system_prompt: str, user_prompt: str, client=None, cfg=None) -> str:
    """
    Single chat completion call.
    Returns the assistant's reply as a plain string.
    """
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
    return response.choices[0].message.content.strip()
