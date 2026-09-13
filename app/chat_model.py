"""The chat model, built from the settings.

Every provider here speaks the OpenAI chat completions API, so one client covers
DeepSeek, an OpenAI model and a model running on your own machine (vLLM, Ollama,
unsloth). What changes between them is the base URL, the model name and the key —
`OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENAI_API_KEY`, the names the OpenAI client
already reads, so a local server is pointed at the same way as a hosted one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.config import Settings, settings

# A local server wants a key on the wire and ignores its value.
LOCAL_API_KEY = "not-needed"


@dataclass(frozen=True)
class ChatProfile:
    """Where a provider is reached by default, and what it needs on the wire."""

    base_url: str
    model: str
    # Used instead of OPENAI_API_KEY: a local server ignores the key and has no
    # business being handed a real one.
    api_key_default: str = ""
    # Provider-specific parameters that do not belong in the graph.
    extra_body: dict[str, Any] = field(default_factory=dict)


PRESETS: dict[str, ChatProfile] = {
    "deepseek": ChatProfile(
        base_url="https://api.deepseek.com/v1",
        model="deepseek-v4-flash",
        # Keeps the console on the final answer rather than on DeepSeek's
        # internal reasoning stream, which is billed like any other token.
        # This body travels with this preset: a model elsewhere is reached by
        # setting CHAT_PROVIDER to "openai" or to "local".
        extra_body={"thinking": {"type": "disabled"}},
    ),
    "openai": ChatProfile(
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
    ),
    "local": ChatProfile(
        base_url="http://localhost:8000/v1",
        model="",
        api_key_default=LOCAL_API_KEY,
    ),
}


def build_chat_model(config: Settings = settings) -> BaseChatModel:
    """Build the model the graph answers with.

    Nothing is contacted here: the model is only a client until a question is
    asked. A missing key or a model name left empty fails now, with the variable
    to set, rather than on the first answer.
    """
    provider = config.chat_provider.strip().lower()
    profile = PRESETS.get(provider)
    if profile is None:
        raise RuntimeError(
            f"Unknown CHAT_PROVIDER: {config.chat_provider!r}. "
            f"Use one of: {', '.join(PRESETS)}."
        )

    model = config.openai_model or profile.model
    if not model:
        raise RuntimeError(
            f"Missing environment variables: OPENAI_MODEL "
            f"(CHAT_PROVIDER is {provider!r}, which has no default model)"
        )

    # The preset's placeholder wins where there is one: what is under
    # OPENAI_API_KEY belongs to the endpoint the chat is pointed at.
    api_key = profile.api_key_default or config.openai_api_key
    if not api_key:
        raise RuntimeError(
            f"Missing environment variables: OPENAI_API_KEY "
            f"(needed by CHAT_PROVIDER={provider!r})"
        )

    return ChatOpenAI(
        model=model,
        base_url=config.openai_base_url or profile.base_url,
        api_key=api_key,
        temperature=0,
        max_retries=2,
        extra_body=dict(profile.extra_body) or None,
    )
