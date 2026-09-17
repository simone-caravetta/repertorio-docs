"""The chat model, built from the settings.

Every provider here speaks the OpenAI chat completions API, so one client covers
DeepSeek, an OpenAI model and a model running on your own machine. What changes
between them is the base URL, the model name and the key, under the names the
OpenAI client already reads.
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
    # Used in place of the key in OPENAI_API_KEY. A local server ignores the key
    # and has no reason to be handed a real one.
    api_key_default: str = ""
    # Parameters for one provider, which the graph does not need to know about.
    extra_body: dict[str, Any] = field(default_factory=dict)


PRESETS: dict[str, ChatProfile] = {
    "deepseek": ChatProfile(
        base_url="https://api.deepseek.com/v1",
        model="deepseek-v4-flash",
        # Keeps the console on the final answer instead of the reasoning stream
        # of DeepSeek, which is billed like any other token. The body travels
        # with this preset, and a model elsewhere is reached by setting
        # CHAT_PROVIDER to "openai" or "local".
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

    Nothing is contacted here, because the model is only a client until a
    question is asked. A missing key or an empty model name fails at this point
    with the variable to set, instead of failing on the first answer.
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

    # Where the preset has a placeholder for the key it wins, because the key in
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
