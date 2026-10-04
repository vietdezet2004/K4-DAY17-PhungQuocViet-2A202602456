from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Provider aliases – maps every reasonable variant to the canonical name
# ---------------------------------------------------------------------------
_ALIASES: dict[str, str] = {
    # openai
    "openai": "openai",
    "open_ai": "openai",
    "open-ai": "openai",
    "gpt": "openai",
    # custom / openai-compatible
    "custom": "custom",
    "custom_openai": "custom",
    "custom-openai": "custom",
    # gemini
    "gemini": "gemini",
    "google": "gemini",
    "google_gemini": "gemini",
    "google-gemini": "gemini",
    # anthropic
    "anthropic": "anthropic",
    "anthorpic": "anthropic",   # common typo kept in scaffold docstring
    "claude": "anthropic",
    # ollama
    "ollama": "ollama",
    # openrouter
    "openrouter": "openrouter",
    "open_router": "openrouter",
    "open-router": "openrouter",
}

_SUPPORTED = {"openai", "custom", "gemini", "anthropic", "ollama", "openrouter"}


@dataclass
class ProviderConfig:
    """Configuration shared by both agents.

    Fields
    ------
    provider    : canonical provider name (normalised before use)
    model_name  : model identifier string, e.g. "gpt-4o-mini"
    temperature : sampling temperature (0.0 = deterministic)
    api_key     : optional API key (may also come from env vars)
    base_url    : optional base URL override (required for "custom" / "ollama")
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Map any known alias to the canonical provider name.

    Parameters
    ----------
    value : raw provider string from env var or config

    Returns
    -------
    Canonical lowercase provider name.

    Raises
    ------
    ValueError if the alias is unknown.
    """
    key = value.strip().lower()
    canonical = _ALIASES.get(key)
    if canonical is None:
        raise ValueError(
            f"Unknown provider {value!r}. "
            f"Supported providers: {sorted(_SUPPORTED)}"
        )
    return canonical


def build_chat_model(config: ProviderConfig):
    """Instantiate and return a LangChain chat model for the given provider.

    The function resolves the provider alias first so callers do not need to
    normalise the string themselves.

    Raises
    ------
    ImportError  if the required LangChain integration package is missing.
    ValueError   if the provider is not recognised.
    """
    provider = normalize_provider(config.provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI  # type: ignore

        kwargs: dict = dict(
            model=config.model_name,
            temperature=config.temperature,
        )
        if config.api_key:
            kwargs["openai_api_key"] = config.api_key
        return ChatOpenAI(**kwargs)

    if provider == "custom":
        from langchain_openai import ChatOpenAI  # type: ignore

        kwargs = dict(
            model=config.model_name,
            temperature=config.temperature,
        )
        if config.base_url:
            kwargs["base_url"] = config.base_url
        if config.api_key:
            kwargs["openai_api_key"] = config.api_key
        return ChatOpenAI(**kwargs)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore

        kwargs = dict(
            model=config.model_name,
            temperature=config.temperature,
        )
        if config.api_key:
            kwargs["google_api_key"] = config.api_key
        return ChatGoogleGenerativeAI(**kwargs)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic  # type: ignore

        kwargs = dict(
            model=config.model_name,
            temperature=config.temperature,
        )
        if config.api_key:
            kwargs["anthropic_api_key"] = config.api_key
        return ChatAnthropic(**kwargs)

    if provider == "ollama":
        from langchain_ollama import ChatOllama  # type: ignore

        kwargs = dict(
            model=config.model_name,
            temperature=config.temperature,
        )
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOllama(**kwargs)

    if provider == "openrouter":
        try:
            from langchain_openrouter import ChatOpenRouter  # type: ignore

            kwargs = dict(
                model=config.model_name,
                temperature=config.temperature,
            )
            if config.api_key:
                kwargs["openrouter_api_key"] = config.api_key
            return ChatOpenRouter(**kwargs)
        except ImportError:
            # Fallback: OpenRouter is OpenAI-compatible
            from langchain_openai import ChatOpenAI  # type: ignore

            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                base_url="https://openrouter.ai/api/v1",
                openai_api_key=config.api_key or "sk-or-placeholder",
            )

    # Should never reach here after normalize_provider(), but keeps mypy happy.
    raise ValueError(f"Unhandled provider: {provider!r}")
