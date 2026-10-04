from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider

# ---------------------------------------------------------------------------
# Try to load .env if python-dotenv is available (optional – not a hard dep)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv as _load_dotenv  # type: ignore

    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False


@dataclass
class LabConfig:
    """Shared configuration for the Day-17 Memory Systems lab.

    Paths
    -----
    base_dir  : repo root (all other paths are relative to this)
    data_dir  : benchmark input JSON files
    state_dir : runtime state written by agents (User.md files, etc.)

    Compact-memory
    --------------
    compact_threshold_tokens : token budget before an older message block
                               is compressed into a summary  (default 800)
    compact_keep_messages    : number of most-recent messages to keep intact
                               after each compaction  (default 4)

    Models
    ------
    model       : main chat model used by both agents
    judge_model : secondary model used to evaluate quality (may equal `model`)
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


# ---------------------------------------------------------------------------
# Defaults – used when env vars are absent
# ---------------------------------------------------------------------------
_DEFAULT_PROVIDER = "openai"
_DEFAULT_MODEL = "gpt-4o-mini"
_DEFAULT_TEMPERATURE = 0.0
_DEFAULT_COMPACT_THRESHOLD = 800   # tokens before compaction kicks in
_DEFAULT_COMPACT_KEEP = 4          # recent messages to keep after compaction


def _get(key: str, default: str = "") -> str:
    """Return stripped env var value or *default*."""
    return os.environ.get(key, default).strip()


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a fully-populated :class:`LabConfig`.

    Resolution order
    ----------------
    1. Resolve repo root: *base_dir* argument  →  parent of this file's parent.
    2. Load ``.env`` from repo root (if python-dotenv is installed and file exists).
    3. Read env vars; fall back to sensible offline-safe defaults so the lab
       can run without any API key in offline / test mode.
    4. Create ``state/`` directory tree if it does not exist.
    5. Return a :class:`LabConfig` instance.

    Supported env vars
    ------------------
    LLM_PROVIDER          one of: openai, custom, gemini, anthropic, ollama, openrouter
    LLM_MODEL             model identifier string  (e.g. gpt-4o-mini)
    LLM_TEMPERATURE       float, defaults to 0.0
    OPENAI_API_KEY
    GEMINI_API_KEY
    ANTHROPIC_API_KEY
    OLLAMA_BASE_URL
    OPENROUTER_API_KEY
    CUSTOM_BASE_URL
    CUSTOM_API_KEY
    JUDGE_PROVIDER        defaults to LLM_PROVIDER
    JUDGE_MODEL           defaults to LLM_MODEL
    COMPACT_THRESHOLD     int, token threshold for compaction  (default 800)
    COMPACT_KEEP          int, messages to keep after compaction  (default 4)
    """
    # ------------------------------------------------------------------ root
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    # -------------------------------------------------- optional .env loader
    if _DOTENV_AVAILABLE:
        env_file = root / ".env"
        if env_file.exists():
            _load_dotenv(env_file, override=False)  # do not override already-set vars

    # -------------------------------------------------- provider / model
    raw_provider = _get("LLM_PROVIDER", _DEFAULT_PROVIDER)
    provider = normalize_provider(raw_provider)

    model_name = _get("LLM_MODEL", _DEFAULT_MODEL)

    try:
        temperature = float(_get("LLM_TEMPERATURE", str(_DEFAULT_TEMPERATURE)))
    except ValueError:
        temperature = _DEFAULT_TEMPERATURE

    # Resolve API key / base URL depending on provider
    api_key: str | None = None
    base_url: str | None = None

    if provider == "openai":
        api_key = _get("OPENAI_API_KEY") or None
    elif provider == "gemini":
        api_key = _get("GEMINI_API_KEY") or None
    elif provider == "anthropic":
        api_key = _get("ANTHROPIC_API_KEY") or None
    elif provider == "ollama":
        base_url = _get("OLLAMA_BASE_URL", "http://localhost:11434") or None
    elif provider == "openrouter":
        api_key = _get("OPENROUTER_API_KEY") or None
    elif provider == "custom":
        base_url = _get("CUSTOM_BASE_URL") or None
        api_key = _get("CUSTOM_API_KEY") or None

    main_model = ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )

    # -------------------------------------------------- judge model
    raw_judge_provider = _get("JUDGE_PROVIDER", raw_provider)
    judge_provider = normalize_provider(raw_judge_provider)
    judge_model_name = _get("JUDGE_MODEL", model_name)

    # Reuse the same api_key / base_url logic for the judge
    judge_api_key: str | None = api_key
    judge_base_url: str | None = base_url
    if judge_provider != provider:
        # Different provider for judge — resolve its own credentials
        judge_api_key = None
        judge_base_url = None
        if judge_provider == "openai":
            judge_api_key = _get("OPENAI_API_KEY") or None
        elif judge_provider == "gemini":
            judge_api_key = _get("GEMINI_API_KEY") or None
        elif judge_provider == "anthropic":
            judge_api_key = _get("ANTHROPIC_API_KEY") or None
        elif judge_provider == "ollama":
            judge_base_url = _get("OLLAMA_BASE_URL", "http://localhost:11434") or None
        elif judge_provider == "openrouter":
            judge_api_key = _get("OPENROUTER_API_KEY") or None
        elif judge_provider == "custom":
            judge_base_url = _get("CUSTOM_BASE_URL") or None
            judge_api_key = _get("CUSTOM_API_KEY") or None

    judge_cfg = ProviderConfig(
        provider=judge_provider,
        model_name=judge_model_name,
        temperature=0.0,  # judge always deterministic
        api_key=judge_api_key,
        base_url=judge_base_url,
    )

    # -------------------------------------------------- compact memory
    try:
        compact_threshold = int(_get("COMPACT_THRESHOLD", str(_DEFAULT_COMPACT_THRESHOLD)))
    except ValueError:
        compact_threshold = _DEFAULT_COMPACT_THRESHOLD

    try:
        compact_keep = int(_get("COMPACT_KEEP", str(_DEFAULT_COMPACT_KEEP)))
    except ValueError:
        compact_keep = _DEFAULT_COMPACT_KEEP

    # -------------------------------------------------- directories
    data_dir = root / "data"
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)           # create if missing
    (state_dir / "profiles").mkdir(parents=True, exist_ok=True)

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold,
        compact_keep_messages=compact_keep,
        model=main_model,
        judge_model=judge_cfg,
    )
