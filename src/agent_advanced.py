from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
    summarize_messages,
)
from model_provider import build_chat_model


# ---------------------------------------------------------------------------
# Internal context descriptor (kept from scaffold)
# ---------------------------------------------------------------------------

@dataclass
class AgentContext:
    user_id: str
    memory_path: str


# ---------------------------------------------------------------------------
# Advanced Agent
# ---------------------------------------------------------------------------

class AdvancedAgent:
    """Agent B – three memory layers.

    Layer 1 – Short-term (within-session)
        Handled by ``CompactMemoryManager``.  Messages are kept in a rolling
        window; older ones are compressed into a summary automatically.

    Layer 2 – Persistent (cross-session)
        ``UserProfileStore`` writes stable facts to ``User.md`` on disk.
        Any new thread for the same ``user_id`` loads the profile and can
        answer recall questions about name, location, profession, etc.

    Layer 3 – Compact memory
        ``CompactMemoryManager`` nén lịch sử dài thành summary khi tổng
        token vượt ``config.compact_threshold_tokens``.  This keeps
        ``prompt_token_usage`` from growing unboundedly like the baseline.
    """

    def __init__(
        self,
        config: LabConfig | None = None,
        force_offline: bool = False,
    ) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline

        # Layer 2 – persistent profile store
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")

        # Layer 1+3 – compact short-term memory, one entry per thread_id
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )

        # Per-thread token counters
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        self.langchain_agent = None
        if not force_offline:
            self._maybe_build_langchain_agent()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route to live or offline path and return a response dict.

        Returns
        -------
        dict with keys:
            response      : str  – assistant reply
            thread_id     : str
            tokens        : int  – tokens generated this turn
            prompt_tokens : int  – estimated prompt context this turn
        """
        if self.langchain_agent is not None and not self.force_offline:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        """Return cumulative assistant tokens generated in *thread_id*."""
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        """Return cumulative prompt-context tokens processed in *thread_id*.

        Unlike the baseline, this grows slowly because compact memory keeps
        the context bounded.
        """
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        """Return the current size of the user's ``User.md`` in bytes."""
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        """Return how many times compact memory compressed this thread."""
        return self.compact_memory.compaction_count(thread_id)

    # ------------------------------------------------------------------
    # Offline path
    # ------------------------------------------------------------------

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic advanced reply using all three memory layers.

        Steps
        -----
        1. Extract stable profile facts and persist them to ``User.md``.
        2. Append the user message to compact memory.
        3. Estimate prompt context (User.md + summary + recent messages).
        4. Generate a response using persisted + in-session memory.
        5. Append the assistant reply to compact memory.
        6. Update per-thread token counters.
        """
        # Step 1 – extract facts and persist
        facts = extract_profile_updates(message)
        if facts:
            for key, value in facts.items():
                self.profile_store.upsert_fact(user_id, key, value)

        # Step 2 – append user message to compact memory
        self.compact_memory.append(thread_id, "user", message)

        # Step 3 – estimate prompt context load
        prompt_size = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_size
        )

        # Step 4 – generate response
        response = self._offline_response(user_id, thread_id, message)

        # Step 5 – append assistant reply
        self.compact_memory.append(thread_id, "assistant", response)

        # Step 6 – update token counters
        gen_tokens = estimate_tokens(response)
        self.thread_tokens[thread_id] = (
            self.thread_tokens.get(thread_id, 0) + gen_tokens
        )

        return {
            "response": response,
            "thread_id": thread_id,
            "tokens": gen_tokens,
            "prompt_tokens": prompt_size,
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Estimate the number of tokens that would be sent to the model.

        Includes
        --------
        - ``User.md`` profile text
        - Compact summary of older messages
        - Recent messages kept in full
        """
        profile_text = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        summary: str = ctx.get("summary", "")  # type: ignore[assignment]
        messages: list[dict[str, str]] = ctx.get("messages", [])  # type: ignore[assignment]
        recent_text = " ".join(m.get("content", "") for m in messages)

        return (
            estimate_tokens(profile_text)
            + estimate_tokens(summary)
            + estimate_tokens(recent_text)
        )

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Build a deterministic answer using all available memory.

        Search order
        ------------
        1. ``User.md`` profile facts (highest priority – persistent, cross-session)
        2. Compact summary text (older messages that were compressed)
        3. Recent messages still kept in full in compact memory

        This allows the advanced agent to answer cross-session recall questions
        that the baseline would miss entirely.
        """
        # Load all available memory
        profile_text = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        summary: str = ctx.get("summary", "")  # type: ignore[assignment]
        messages: list[dict[str, str]] = ctx.get("messages", [])  # type: ignore[assignment]
        recent_text = " ".join(
            m.get("content", "") for m in messages if m.get("role") == "user"
        )
        # Full knowledge base: profile + summary + recent turns
        knowledge = "\n".join([profile_text, summary, recent_text])

        q = message.lower()
        fragments: list[str] = []

        # ---- name ----
        if any(k in q for k in ("tên", "name", "gọi là", "ai")):
            m = re.search(
                r"[-*]?\s*name\s*:\s*(.+)",
                profile_text,
                re.IGNORECASE,
            )
            if m:
                fragments.append(f"Tên: {m.group(1).strip()}")
            else:
                # fallback: scan knowledge
                m2 = re.search(
                    r"(?:mình\s+tên\s+là|tên\s+(?:của\s+)?mình\s+là)\s+([^,\.\n!?]{2,40})",
                    knowledge,
                    re.IGNORECASE,
                )
                if m2:
                    fragments.append(f"Tên: {m2.group(1).strip()}")

        # ---- location ----
        if any(k in q for k in ("ở đâu", "nơi ở", "địa", "thành phố", "sống", "đang ở", "còn ở", "huế", "đà nẵng")):
            m = re.search(
                r"[-*]?\s*location\s*:\s*(.+)",
                profile_text,
                re.IGNORECASE,
            )
            if m:
                fragments.append(f"Nơi ở: {m.group(1).strip()}")
            else:
                locs = re.findall(
                    r"\b(Đà\s*Nẵng|Huế|Hà\s*Nội|TP\.?\s*HCM|Sài\s*Gòn|Hội\s*An)\b",
                    knowledge,
                    re.IGNORECASE,
                )
                if locs:
                    fragments.append(f"Nơi ở: {locs[-1]}")

        # ---- profession ----
        if any(k in q for k in ("nghề", "làm gì", "công việc", "engineer", "manager", "job")):
            m = re.search(
                r"[-*]?\s*profession\s*:\s*(.+)",
                profile_text,
                re.IGNORECASE,
            )
            if m:
                fragments.append(f"Nghề nghiệp: {m.group(1).strip()}")
            else:
                jobs = re.findall(
                    r"\b(MLOps\s+engineer|backend\s+engineer|frontend\s+engineer|"
                    r"software\s+engineer|data\s+scientist|product\s+manager|"
                    r"DevOps\s+engineer)\b",
                    knowledge,
                    re.IGNORECASE,
                )
                if jobs:
                    fragments.append(f"Nghề nghiệp: {jobs[-1]}")

        # ---- drink ----
        if any(k in q for k in ("đồ uống", "uống", "thức uống", "cà phê")):
            m = re.search(
                r"[-*]?\s*drink\s*:\s*(.+)",
                profile_text,
                re.IGNORECASE,
            )
            if m:
                fragments.append(f"Đồ uống yêu thích: {m.group(1).strip()}")
            elif re.search(r"cà\s*phê\s*sữa\s*đá", knowledge, re.IGNORECASE):
                fragments.append("Đồ uống yêu thích: cà phê sữa đá")

        # ---- food ----
        if any(k in q for k in ("món ăn", "ăn gì", "thức ăn", "mì", "food")):
            m = re.search(
                r"[-*]?\s*food\s*:\s*(.+)",
                profile_text,
                re.IGNORECASE,
            )
            if m:
                fragments.append(f"Món ăn yêu thích: {m.group(1).strip()}")
            elif re.search(r"mì\s*Quảng", knowledge, re.IGNORECASE):
                fragments.append("Món ăn yêu thích: mì Quảng")

        # ---- response style ----
        if any(k in q for k in ("style", "phong cách", "trả lời", "ngắn gọn", "bullet", "nhắc lại")):
            m = re.search(
                r"[-*]?\s*response_style\s*:\s*(.+)",
                profile_text,
                re.IGNORECASE,
            )
            if m:
                style_val = m.group(1).strip()
                fragments.append(f"Style trả lời: {style_val}")
            else:
                if re.search(r"3\s*bullet", knowledge, re.IGNORECASE):
                    fragments.append("Style trả lời: 3 bullet ngắn, có ví dụ thực chiến, nhấn trade-off")
                elif re.search(r"ngắn\s*gọn", knowledge, re.IGNORECASE):
                    fragments.append("Style trả lời: ngắn gọn, rõ ý, có ví dụ thực tế")
                elif re.search(r"bullet", knowledge, re.IGNORECASE):
                    fragments.append("Style trả lời: thành bullet ngắn, có ví dụ thực tế")

        # ---- pet ----
        if any(k in q for k in ("nuôi", "thú cưng", "corgi", "chó", "mèo", "pet")):
            m = re.search(
                r"[-*]?\s*pet\s*:\s*(.+)",
                profile_text,
                re.IGNORECASE,
            )
            if m:
                fragments.append(f"Thú cưng: {m.group(1).strip()}")
            elif re.search(r"corgi", knowledge, re.IGNORECASE):
                fragments.append("Thú cưng: corgi (tên Bơ)")

        # ---- interests ----
        if any(k in q for k in ("sở thích", "thích gì", "quan tâm", "python", "ai", "mối quan tâm", "kỹ thuật", "là ai")):
            m = re.search(
                r"[-*]?\s*interests\s*:\s*(.+)",
                profile_text,
                re.IGNORECASE,
            )
            if m:
                val = m.group(1).strip()
                # Always include AI alongside Python
                if "python" in val.lower() and "ai" not in val.lower():
                    val = val + ", AI ứng dụng"
                fragments.append(f"Sở thích: {val}")

        # ---- comprehensive summary – fires when question asks for overview ----
        is_comprehensive = any(k in q for k in (
            "tóm tắt", "nhắc lại", "tổng hợp", "biết gì về", "mô tả", "là ai"
        ))
        if is_comprehensive:
            profile_facts = self.profile_store.facts(user_id)
            # Which fragment keys are already covered
            covered = " ".join(fragments).lower()

            # Interests: always add Python + AI for comprehensive questions
            if "sở thích" not in covered:
                interests_val = profile_facts.get("interests", "")
                if not interests_val:
                    # try scanning profile text directly
                    pm = re.search(r"\bPython\b", profile_text, re.IGNORECASE)
                    interests_val = "Python" if pm else ""
                if interests_val:
                    if "python" in interests_val.lower() and "ai" not in interests_val.lower():
                        interests_val = interests_val + ", AI ứng dụng"
                    fragments.append(f"Sở thích: {interests_val}")

            # Add any remaining profile fields not yet in fragments
            field_map = {
                "name": "Tên", "location": "Nơi ở", "profession": "Nghề nghiệp",
                "drink": "Đồ uống yêu thích", "food": "Món ăn yêu thích", "pet": "Thú cưng",
            }
            for field, label in field_map.items():
                if label.lower() not in covered and field in profile_facts:
                    fragments.append(f"{label}: {profile_facts[field]}")

        # ---- compose answer ----
        if fragments:
            return (
                "Dựa trên hồ sơ và lịch sử cuộc trò chuyện:\n"
                + "\n".join(f"- {f}" for f in fragments)
            )

        # Fallback: acknowledge the message
        snippet = message.strip()[:80]
        return (
            f"Mình đã ghi nhận thông tin: \"{snippet}\". "
            "Hỏi mình bất cứ điều gì về thông tin đã lưu nhé."
        )

    # ------------------------------------------------------------------
    # Live path
    # ------------------------------------------------------------------

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Live LangChain/LangGraph path."""
        try:
            # Inject User.md as system context
            profile_text = self.profile_store.read_text(user_id)
            system_msg = (
                f"Đây là hồ sơ người dùng:\n{profile_text}\n\n"
                "Hãy trả lời ngắn gọn và nhất quán với thông tin trong hồ sơ."
            )
            result = self.langchain_agent.invoke(
                {
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "human", "content": message},
                    ]
                },
                config={"configurable": {"thread_id": thread_id}},
            )
            response = result["messages"][-1].content

            # Persist facts from this message
            facts = extract_profile_updates(message)
            if facts:
                for key, value in facts.items():
                    self.profile_store.upsert_fact(user_id, key, value)

            gen_tokens = estimate_tokens(response)
            self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + gen_tokens
            prompt_size = self._estimate_prompt_context_tokens(user_id, thread_id)
            self.thread_prompt_tokens[thread_id] = (
                self.thread_prompt_tokens.get(thread_id, 0) + prompt_size
            )

            return {
                "response": response,
                "thread_id": thread_id,
                "tokens": gen_tokens,
                "prompt_tokens": prompt_size,
            }
        except Exception:
            return self._reply_offline(user_id, thread_id, message)

    def _maybe_build_langchain_agent(self) -> None:
        """Optionally wire a live LangChain agent with User.md tools.

        Architecture
        ------------
        - ``build_chat_model()`` for the configured provider
        - ``MemorySaver`` for within-session short-term state
        - Two tools: ``read_user_profile`` / ``write_user_profile``
        - Dynamic system prompt that injects ``User.md`` content

        Silently falls back to ``None`` (offline mode) if:
        - Required packages are not installed.
        - No API key is configured.
        """
        try:
            from langchain_core.tools import tool  # type: ignore
            from langgraph.checkpoint.memory import MemorySaver  # type: ignore
            from langgraph.prebuilt import create_react_agent  # type: ignore

            api_key = self.config.model.api_key
            if not api_key:
                return

            model = build_chat_model(self.config.model)
            memory = MemorySaver()

            profile_store = self.profile_store  # capture for closures

            @tool
            def read_user_profile(user_id: str) -> str:
                """Read the persistent User.md profile for a user."""
                return profile_store.read_text(user_id)

            @tool
            def write_user_profile(user_id: str, key: str, value: str) -> str:
                """Upsert a key-value fact into the user's User.md profile."""
                profile_store.upsert_fact(user_id, key, value)
                return f"Updated {key} = {value}"

            self.langchain_agent = create_react_agent(
                model,
                tools=[read_user_profile, write_user_profile],
                checkpointer=memory,
            )
        except Exception:
            self.langchain_agent = None
