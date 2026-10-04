from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Task 3.1 – Token estimator
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Heuristic token estimator (no external dependency required).

    Rule: ~4 characters per token, which is the widely-used approximation for
    English/Latin scripts.  Vietnamese characters are multi-byte but still
    roughly one token each when the text is already in Unicode, so the same
    divisor gives a conservative over-estimate — safe for budget checks.

    Returns 0 for empty / whitespace-only input.
    """
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // 4)


# ---------------------------------------------------------------------------
# Task 3.2 – UserProfileStore
# ---------------------------------------------------------------------------

_DEFAULT_PROFILE_TEMPLATE = "# User Profile\n\n"


@dataclass
class UserProfileStore:
    """Persistent ``User.md`` storage, one file per user.

    Directory layout::

        <root_dir>/<user_id>/User.md
    """

    root_dir: Path

    # ------------------------------------------------------------------
    def path_for(self, user_id: str) -> Path:
        """Return the ``User.md`` path for *user_id*.

        The user id is sanitised: only alphanumeric characters, hyphens,
        underscores and dots are kept; everything else becomes ``_``.
        This avoids path-traversal and illegal filename characters.
        """
        safe_id = re.sub(r"[^\w.\-]", "_", user_id)
        return self.root_dir / safe_id / "User.md"

    # ------------------------------------------------------------------
    def read_text(self, user_id: str) -> str:
        """Return file content, or the default empty template if not found."""
        p = self.path_for(user_id)
        if p.exists():
            return p.read_text(encoding="utf-8")
        return _DEFAULT_PROFILE_TEMPLATE

    # ------------------------------------------------------------------
    def write_text(self, user_id: str, content: str) -> Path:
        """Write *content* to ``User.md``, creating parent directories as needed."""
        p = self.path_for(user_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return p

    # ------------------------------------------------------------------
    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        """Replace the first occurrence of *search_text* in ``User.md``.

        Returns ``True`` if the file was changed, ``False`` otherwise
        (e.g. the search string was not found).
        """
        current = self.read_text(user_id)
        if search_text not in current:
            return False
        updated = current.replace(search_text, replacement, 1)
        self.write_text(user_id, updated)
        return True

    # ------------------------------------------------------------------
    def file_size(self, user_id: str) -> int:
        """Return file size in bytes, or 0 if the file does not exist."""
        p = self.path_for(user_id)
        if p.exists():
            return p.stat().st_size
        return 0

    # ------------------------------------------------------------------
    # Optional helpers – useful for structured fact access
    # ------------------------------------------------------------------

    def facts(self, user_id: str) -> dict[str, str]:
        """Parse ``key: value`` lines from ``User.md`` into a dict."""
        result: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            if ":" in line:
                key, _, val = line.partition(":")
                key = key.strip().lstrip("-").strip()
                val = val.strip()
                if key and val:
                    result[key] = val
        return result

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        """Insert or update a ``key: value`` line in ``User.md``.

        If a line starting with ``key:`` already exists it is replaced in-place;
        otherwise the fact is appended.
        """
        current = self.read_text(user_id)
        # Try in-place replacement first
        pattern = re.compile(
            r"^([ \t]*[-*]?[ \t]*)" + re.escape(key) + r"[ \t]*:.*$",
            re.MULTILINE | re.IGNORECASE,
        )
        new_line = f"- {key}: {value}"
        if pattern.search(current):
            updated = pattern.sub(new_line, current, count=1)
        else:
            # Append after last non-empty line
            updated = current.rstrip("\n") + "\n" + new_line + "\n"
        self.write_text(user_id, updated)


# ---------------------------------------------------------------------------
# Task 3.3 – extract_profile_updates
# ---------------------------------------------------------------------------

# Patterns map fact key -> list of regex patterns.
# Each pattern must have a named capture group called ``value``.
# Patterns are tried in order; first match wins for that key.
#
# Guard phrases: if the message looks like a pure question or negation we skip it.
_QUESTION_GUARDS = re.compile(
    r"(bạn|mình)\s+(có\s+biết|có\s+nhớ|có\s+thể|thử\s+nhắc|nhắc\s+lại)",
    re.IGNORECASE,
)

_NOISE_JOB = re.compile(
    r"hay\s+là\s+chuyển\s+sang|chỉ\s+là\s+câu\s+đùa|đùa\s+với|chỉ\s+đùa",
    re.IGNORECASE,
)

_NOISE_LOCATION = re.compile(
    r"(đi\s+họp|bay\s+ra\s+họp|ghé\s+qua|nơi\s+mình\s+vừa\s+bay|chỉ\s+là\s+nơi)",
    re.IGNORECASE,
)

# Correction markers: these signals mean a new fact overwrites the old one
_CORRECTION_MARKERS = re.compile(
    r"(đính\s+chính|cập\s+nhật|không\s+còn.*nữa|giờ\s+(mình|là)|thực\s+ra|chuyển\s+sang|đổi\s+sang|bây\s+giờ\s+mình)",
    re.IGNORECASE,
)

_PATTERNS: dict[str, list[re.Pattern]] = {
    "name": [
        # "mình tên là X" – stop at comma/period; allow multi-word names (e.g. "DũngCT Stress")
        re.compile(r"(?:mình\s+tên\s+là|tên\s+(?:của\s+)?mình\s+là)\s+(?P<value>[^,\.\n!?]{2,40}?)(?=\s*[,\.!\?\n]|$)", re.IGNORECASE),
    ],
    "location": [
        # Specific Vietnamese city names – short match, no false positives
        re.compile(r"\bở\s+(?P<value>Đà\s*Nẵng|Huế|Hà\s*Nội|TP\.?\s*HCM|Sài\s*Gòn|Hội\s*An)\b", re.IGNORECASE),
        re.compile(r"\bhiện\s+(?:tại\s+)?(?:mình\s+)?(?:đang\s+)?ở\s+(?P<value>Đà\s*Nẵng|Huế|Hà\s*Nội|TP\.?\s*HCM|Sài\s*Gòn)", re.IGNORECASE),
        re.compile(r"\blàm\s+việc\s+ở\s+(?P<value>Đà\s*Nẵng|Huế|Hà\s*Nội|TP\.?\s*HCM|Sài\s*Gòn)", re.IGNORECASE),
        re.compile(r"\bnơi\s+ở\s+(?:hiện\s+tại\s+)?(?:là\s+)?(?P<value>Đà\s*Nẵng|Huế|Hà\s*Nội|TP\.?\s*HCM|Sài\s*Gòn)", re.IGNORECASE),
    ],
    "profession": [
        # "Nghề nghiệp hiện tại vẫn là X" – highest priority (override joke corrections)
        re.compile(r"nghề\s+nghiệp\s+hiện\s+tại\s+(?:vẫn\s+)?là\s+(?P<value>MLOps\s+engineer|backend\s+engineer|frontend\s+engineer|fullstack\s+engineer|software\s+engineer|product\s+manager|data\s+scientist|data\s+engineer|AI\s+researcher|DevOps\s+engineer)\b", re.IGNORECASE),
        # Correction form: "chuyển sang / sang / giờ là <job>"  – pick the NEW job after correction marker
        re.compile(r"(?:chuyển\s+sang|sang\s+làm|giờ\s+(?:là|làm)|hiện\s+(?:tại\s+)?(?:là|làm))\s+(?P<value>MLOps\s+engineer|backend\s+engineer|frontend\s+engineer|fullstack\s+engineer|software\s+engineer|product\s+manager|data\s+scientist|data\s+engineer|AI\s+researcher|DevOps\s+engineer)\b", re.IGNORECASE),
        # Plain "làm <job>" form
        re.compile(r"\blàm\s+(?P<value>MLOps\s+engineer|backend\s+engineer|frontend\s+engineer|fullstack\s+engineer|software\s+engineer|data\s+scientist|data\s+engineer|AI\s+researcher|DevOps\s+engineer)\b", re.IGNORECASE),
        re.compile(r"\bnghề\s+(?:nghiệp\s+)?(?:hiện\s+tại\s+)?(?:là\s+)?(?P<value>MLOps\s+engineer|backend\s+engineer|frontend\s+engineer|product\s+manager|data\s+scientist|AI\s+researcher)\b", re.IGNORECASE),
        re.compile(r"\bcông\s+việc\s+(?:hiện\s+tại\s+)?(?:là\s+)?(?P<value>MLOps\s+engineer|backend\s+engineer|frontend\s+engineer|product\s+manager|data\s+scientist|AI\s+researcher)\b", re.IGNORECASE),
    ],
    "drink": [
        re.compile(r"đồ\s+uống\s+yêu\s+thích\s+(?:là|của\s+mình\s+là)\s+(?P<value>cà\s*phê\s*sữa\s*đá|cà\s*phê[^,\.\n!?]{0,20}|trà[^,\.\n!?]{0,20})", re.IGNORECASE),
        # "vẫn uống" or "thích uống" – capture fixed phrase only, trim "như cũ" etc.
        re.compile(r"(?:vẫn\s+uống|thích\s+uống)\s+(?P<value>cà\s*phê\s*sữa\s*đá|cà\s*phê[^,\.\n!?]{0,20}|trà[^,\.\n!?]{0,20})", re.IGNORECASE),
        # bare occurrence of "cà phê sữa đá"
        re.compile(r"\b(?P<value>cà\s*phê\s*sữa\s*đá)\b", re.IGNORECASE),
    ],
    "food": [
        re.compile(r"(?:món\s+ăn\s+yêu\s+thích\s+là|thích\s+ăn|món\s+ruột\s+là)\s+(?P<value>[^\.,;!\?\n]+)", re.IGNORECASE),
        re.compile(r"\b(?P<value>mì\s*Quảng|phở|bún\s+bò|cơm\s+tấm)\b", re.IGNORECASE),
    ],
    "pet": [
        re.compile(r"nuôi\s+(?:một?\s+)?(?:bé\s+)?(?P<value>\w+(?:\s+\w+)?)\s+tên", re.IGNORECASE),
        re.compile(r"con\s+(?P<value>corgi|mèo|chó|hamster|thỏ)\b", re.IGNORECASE),
    ],
    "response_style": [
        # "muốn bạn trả lời ngắn gọn" / "thành 3 bullet"
        re.compile(r"(?:mình\s+)?muốn\s+(?:bạn\s+)?trả\s+lời\s+(?P<value>ngắn\s*gọn[^,\.\n!?]{0,60}|thành\s+\d+\s+bullet[^,\.\n!?]{0,60})", re.IGNORECASE),
        re.compile(r"hãy\s+trả\s+lời\s+(?P<value>ngắn\s*gọn[^,\.\n!?]{0,60}|thành\s+\d+\s+bullet[^,\.\n!?]{0,60}|thành\s+bullet[^,\.\n!?]{0,60})", re.IGNORECASE),
        # "style trả lời X" – only when followed by an actual value (not end of sentence)
        re.compile(r"style\s+trả\s+lời\s+(?:mình\s+thích\s+)?(?:là\s+)?(?P<value>ngắn\s*gọn[^,\.\n!?]{0,60}|\d+\s+bullet[^,\.\n!?]{0,60}|bullet[^,\.\n!?]{0,60})", re.IGNORECASE),
    ],
    "interests": [
        re.compile(r"mình\s+thích\s+(?P<value>Python[^,\.\n!?]{0,60}|AI[^,\.\n!?]{0,60})", re.IGNORECASE),
        re.compile(r"mình\s+(?:đang\s+)?quan\s+tâm[^,\.\n!?]{0,10}(?:đến|tới|về)\s+(?P<value>Python[^,\.\n!?]{0,60}|AI[^,\.\n!?]{0,60})", re.IGNORECASE),
    ],
}

# Maximum value length to avoid capturing run-on sentences
_MAX_VALUE_LEN = 80




# ---------------------------------------------------------------------------
# Bonus Task 8.1 – Confidence threshold
# ---------------------------------------------------------------------------

# Low-confidence signals: the fact is hedged / uncertain
_HEDGE_PATTERNS = re.compile(
    r"\b(có\s+lẽ|có\s+thể|hình\s+như|không\s+chắc|chắc\s+là|nghe\s+nói|"
    r"maybe|perhaps|possibly|probably|might|could\s+be|not\s+sure)\b",
    re.IGNORECASE,
)

# Question-only turn (whole message is interrogative – no new fact declared)
_PURE_QUESTION = re.compile(
    r"^\s*[^.!]*\?\s*$",
    re.DOTALL,
)


def confidence_score(message: str, key: str, value: str) -> float:
    """Return a confidence score in [0.0, 1.0] for a (key, value) fact.

    Heuristics
    ----------
    - Starts at 1.0 (full confidence for direct, declarative statements).
    - Drops to 0.4 if the message contains hedge words ("có lẽ", "maybe", …).
    - Drops to 0.2 if the whole message appears to be a pure question.
    - Correction markers ("đính chính", "giờ là", …) bump score back up to 0.95
      because explicit corrections are high-signal.

    The threshold used in ``extract_profile_updates`` is 0.5:
    facts below this score are discarded.
    """
    score = 1.0

    # Hedge language lowers confidence
    if _HEDGE_PATTERNS.search(message):
        score *= 0.4

    # Pure question has no declarative content
    if _PURE_QUESTION.match(message):
        score *= 0.2

    # Explicit correction → very high confidence regardless
    if _CORRECTION_MARKERS.search(message):
        score = max(score, 0.95)

    return round(score, 2)


# Minimum confidence required to persist a fact into User.md
CONFIDENCE_THRESHOLD = 0.5


def extract_profile_updates(message: str) -> dict[str, str]:
    """Extract stable profile facts from a user message.

    Rules
    -----
    - Skip turns that are pure questions / requests to recall (no new facts).
    - Skip noise signals (joke professions, temporary locations).
    - Correction markers *do* produce facts (the corrected value is the new fact).
    - Values longer than ``_MAX_VALUE_LEN`` are truncated.
    - **Bonus 8.1**: facts with ``confidence_score < CONFIDENCE_THRESHOLD`` are
      discarded so hedged or uncertain statements are not written to User.md.
    - Returns only keys where a value was confidently found.
    """
    facts: dict[str, str] = {}

    for key, patterns in _PATTERNS.items():
        for pattern in patterns:
            m = pattern.search(message)
            if not m:
                continue
            value = m.group("value").strip().rstrip(".,;!?")

            # Per-key noise guard: profession
            if key == "profession" and _NOISE_JOB.search(message):
                _authoritative = re.compile(
                    r"nghề\s+nghiệp\s+hiện\s+tại|công\s+việc\s+hiện\s+tại|nghề\s+hiện\s+tại",
                    re.IGNORECASE,
                )
                is_first_pattern = pattern == _PATTERNS["profession"][0]
                if not (is_first_pattern or _authoritative.search(message)):
                    noise_m = _NOISE_JOB.search(message)
                    if noise_m and abs(m.start() - noise_m.start()) < 120:
                        continue

            # Per-key noise guard: location
            if key == "location" and _NOISE_LOCATION.search(message):
                noise_m = _NOISE_LOCATION.search(message)
                if noise_m and abs(m.start() - noise_m.start()) < 120:
                    continue

            # Length guard
            if len(value) > _MAX_VALUE_LEN:
                value = value[:_MAX_VALUE_LEN]

            # Trim trailing noise words ("như cũ", "vẫn vậy", …)
            value = re.sub(
                r"\s+(như\s+cũ|vẫn\s+vậy|thôi|nhé|nha|đó|rồi)\s*$",
                "",
                value,
                flags=re.IGNORECASE,
            ).strip()

            # Skip implausibly short values
            if len(value) < 3:
                continue

            # Bonus 8.1 – confidence threshold filter
            score = confidence_score(message, key, value)
            if score < CONFIDENCE_THRESHOLD:
                continue

            if value:
                facts[key] = value
                break  # first match wins per key

    return facts

def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact text summary of the *oldest* messages in a thread.

    Parameters
    ----------
    messages  : list of ``{"role": ..., "content": ...}`` dicts
    max_items : how many messages to include in the summary

    The summary keeps each entry on one line formatted as ``role: content``
    (truncated to 120 chars per entry) so it stays compact but still allows
    the advanced agent to recall key facts when the summary is injected into
    the prompt.
    """
    if not messages:
        return ""

    selected = messages[:max_items]
    lines: list[str] = []
    for msg in selected:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        # Truncate very long individual messages
        if len(content) > 120:
            content = content[:117] + "..."
        lines.append(f"{role}: {content}")

    return "[Tóm tắt lịch sử]\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Task 3.5 – CompactMemoryManager
# ---------------------------------------------------------------------------

@dataclass
class CompactMemoryManager:
    """Short-term memory with automatic compaction for long threads.

    When the total token count of all messages in a thread exceeds
    ``threshold_tokens``, the oldest ``(len - keep_messages)`` messages are
    compressed into a summary string.  The ``keep_messages`` most-recent
    messages are kept in full.

    Per-thread state is stored in ``self.state[thread_id]``:

    .. code-block:: python

        {
            "messages":    [{"role": ..., "content": ...}, ...],
            "summary":     "...",      # accumulated compact summary text
            "compactions": int,        # number of times compaction ran
        }
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def _init_thread(self, thread_id: str) -> None:
        """Initialise per-thread state if it does not exist yet."""
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }

    # ------------------------------------------------------------------
    def _total_tokens(self, thread_id: str) -> int:
        """Estimate total tokens for all messages in *thread_id*."""
        thread = self.state[thread_id]
        messages: list[dict[str, str]] = thread["messages"]  # type: ignore[assignment]
        summary: str = thread["summary"]  # type: ignore[assignment]
        total = estimate_tokens(summary)
        for msg in messages:
            total += estimate_tokens(msg.get("content", ""))
        return total

    # ------------------------------------------------------------------
    def _compact(self, thread_id: str) -> None:
        """Compress the oldest messages into a summary, keeping recent ones."""
        thread = self.state[thread_id]
        messages: list[dict[str, str]] = thread["messages"]  # type: ignore[assignment]

        if len(messages) <= self.keep_messages:
            return  # nothing old enough to compact

        n_to_compact = len(messages) - self.keep_messages
        old_messages = messages[:n_to_compact]
        recent_messages = messages[n_to_compact:]

        new_summary_piece = summarize_messages(old_messages, max_items=len(old_messages))

        existing_summary: str = thread["summary"]  # type: ignore[assignment]
        if existing_summary:
            thread["summary"] = existing_summary + "\n" + new_summary_piece
        else:
            thread["summary"] = new_summary_piece

        thread["messages"] = recent_messages
        thread["compactions"] = int(thread["compactions"]) + 1  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    def append(self, thread_id: str, role: str, content: str) -> None:
        """Append one message and trigger compaction if the budget is exceeded."""
        self._init_thread(thread_id)
        messages: list[dict[str, str]] = self.state[thread_id]["messages"]  # type: ignore[assignment]
        messages.append({"role": role, "content": content})

        if self._total_tokens(thread_id) > self.threshold_tokens:
            self._compact(thread_id)

    # ------------------------------------------------------------------
    def context(self, thread_id: str) -> dict[str, object]:
        """Return the full per-thread state dict.

        Keys: ``messages``, ``summary``, ``compactions``.
        An empty default is returned for unknown thread ids.
        """
        self._init_thread(thread_id)
        return dict(self.state[thread_id])

    # ------------------------------------------------------------------
    def compaction_count(self, thread_id: str) -> int:
        """Return the number of compactions that occurred for *thread_id*."""
        self._init_thread(thread_id)
        return int(self.state[thread_id]["compactions"])  # type: ignore[arg-type]
