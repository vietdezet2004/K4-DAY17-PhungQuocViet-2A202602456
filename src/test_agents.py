"""
test_agents.py – behavioural tests for the Day-17 Memory Systems lab.

Run with:
    pytest src/test_agents.py -v
"""
from __future__ import annotations

from pathlib import Path

import pytest

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from memory_store import UserProfileStore


# ---------------------------------------------------------------------------
# Task 7.1 – make_config
# ---------------------------------------------------------------------------

def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated LabConfig for tests.

    - state_dir  → tmp_path/state   (no bleed between tests)
    - compact threshold is very small so compaction fires quickly
    - no API key required (offline mode)
    """
    root = Path(__file__).resolve().parent.parent
    cfg = load_config(root)

    # Redirect runtime state into pytest's isolated temp directory
    cfg.state_dir = tmp_path / "state"
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    (cfg.state_dir / "profiles").mkdir(parents=True, exist_ok=True)

    # Small threshold → compaction fires after just a few messages
    cfg.compact_threshold_tokens = 40
    cfg.compact_keep_messages = 2

    return cfg


# ---------------------------------------------------------------------------
# Task 7.2 – test_user_markdown_read_write_edit
# ---------------------------------------------------------------------------

def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """User.md can be created, read back, and edited in-place."""
    cfg = make_config(tmp_path)
    store = UserProfileStore(cfg.state_dir / "profiles")

    uid = "testuser"

    # 1. Read a non-existent profile → default template (no error)
    default = store.read_text(uid)
    assert "User Profile" in default, "Default profile should contain header"
    assert store.file_size(uid) == 0, "File should not exist yet"

    # 2. Write a new profile
    path = store.write_text(uid, "# User Profile\n\n- name: Alice\n- location: Hanoi\n")
    assert path.exists(), "write_text should create the file"
    assert store.file_size(uid) > 0

    # 3. Read back
    content = store.read_text(uid)
    assert "Alice" in content
    assert "Hanoi" in content

    # 4. Edit existing value
    changed = store.edit_text(uid, "location: Hanoi", "location: HCM")
    assert changed is True
    assert "HCM" in store.read_text(uid)
    assert "Hanoi" not in store.read_text(uid)

    # 5. Edit non-existent value → returns False
    not_changed = store.edit_text(uid, "location: Hanoi", "location: X")
    assert not_changed is False

    # 6. Upsert: insert new fact
    store.upsert_fact(uid, "profession", "engineer")
    assert "profession" in store.read_text(uid)

    # 7. Upsert: update existing fact
    store.upsert_fact(uid, "location", "Da Nang")
    text = store.read_text(uid)
    assert "Da Nang" in text
    # Old value must be gone
    assert "HCM" not in text

    # 8. facts() returns a dict
    f = store.facts(uid)
    assert isinstance(f, dict)
    assert f.get("name") == "Alice"
    assert f.get("location") == "Da Nang"
    assert f.get("profession") == "engineer"


# ---------------------------------------------------------------------------
# Task 7.3 – test_compact_trigger
# ---------------------------------------------------------------------------

def test_compact_trigger(tmp_path: Path) -> None:
    """Long threads trigger compaction in the Advanced agent."""
    cfg = make_config(tmp_path)   # threshold=40 tokens
    agent = AdvancedAgent(config=cfg, force_offline=True)

    uid = "compactuser"
    tid = "long-thread"

    # Repeated long messages should push total tokens well past 40
    long_msg = "Đây là một đoạn tin nhắn khá dài để vượt ngưỡng compact nhỏ. " * 4

    for i in range(6):
        agent.reply(uid, tid, f"Tin số {i}: {long_msg}")

    count = agent.compaction_count(tid)
    assert count > 0, (
        f"Expected at least 1 compaction on a long thread "
        f"(threshold={cfg.compact_threshold_tokens}), got {count}"
    )

    # Context object must have the right keys
    ctx = agent.compact_memory.context(tid)
    assert "messages" in ctx
    assert "summary" in ctx
    assert "compactions" in ctx

    # Summary must be non-empty after compaction
    assert ctx["summary"], "Summary should be non-empty after compaction"

    # Remaining messages count ≤ keep_messages
    assert len(ctx["messages"]) <= cfg.compact_keep_messages + 1  # +1 for last appended


# ---------------------------------------------------------------------------
# Task 7.4 – test_cross_session_recall
# ---------------------------------------------------------------------------

def test_cross_session_recall(tmp_path: Path) -> None:
    """Advanced agent remembers facts across sessions; Baseline does not."""
    cfg = make_config(tmp_path)
    adv  = AdvancedAgent(config=cfg, force_offline=True)
    base = BaselineAgent(config=cfg, force_offline=True)

    uid = "dungct"

    # ---- Session 1: provide facts ----
    session1 = "session-1"
    adv.reply(uid,  session1, "Mình tên là DũngCT.")
    adv.reply(uid,  session1, "Mình ở Huế và đang làm MLOps engineer.")
    adv.reply(uid,  session1, "Đồ uống yêu thích là cà phê sữa đá.")
    base.reply(uid, session1, "Mình tên là DũngCT.")
    base.reply(uid, session1, "Mình ở Huế và đang làm MLOps engineer.")
    base.reply(uid, session1, "Đồ uống yêu thích là cà phê sữa đá.")

    # ---- Session 2: brand-new thread_id ----
    session2 = "session-2-brand-new"

    adv_r  = adv.reply(uid,  session2, "Mình tên gì và làm nghề gì?")
    base_r = base.reply(uid, session2, "Mình tên gì và làm nghề gì?")

    adv_answer  = adv_r["response"].lower()
    base_answer = base_r["response"].lower()

    # Normalize Vietnamese unicode so ũ == u for matching purposes
    import unicodedata
    def _norm(s: str) -> str:
        return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()

    adv_norm  = _norm(adv_answer)
    base_norm = _norm(base_answer)

    # Advanced must recall name and profession from User.md
    assert "dungct" in adv_norm, (
        f"Advanced should recall name from persistent memory.\nGot: {adv_r['response']}"
    )
    assert "mlops" in adv_norm or "engineer" in adv_norm, (
        f"Advanced should recall profession.\nGot: {adv_r['response']}"
    )

    # Baseline must NOT recall (no persistent memory)
    baseline_recalls_name = "dungct" in base_norm
    baseline_recalls_job  = "mlops" in base_norm or "engineer" in base_norm
    assert not (baseline_recalls_name and baseline_recalls_job), (
        f"Baseline should not have cross-session recall.\nGot: {base_r['response']}"
    )

    # ---- Also verify drink recall ----
    adv_r2 = adv.reply(uid, session2, "Đồ uống yêu thích của mình là gì?")
    adv_r2_norm = _norm(adv_r2["response"].lower())
    assert "ca phe" in adv_r2_norm or "coffee" in adv_r2_norm, (
        f"Advanced should recall drink.\nGot: {adv_r2['response']}"
    )


# ---------------------------------------------------------------------------
# Task 7.5 – test_compact_reduces_prompt_load_on_long_thread
# ---------------------------------------------------------------------------

def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Advanced agent's prompt token usage grows slower than Baseline on long threads."""
    cfg = make_config(tmp_path)

    # Use a slightly larger threshold so the difference is meaningful
    cfg.compact_threshold_tokens = 60
    cfg.compact_keep_messages = 2

    adv  = AdvancedAgent(config=cfg, force_offline=True)
    base = BaselineAgent(config=cfg, force_offline=True)

    uid = "loadtest"
    tid = "load-thread"

    # 12 turns with long messages to accumulate significant prompt load
    long_msg = "Đây là nội dung tin nhắn dài để tạo áp lực lên prompt context. " * 5

    for i in range(12):
        adv.reply(uid,  tid, f"Turn {i}: {long_msg}")
        base.reply(uid, tid, f"Turn {i}: {long_msg}")

    adv_prompt  = adv.prompt_token_usage(tid)
    base_prompt = base.prompt_token_usage(tid)

    # Advanced compaction must have fired
    assert adv.compaction_count(tid) > 0, "Expected compaction to trigger on long thread"

    # Advanced prompt load must be lower than baseline
    # (compact memory bounds context; baseline grows unboundedly)
    assert adv_prompt < base_prompt, (
        f"Advanced prompt tokens ({adv_prompt}) should be less than "
        f"Baseline prompt tokens ({base_prompt}) on a long thread"
    )

    # Sanity: token_usage (generated tokens) should be non-zero for both
    assert adv.token_usage(tid) > 0
    assert base.token_usage(tid) > 0
