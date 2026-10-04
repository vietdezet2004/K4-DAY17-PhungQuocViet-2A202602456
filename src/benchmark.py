from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float          # 0.0 – 1.0
    response_quality: float      # 0.0 – 1.0
    memory_growth_bytes: int
    compactions: int


# ---------------------------------------------------------------------------
# Task 6.1 – load_conversations
# ---------------------------------------------------------------------------

def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read a JSON conversation file and return the list of conversation dicts."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data)}")
    return data


# ---------------------------------------------------------------------------
# Task 6.2 – recall_points
# ---------------------------------------------------------------------------

def recall_points(answer: str, expected: list[str]) -> float:
    """Score how many expected strings appear in the answer.

    Returns
    -------
    1.0  – all expected strings found
    0.5  – more than half found (but not all)
    0.0  – half or fewer found
    """
    if not expected:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for e in expected if e.lower() in answer_lower)
    ratio = hits / len(expected)
    if ratio == 1.0:
        return 1.0
    if ratio > 0.5:
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# Task 6.3 – heuristic_quality
# ---------------------------------------------------------------------------

def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight quality score for offline mode.

    Combines three signals:
    1. Recall ratio  – fraction of expected facts present
    2. Length bonus  – answer between 20 and 400 chars scores full; very short
                       or extremely long answers are penalised
    3. Structure     – presence of bullet points or numbered lists

    Returns a float in [0.0, 1.0].
    """
    if not answer:
        return 0.0

    # Signal 1: recall ratio (0 – 0.6 weight)
    answer_lower = answer.lower()
    hits = sum(1 for e in expected if e.lower() in answer_lower) if expected else 1
    recall_ratio = hits / len(expected) if expected else 1.0
    recall_component = recall_ratio * 0.6

    # Signal 2: length bonus (0 – 0.3 weight)
    length = len(answer.strip())
    if 20 <= length <= 400:
        length_component = 0.3
    elif length < 10:
        length_component = 0.0
    else:
        # Gradually penalise very long answers
        length_component = max(0.0, 0.3 - (length - 400) / 2000)

    # Signal 3: structure bonus (0 – 0.1 weight)
    has_structure = any(
        marker in answer for marker in ("- ", "* ", "• ", "\n1.", "\n2.")
    )
    structure_component = 0.1 if has_structure else 0.0

    return round(recall_component + length_component + structure_component, 3)


# ---------------------------------------------------------------------------
# Task 6.4 – run_agent_benchmark
# ---------------------------------------------------------------------------

def run_agent_benchmark(
    agent_name: str,
    agent,
    conversations: list[dict[str, Any]],
    config: LabConfig,
) -> BenchmarkRow:
    """Evaluate one agent over a list of conversations.

    Steps
    -----
    1. Feed every turn of every conversation to the agent.
    2. Accumulate agent token and prompt token usage.
    3. Ask each recall question in a dedicated **fresh** thread.
    4. Average recall_points and heuristic_quality over all recall questions.
    5. Record memory file growth (advanced agent only) and compaction count.
    """
    total_agent_tokens = 0
    total_prompt_tokens = 0
    total_compactions = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    # Track which thread_ids were used for feeding turns
    feed_thread_ids: list[str] = []

    for conv in conversations:
        uid: str = conv.get("user_id", "unknown")
        tid: str = conv.get("id", f"conv-{len(feed_thread_ids)}")
        feed_thread_ids.append(tid)

        # Step 1+2 – feed all turns
        for turn in conv.get("turns", []):
            agent.reply(uid, tid, turn)

        total_compactions += agent.compaction_count(tid)

    # Accumulate token usage after all conversations
    for tid in feed_thread_ids:
        total_agent_tokens += agent.token_usage(tid)
        total_prompt_tokens += agent.prompt_token_usage(tid)

    # Step 3+4 – recall questions in a single fresh thread per benchmark run
    recall_thread = f"recall-{agent_name}"
    for conv in conversations:
        uid = conv.get("user_id", "unknown")
        for rq in conv.get("recall_questions", []):
            question: str = rq.get("question", "")
            expected: list[str] = rq.get("expected_contains", [])

            result = agent.reply(uid, recall_thread, question)
            answer: str = result.get("response", "")

            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    # Step 5 – memory file size (only advanced agent has User.md)
    memory_bytes = 0
    if conversations:
        uid = conversations[0].get("user_id", "unknown")
        if hasattr(agent, "memory_file_size"):
            memory_bytes = agent.memory_file_size(uid)

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=round(avg_recall, 3),
        response_quality=round(avg_quality, 3),
        memory_growth_bytes=memory_bytes,
        compactions=total_compactions,
    )


# ---------------------------------------------------------------------------
# Task 6.5 – format_rows
# ---------------------------------------------------------------------------

def format_rows(rows: list[BenchmarkRow]) -> str:
    """Return a human-readable table string (tabulate if available, else plain)."""
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    data = [
        [
            r.agent_name,
            r.agent_tokens_only,
            r.prompt_tokens_processed,
            f"{r.recall_score:.3f}",
            f"{r.response_quality:.3f}",
            r.memory_growth_bytes,
            r.compactions,
        ]
        for r in rows
    ]

    try:
        from tabulate import tabulate  # type: ignore
        return tabulate(data, headers=headers, tablefmt="github")
    except ImportError:
        # Fallback: plain text aligned table
        col_widths = [max(len(str(headers[i])), max(len(str(row[i])) for row in data))
                      for i in range(len(headers))]
        sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
        def fmt_row(row: list) -> str:
            return "| " + " | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(row))) + " |"
        lines = [sep, fmt_row(headers), sep]
        for row in data:
            lines.append(fmt_row(row))
        lines.append(sep)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Task 6.6 – main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run Standard and Long-Context Stress benchmarks, print comparison tables."""
    root = Path(__file__).resolve().parent.parent
    config = load_config(root)

    # ------------------------------------------------------------------ paths
    standard_path = config.data_dir / "conversations.json"
    stress_path   = config.data_dir / "advanced_long_context.json"

    standard_convs = load_conversations(standard_path)
    stress_convs   = load_conversations(stress_path)

    # ------------------------------------------------------------------ helpers
    def make_agents(tmp_dir: Path):
        """Create a fresh pair of agents backed by an isolated state directory."""
        cfg = load_config(root)
        cfg.state_dir.mkdir(parents=True, exist_ok=True)
        (cfg.state_dir / "profiles").mkdir(parents=True, exist_ok=True)
        baseline = BaselineAgent(config=cfg, force_offline=True)
        advanced = AdvancedAgent(config=cfg, force_offline=True)
        return baseline, advanced

    # ================================================================
    # 1. Standard Benchmark
    # ================================================================
    print("=" * 70)
    print("Standard Benchmark  (data/conversations.json)")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp:
        baseline, advanced = make_agents(Path(tmp))

        rows_std = [
            run_agent_benchmark("Baseline", baseline, standard_convs, config),
            run_agent_benchmark("Advanced", advanced, standard_convs, config),
        ]

    print(format_rows(rows_std))
    print()

    # Quick interpretation
    adv_std, base_std = rows_std[1], rows_std[0]
    recall_lift = adv_std.recall_score - base_std.recall_score
    print(f"  [+] Recall lift (Advanced vs Baseline): +{recall_lift:.3f}")
    print()

    # ================================================================
    # 2. Long-Context Stress Benchmark
    # ================================================================
    print("=" * 70)
    print("Long-Context Stress Benchmark  (data/advanced_long_context.json)")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp:
        baseline, advanced = make_agents(Path(tmp))

        rows_stress = [
            run_agent_benchmark("Baseline", baseline, stress_convs, config),
            run_agent_benchmark("Advanced", advanced, stress_convs, config),
        ]

    print(format_rows(rows_stress))
    print()

    # Prompt token comparison
    adv_s, base_s = rows_stress[1], rows_stress[0]
    if base_s.prompt_tokens_processed > 0:
        saving_pct = (1 - adv_s.prompt_tokens_processed / base_s.prompt_tokens_processed) * 100
        print(f"  [-] Prompt token saving (Advanced vs Baseline): {saving_pct:.1f}%")
    print(f"  [+] Compactions triggered by Advanced: {adv_s.compactions}")
    print()


if __name__ == "__main__":
    main()
