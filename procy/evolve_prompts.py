"""Fixed-policy prompt templates matching OpenEvolve's algorithm.

Claude Code writes files directly, so prompts tell it to modify the target
file on disk.  The evaluator then runs against the file.  After Claude
responds, we read the file to store the new code in the population.
"""
from __future__ import annotations

# Matches openevolve/prompts/defaults/system_message.txt
SYSTEM_MESSAGE = (
    "You are an expert software developer tasked with iteratively improving "
    "a codebase. Your goal is to maximize the FITNESS SCORE while exploring "
    "diverse solutions across feature dimensions. The system maintains a "
    "collection of diverse programs - both high fitness AND diversity are valuable."
)

# ── User message template ────────────────────────────────────────────────
# Adapted from openevolve full_rewrite_user.txt + evolution_history.txt
# Key difference: tells Claude to modify the file on disk (not output code in chat)

FULL_REWRITE_USER = """\
Improve the code to maximize the evaluator score. \
Run the evaluator: {eval_command} — it outputs JSON metrics. \
Current best fitness: {fitness:.4f}. {focus}

Past results: sqlite3 {db_path} "SELECT commit_hash, fitness_score, changes_description FROM programs WHERE session_id='{session_id}' ORDER BY fitness_score DESC LIMIT 5"

Use git diff to study what top approaches changed. Try a different approach from what's been tried."""

# ── Diff mode template (matching openevolve diff_user.txt) ───────────────

DIFF_USER = """\
Make targeted improvements to maximize the evaluator score. \
Run: {eval_command} — it outputs JSON metrics. \
Current best fitness: {fitness:.4f}. {focus}

Past results: sqlite3 {db_path} "SELECT commit_hash, fitness_score, changes_description FROM programs WHERE session_id='{session_id}' ORDER BY fitness_score DESC LIMIT 5"

Focus on targeted changes, not a full rewrite. Try something different from previous attempts."""

# ── Sub-templates (matching openevolve top_program.txt, etc.) ────────────

TOP_PROGRAM_ENTRY = """\
### Program {rank}
- Fitness Score: {fitness:.4f}
- Key Implementation:
```{language}
{code}
```
- Notable Features: {key_features}
"""

INSPIRATION_ENTRY = """\
### Program {rank}
- Fitness Score: {fitness:.4f}
- Program Type: {program_type}
```{language}
{code}
```
- Unique Features: {unique_features}
"""

# Matches openevolve previous_attempt.txt
PREVIOUS_ATTEMPT = """\
### Attempt {attempt_number}
- Changes: {changes}
- Metrics: {performance}
- Outcome: {outcome}
"""

INSPIRATIONS_SECTION = """\
## Inspiration Programs

These programs represent diverse approaches and creative solutions that may inspire new ideas:

{inspiration_programs}
"""

METRICS_LINE = "- {name}: {value}"

# ── Fragments (matching openevolve fragments.json) ───────────────────────

FRAGMENTS = {
    "fitness_improved": "Fitness improved: {prev:.4f} → {current:.4f}",
    "fitness_declined": "Fitness declined: {prev:.4f} → {current:.4f}. Consider revising recent changes.",
    "fitness_stable": "Fitness unchanged at {current:.4f}",
    "outcome_all_improved": "All metrics improved",
    "outcome_all_regressed": "All metrics regressed",
    "outcome_mixed": "Mixed results",
    "no_specific_guidance": "Focus on improving fitness while maintaining diversity",
    "default_improvement": "Focus on improving the fitness score while exploring diverse solutions",
}


# ── Builder helpers ──────────────────────────────────────────────────────

def format_metrics(metrics: dict) -> str:
    if not metrics:
        return "  (no metrics)"
    return "\n".join(
        METRICS_LINE.format(name=k, value=v)
        for k, v in metrics.items()
    )


def _identify_improvement_areas(parent: dict, history: list[dict]) -> str:
    """Analyze fitness trajectory and suggest focus areas (matches openevolve)."""
    fitness = parent.get("fitness_score") or 0
    metrics = parent.get("metrics") or {}
    if isinstance(metrics, str):
        import json
        try:
            metrics = json.loads(metrics)
        except Exception:
            metrics = {}

    areas = []

    # Fitness trajectory
    if len(history) >= 2:
        prev = history[-2].get("fitness_score") or 0
        curr = history[-1].get("fitness_score") or 0
        if curr > prev:
            areas.append(FRAGMENTS["fitness_improved"].format(prev=prev, current=curr))
        elif curr < prev:
            areas.append(FRAGMENTS["fitness_declined"].format(prev=prev, current=curr))
        else:
            areas.append(FRAGMENTS["fitness_stable"].format(current=curr))

    if not areas:
        areas.append(FRAGMENTS["default_improvement"])

    return " | ".join(areas)


def format_history(history: list[dict], max_entries: int = 10) -> str:
    if not history:
        return "  (no previous attempts)"
    recent = history[-max_entries:]
    blocks = []
    prev_metrics = None
    for idx, h in enumerate(recent):
        fitness = h.get("fitness_score") or 0
        metrics = h.get("metrics") or {}
        if isinstance(metrics, str):
            import json
            try:
                metrics = json.loads(metrics)
            except Exception:
                metrics = {}

        # Determine outcome
        if prev_metrics and metrics:
            improved = sum(1 for k in metrics if k in prev_metrics and metrics[k] > prev_metrics[k])
            regressed = sum(1 for k in metrics if k in prev_metrics and metrics[k] < prev_metrics[k])
            if improved > 0 and regressed == 0:
                outcome = FRAGMENTS["outcome_all_improved"]
            elif regressed > 0 and improved == 0:
                outcome = FRAGMENTS["outcome_all_regressed"]
            else:
                outcome = FRAGMENTS["outcome_mixed"]
        else:
            outcome = "Baseline"

        blocks.append(PREVIOUS_ATTEMPT.format(
            attempt_number=h.get("iteration", idx + 1),
            changes=h.get("changes_description", "Unknown changes")[:100],
            performance=format_metrics(metrics),
            outcome=outcome,
        ))
        prev_metrics = metrics
    return "\n".join(blocks)


def format_top_programs(programs: list[dict], language: str = "python") -> str:
    if not programs:
        return "  (none yet)"
    blocks = []
    for i, p in enumerate(programs, 1):
        metrics = p.get("metrics") or {}
        if isinstance(metrics, str):
            import json
            try:
                metrics = json.loads(metrics)
            except Exception:
                metrics = {}
        # Identify key features
        features = []
        if metrics:
            best_metric = max(metrics, key=lambda k: metrics[k])
            features.append(f"Strong in {best_metric}")
        blocks.append(TOP_PROGRAM_ENTRY.format(
            rank=i,
            fitness=p.get("fitness_score") or 0,
            language=language,
            code=p.get("code", "")[:2000],
            key_features=", ".join(features) if features else "N/A",
        ))
    return "\n".join(blocks)


def format_inspirations(programs: list[dict], language: str = "python") -> str:
    if not programs:
        return ""
    blocks = []
    type_labels = ["Diverse", "Alternative", "Experimental", "Exploratory"]
    for i, p in enumerate(programs):
        metrics = p.get("metrics") or {}
        if isinstance(metrics, str):
            import json
            try:
                metrics = json.loads(metrics)
            except Exception:
                metrics = {}
        meta = p.get("metadata") or {}
        if isinstance(meta, str):
            import json
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        ptype = "Migrant" if meta.get("migrant") else type_labels[i % len(type_labels)]
        features = []
        if metrics:
            for k, v in metrics.items():
                features.append(f"Alternative {k} approach")
                break
        blocks.append(INSPIRATION_ENTRY.format(
            rank=i + 1,
            fitness=p.get("fitness_score") or 0,
            program_type=ptype,
            language=language,
            code=p.get("code", "")[:2000],
            unique_features=", ".join(features) if features else "Different approach to the problem",
        ))
    section = INSPIRATIONS_SECTION.format(inspiration_programs="\n".join(blocks))
    return section


def build_prompt(
    *,
    best_fitness: float = 0,
    db_path: str = "",
    session_id: str = "",
    eval_command: str = "python3 eval.py",
    focus: str = "",
    mode: str = "full_rewrite",
) -> tuple[str, str]:
    """Build the evolve prompt — short and directive.

    Returns (system_message, user_message).
    """
    template = DIFF_USER if mode == "diff" else FULL_REWRITE_USER
    user_msg = template.format(
        fitness=best_fitness,
        db_path=db_path,
        session_id=session_id,
        eval_command=eval_command,
        focus=focus,
    )
    return SYSTEM_MESSAGE, user_msg
