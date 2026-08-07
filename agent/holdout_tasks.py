"""Holdout task set for Gate 1B — ExecCritic behavioral replay.

A holdout task set is a frozen collection of reference tasks used to A-B test
candidate skills before admission. The set lives at
``~/.hermes/skills/.holdout_tasks.json`` and has the format:

    {
      "version": 1,
      "tasks": [
        {
          "id": "unique-task-id",
          "prompt": "The task prompt the replay agent must solve",
          "category": "git|testing|debugging|general",
          "expected_keywords": ["keyword1", "keyword2"],
          "timeout_seconds": 120
        }
      ]
    }

Design notes
------------
- Tasks are intentionally simple, deterministic, and fast (≤2 min each).
- Each task has a ``prompt`` (what the agent must do) and optional
  ``expected_keywords`` (words/phrases a good answer should mention).
- The A-B replay runs the task twice: once without the candidate skill
  (control) and once with the skill injected into context (treatment).
- Quality is scored by an LLM judge (single call) that compares the two
  outputs and decides which is better, or if they're equivalent.
- A skill passes Gate 1B if the treatment output is NOT WORSE than the
  control on any task, and is STRICTLY BETTER on at least one (i.e., the
  skill adds value or is neutral — never harmful).

The task set is user-extensible. A minimal seed set is provided by default
when no holdout file exists. Users can add domain-specific tasks to match
their workflow.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Default seed task set ──────────────────────────────────────────────
# These are intentionally generic, fast tasks that exercise common agent
# capabilities. They are NOT domain-specific — they test whether a skill
# improves or degrades baseline agent performance on routine work.

_SEED_TASKS: dict[str, Any] = {
    "version": 1,
    "tasks": [
        {
            "id": "git-status-interpret",
            "prompt": (
                "Given a git repository with uncommitted changes, explain "
                "what `git status --short` would show for a repo with one "
                "modified file and one untracked file. Keep the answer under "
                "5 lines."
            ),
            "category": "git",
            "expected_keywords": ["modified", "untracked", "M ", "?? "],
            "timeout_seconds": 60,
        },
        {
            "id": "python-list-comprehension",
            "prompt": (
                "Write a Python list comprehension that filters even numbers "
                "from a list and squares them. Explain in one sentence how "
                "it works."
            ),
            "category": "general",
            "expected_keywords": ["for", "if", "% 2", "x**2", "squared"],
            "timeout_seconds": 60,
        },
        {
            "id": "error-message-rewrite",
            "prompt": (
                "Rewrite this error message to be user-friendly: "
                "'Error 404: Resource not found at /api/v2/users/12345'. "
                "Keep it under 2 sentences."
            ),
            "category": "general",
            "expected_keywords": ["not found", "user", "modify"],
            "timeout_seconds": 60,
        },
    ],
}


def get_holdout_path() -> Path:
    """Return the path to the holdout task set file."""
    from hermes_constants import get_skills_dir
    return get_skills_dir() / ".holdout_tasks.json"


def load_holdout_tasks() -> list[dict[str, Any]]:
    """Load the holdout task set, creating a seed set if none exists.

    Returns a list of task dicts. Each has:
      id, prompt, category, expected_keywords (optional), timeout_seconds.
    """
    path = get_holdout_path()

    if not path.exists():
        logger.info("Holdout task set not found at %s; creating seed set", path)
        try:
            path.write_text(
                json.dumps(_SEED_TASKS, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("Could not write seed holdout tasks: %s", e)
        return list(_SEED_TASKS["tasks"])

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Could not parse holdout tasks at %s: %s", path, e)
        return list(_SEED_TASKS["tasks"])

    tasks = data.get("tasks", [])
    if not tasks:
        logger.warning("Holdout task set at %s is empty; using seed", path)
        return list(_SEED_TASKS["tasks"])

    return tasks


def save_holdout_tasks(tasks: list[dict[str, Any]]) -> Path:
    """Write a holdout task set to disk."""
    path = get_holdout_path()
    data = {"version": 1, "tasks": tasks}
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path
