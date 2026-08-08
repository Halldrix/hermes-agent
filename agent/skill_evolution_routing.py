"""
SEE Prototype — Per-Role Model Routing

_delegate_role: spawns an AIAgent with provider:model configurable per role
of the reflection agent, without going through delegate_task (which only accepts override
global via delegation.* in config.yaml).

Roles:
  - hypothesis: generates falsifiable hypotheses (recoverable, inherited)
  - test:       synthesizes executable tests (critical link, EXPENSIVE)
  - patch:      proposes patches with ordinal ranking (recoverable, inherited)
  - execute:    executes the task with injected skill (output only, CHEAP)
  - decide:     chooses node to deploy (has deterministic fallback, inherited)

Configuration (config.yaml):
  evolution:
    models:
      test:
        provider: anthropic
        model: claude-opus-4-7
      execute:
        provider: openrouter
        model: meta-llama/llama-3.3-70b-instruct

If a role has no override, it falls back to global delegation.*, then to parent_agent.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)

# AIAgent imported lazily to not break tests without the package
_AIAgent = None
def _get_aiagent():
    global _AIAgent
    if _AIAgent is None:
        from run_agent import AIAgent
        _AIAgent = AIAgent
    return _AIAgent


# ── Token defaults per role (for predictive cost estimation) ──
_DEFAULT_TOKEN_BUDGET = {
    "hypothesis": {"input": 3000, "output": 500},
    "test":       {"input": 2000, "output": 800},
    "patch":      {"input": 4000, "output": 1000},
    "execute":    {"input": 6000, "output": 2000},
    "decide":     {"input": 2000, "output": 200},
}


# ──────────────────────────────────────────────────────────────────────
# 1. Credential resolution per role
# ──────────────────────────────────────────────────────────────────────

def _resolve_role_credentials(role: str, parent_agent) -> dict:
    """Resolve provider:model for a role of the reflection agent.

    Order of precedence:
    1. evolution.models.<role>.provider/model (from config.yaml)
    2. delegation.provider/model (global)
    3. inheritance from parent_agent

    Returns dict:
        {model, provider, base_url, api_key, api_mode}
    """
    from hermes_cli.config import load_config, cfg_get

    try:
        cfg = load_config()
    except Exception:
        cfg = {}

    # 1. Role-specific override
    role_provider = cfg_get(cfg, "evolution", "models", role, "provider")
    role_model = cfg_get(cfg, "evolution", "models", role, "model")
    role_provider = str(role_provider).strip() if role_provider else None
    role_model = str(role_model).strip() if role_model else None

    # 2. Fallback to global delegation.*
    if not role_provider and not role_model:
        delegation_cfg = cfg.get("delegation", {}) if isinstance(cfg, dict) else {}
        return _resolve_delegation_fallback(delegation_cfg, parent_agent)

    # 3. Resolve runtime provider if provider was specified
    if role_provider:
        try:
            from hermes_cli.runtime_provider import resolve_runtime_provider
            runtime = resolve_runtime_provider(
                requested=role_provider,
                target_model=role_model,
            )
            return {
                "model": role_model or runtime.get("model"),
                "provider": role_provider,
                "base_url": runtime.get("base_url"),
                "api_key": runtime.get("api_key"),
                "api_mode": runtime.get("api_mode", "chat_completions"),
            }
        except Exception as e:
            logger.warning(
                "evolution: failed to resolve provider '%s' for role '%s': %s. "
                "Falling back to parent inheritance.",
                role_provider, role, e,
            )
            return _resolve_delegation_fallback(
                cfg.get("delegation", {}) if isinstance(cfg, dict) else {},
                parent_agent,
            )

    # 4. Only model set, no provider → use parent's provider
    return {
        "model": role_model,
        "provider": getattr(parent_agent, "provider", None),
        "base_url": getattr(parent_agent, "base_url", None),
        "api_key": None,  # inherits from parent
        "api_mode": getattr(parent_agent, "api_mode", None),
    }


def _resolve_delegation_fallback(delegation_cfg: dict, parent_agent) -> dict:
    """Fallback to standard delegate_task behavior."""
    configured_model = str(delegation_cfg.get("model") or "").strip() or None
    configured_provider = str(delegation_cfg.get("provider") or "").strip() or None

    if not configured_provider and not configured_model:
        # Inherit everything from parent
        return {
            "model": getattr(parent_agent, "model", None),
            "provider": getattr(parent_agent, "provider", None),
            "base_url": getattr(parent_agent, "base_url", None),
            "api_key": None,
            "api_mode": getattr(parent_agent, "api_mode", None),
        }

    if configured_provider:
        try:
            from hermes_cli.runtime_provider import resolve_runtime_provider
            runtime = resolve_runtime_provider(
                requested=configured_provider,
                target_model=configured_model,
            )
            return {
                "model": configured_model or runtime.get("model"),
                "provider": configured_provider,
                "base_url": runtime.get("base_url"),
                "api_key": runtime.get("api_key"),
                "api_mode": runtime.get("api_mode", "chat_completions"),
            }
        except Exception:
            pass  # fall to the final return

    return {
        "model": configured_model or getattr(parent_agent, "model", None),
        "provider": getattr(parent_agent, "provider", None),
        "base_url": getattr(parent_agent, "base_url", None),
        "api_key": None,
        "api_mode": getattr(parent_agent, "api_mode", None),
    }


# ──────────────────────────────────────────────────────────────────────
# 2. _delegate_role — the reflection agent spawn
# ──────────────────────────────────────────────────────────────────────

def _delegate_role(
    role: str,
    system_prompt: str,
    user_msg: str,
    parent_agent,
    expected_json_key: Optional[str] = None,
    max_iterations: int = 3,
    budget_tracker=None,
) -> Any:
    """Spawns an AIAgent with provider:model configured for the role.

    Does not go through delegate_task to avoid the global override. Builds a
    Direct AIAgent with ephemeral_system_prompt and empty toolsets — the
    reflection agent reasons only, does not use tools.

    Args:
        role: one of "hypothesis", "test", "patch", "execute", "decide".
        system_prompt: system instructions (one of PROMPT_HYPOTHESIS_GEN, etc.).
        user_msg: content of the user message with skill + output + evidence.
        parent_agent: the parent AIAgent (for credential inheritance if no override).
        expected_json_key: if set, parses the response as JSON and extracts that key.
        max_iterations: max iterations of the child's tool loop (default 3).
        budget_tracker: optional BudgetTracker — if provided, actual token usage
            from the child agent is recorded against the budget after the call.

    Returns:
        - If expected_json_key is None: the child's response text.
        - If expected_json_key is set: the value of that key (list/dict), or [] on failure.
    """
    creds = _resolve_role_credentials(role, parent_agent)
    AIAgent = _get_aiagent()

    child = AIAgent(
        model=creds.get("model") or getattr(parent_agent, "model", ""),
        provider=creds.get("provider"),
        base_url=creds.get("base_url"),
        api_key=creds.get("api_key"),
        api_mode=creds.get("api_mode"),
        max_iterations=max_iterations,
        enabled_toolsets=[],               # no tools → only reasoning
        disabled_toolsets=None,
        ephemeral_system_prompt=system_prompt,
        skip_memory=True,                    # don't pollute with parent memory
        skip_background_review=True,         # no disparar curator fork
        skip_context_files=True,             # no cargar AGENTS.md/CLAUDE.md
        load_soul_identity=False,
        quiet_mode=True,                     # silence child logs
        log_prefix=f"[see-{role}] ",
    )

    response_text = ""
    try:
        # AIAgent.chat() returns the assistant's final text as a str.
        # The ephemeral_system_prompt was passed to the constructor, so chat()
        # uses it automatically — no need to pass a messages list.
        response_text = child.chat(user_msg)
        if isinstance(response_text, dict):
            # Some installations return a dict with metadata
            response_text = str(response_text.get("content") or
                                response_text.get("response") or "")
        response_text = str(response_text or "").strip()

        # Record actual token usage in the budget tracker.
        if budget_tracker is not None:
            usage = getattr(child, "last_usage", None)
            if usage is not None:
                budget_tracker.track(
                    role=role,
                    model=creds.get("model") or getattr(parent_agent, "model", ""),
                    input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                    output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
                    provider=creds.get("provider"),
                    base_url=creds.get("base_url"),
                )
    except Exception as e:
        logger.warning("_delegate_role[%s] failed: %s", role, e)
        return [] if expected_json_key else ""
    finally:
        # Release child agent resources
        close = getattr(child, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    if not response_text:
        return [] if expected_json_key else ""

    if expected_json_key:
        return _extract_json_key(response_text, expected_json_key)
    return response_text


def _extract_json_key(text: str, key: str) -> Any:
    """Extract the value of a key from a JSON embedded in text."""
    # Strategy 1: find the first {...} and parse
    j_start = text.find("{")
    j_end = text.rfind("}") + 1
    if j_start != -1 and j_end > j_start:
        try:
            return json.loads(text[j_start:j_end]).get(key, [])
        except json.JSONDecodeError:
            pass
    # Strategy 2: find an array [...] if the key is plural
    if key.endswith("s") or key in ("patches", "hypotheses"):
        a_start = text.find("[")
        a_end = text.rfind("]") + 1
        if a_start != -1 and a_end > a_start:
            try:
                arr = json.loads(text[a_start:a_end])
                return arr if isinstance(arr, list) else []
            except json.JSONDecodeError:
                pass
    logger.warning("_extract_json_key: could not parse '%s' from response", key)
    return []


# ──────────────────────────────────────────────────────────────────────
# 3. Role wrappers (sets system_prompt and expected_json_key)
# ──────────────────────────────────────────────────────────────────────

# Operational prompts — kept in skill_evolution.py in the full design.
# Skeletons here so the prototype is executable.

PROMPT_HYPOTHESIS_GEN = """You are a diagnostic engine for AI agent skills.
Propose 3-5 NEW falsifiable hypotheses about what specific defect in the skill
caused the failure. Each hypothesis must be specific, falsifiable, and atomic.

Output JSON:
{
  "hypotheses": [
    {
      "action": "add" | "refine" | "refute",
      "id": "H<n>",
      "description": "<falsifiable defect description>",
      "observable_behavior": "<what we should observe to confirm/refute>",
      "rationale": "<why this hypothesis explains the failure>"
    }
  ]
}
"""

PROMPT_TEST_GEN = """You are a test synthesis engine. Write a Python function
`def check(output: str, files: dict) -> dict` that checks the cached output
for the predicted behavior described in the hypothesis.

The function must:
- Return: {{"pass": bool, "reason": str, "category": "hard" | "semantic"}}
- Use only json, re, os.path, yaml imports (no subprocess/socket/open).
- Be deterministic and run in under 1 second.

Output ONLY the function definition in a code block.
"""

PROMPT_PATCH_GEN = """You are a skill revision engine. Propose up to {K}
candidate patches to the SKILL.md. Each patch is a targeted find-and-replace.
Rank them ordinally (1=best). Output JSON with 'patches' key.

Each patch MUST have these exact keys:
  - "rank": integer (1=best)
  - "old_string": exact substring from the current SKILL.md to find
  - "new_string": replacement string
  - "rationale": one-line explanation
  - "expected_improvement": what this fixes

If evidence is insufficient, return {{"patches": [], "reason": "..."}}
"""


def delegate_hypothesis(
    skill_content: str,
    task_context: str,
    failure_signal: str,
    output_stdout: str,
    existing_hypotheses: list[dict],
    parent_agent,
    focus: Optional[str] = None,
    round_idx: int = 1,
    total_rounds: int = 3,
    budget_tracker=None,
) -> list[dict]:
    """Launch the reflection agent in hypothesis mode."""
    user_msg = f"""## Current SKILL.md:
```markdown
{skill_content[:10000]}
```

## Task attempted:
{task_context[:2000]}

## Output produced (failed):
```
{output_stdout[:3000]}
```

## Failure signal:
{failure_signal or "(implicit — the output was rejected)"}

## Existing hypotheses (round {round_idx}/{total_rounds}):
{json.dumps(existing_hypotheses, indent=2)}

## Focus hint:
{focus or "(none — explore freely)"}
"""
    return _delegate_role(
        role="hypothesis",
        system_prompt=PROMPT_HYPOTHESIS_GEN,
        user_msg=user_msg,
        parent_agent=parent_agent,
        expected_json_key="hypotheses",
        budget_tracker=budget_tracker,
    )


def delegate_test(
    hypothesis_description: str,
    observable_behavior: str,
    output_stdout: str,
    file_list: list[str],
    parent_agent,
    budget_tracker=None,
) -> Optional[str]:
    """Launch the reflection agent in test mode (expensive model).
    Returns the Python test code, or None on failure.
    """
    user_msg = f"""Hypothesis: {hypothesis_description}
Observable behavior that confirms/refutes: {observable_behavior}
Cached output (stdout): {(output_stdout or '')[:2000]}
Files produced: {file_list or ['(none)']}
"""
    response = _delegate_role(
        role="test",
        system_prompt=PROMPT_TEST_GEN.format(
            hypothesis_description=hypothesis_description,
            observable_behavior=observable_behavior,
            stdout_preview=(output_stdout or '')[:1500],
            file_list=file_list or [],
        ),
        user_msg=user_msg,
        parent_agent=parent_agent,
        budget_tracker=budget_tracker,
    )
    if not response:
        return None
    return _extract_code_block(response)


def delegate_patch(
    skill_content: str,
    evidence_summary: str,
    task_context: str,
    max_children: int,
    parent_agent,
    budget_tracker=None,
) -> list[dict]:
    """Launch the reflection agent in patch mode."""
    user_msg = f"""## Current SKILL.md:
```markdown
{skill_content[:12000]}
```

## Evidence at this node:
{evidence_summary}

## Task context:
{task_context[:1000]}
"""
    return _delegate_role(
        role="patch",
        system_prompt=PROMPT_PATCH_GEN.format(K=max_children),
        user_msg=user_msg,
        parent_agent=parent_agent,
        expected_json_key="patches",
        budget_tracker=budget_tracker,
    )


def delegate_execute(
    skill_content: str,
    task_context: str,
    parent_agent,
    budget_tracker=None,
) -> str:
    """Launch the reflection agent in execute mode (cheap model).
    Executes the task with the injected skill. Returns the output (stdout).
    """
    user_msg = (
        "You have access to the following skill. Follow its guidance when "
        f"relevant to the task:\n\n{skill_content[:12000]}\n\n"
        f"Answer the task below.\n\nTask:\n{task_context}"
    )
    return _delegate_role(
        role="execute",
        system_prompt="Follow the skill's instructions to complete the task. Return only the final output.",
        user_msg=user_msg,
        parent_agent=parent_agent,
        budget_tracker=budget_tracker,
    )


def _extract_code_block(text: str) -> Optional[str]:
    """Extract the first Python code block from the LLM response."""
    # ```python ... ``` or ``` ... ```
    pattern = r"```(?:python)?\s*\n(.*?)\n```"
    m = re.search(pattern, text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # No code block — try the whole response if it looks like Python
    if "def check(" in text:
        return text.strip()
    return None
