"""
SEE Prototype — Model Routing por Rol

_delegate_role: spawnea un AIAgent con provider:model configurable por rol
del reflection agent, sin pasar por delegate_task (que solo acepta override
global via delegation.* en config.yaml).

Roles:
  - hypothesis: genera hipótesis falsables (recuperable, heredado)
  - test:       sintetiza tests ejecutables (eslabón crítico, CARO)
  - patch:      propone parches con ranking ordinal (recuperable, heredado)
  - execute:    ejecuta el task con skill inyectado (solo output, BARATO)
  - decide:     elige nodo a desplegar (hay fallback determinista, heredado)

Configuración (config.yaml):
  evolution:
    models:
      test:
        provider: anthropic
        model: claude-opus-4-7
      execute:
        provider: openrouter
        model: meta-llama/llama-3.3-70b-instruct

Si un rol no tiene override, cae a delegation.* global, luego al parent_agent.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)

# AIAgent se importa bajo demanda para no romper tests que no tengan el paquete
_AIAgent = None
def _get_aiagent():
    global _AIAgent
    if _AIAgent is None:
        from run_agent import AIAgent
        _AIAgent = AIAgent
    return _AIAgent


# ── Defaults de tokens por rol (para estimación predictiva de costo) ──
_DEFAULT_TOKEN_BUDGET = {
    "hypothesis": {"input": 3000, "output": 500},
    "test":       {"input": 2000, "output": 800},
    "patch":      {"input": 4000, "output": 1000},
    "execute":    {"input": 6000, "output": 2000},
    "decide":     {"input": 2000, "output": 200},
}


# ──────────────────────────────────────────────────────────────────────
# 1. Resolución de credenciales por rol
# ──────────────────────────────────────────────────────────────────────

def _resolve_role_credentials(role: str, parent_agent) -> dict:
    """Resuelve provider:model para un rol del reflection agent.

    Orden de precedencia:
    1. evolution.models.<role>.provider/model (de config.yaml)
    2. delegation.provider/model (global)
    3. herencia del parent_agent

    Retorna dict:
        {model, provider, base_url, api_key, api_mode}
    """
    from hermes_cli.config import load_config, cfg_get

    try:
        cfg = load_config()
    except Exception:
        cfg = {}

    # 1. Override específico del rol
    role_provider = cfg_get(cfg, "evolution", "models", role, "provider")
    role_model = cfg_get(cfg, "evolution", "models", role, "model")
    role_provider = str(role_provider).strip() if role_provider else None
    role_model = str(role_model).strip() if role_model else None

    # 2. Fallback a delegation.* global
    if not role_provider and not role_model:
        delegation_cfg = cfg.get("delegation", {}) if isinstance(cfg, dict) else {}
        return _resolve_delegation_fallback(delegation_cfg, parent_agent)

    # 3. Resolver el runtime provider si se específicó provider
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

    # 4. Solo model seteado, sin provider → usar provider del parent
    return {
        "model": role_model,
        "provider": getattr(parent_agent, "provider", None),
        "base_url": getattr(parent_agent, "base_url", None),
        "api_key": None,  # hereda del parent
        "api_mode": getattr(parent_agent, "api_mode", None),
    }


def _resolve_delegation_fallback(delegation_cfg: dict, parent_agent) -> dict:
    """Fallback al comportamiento estándar de delegate_task."""
    configured_model = str(delegation_cfg.get("model") or "").strip() or None
    configured_provider = str(delegation_cfg.get("provider") or "").strip() or None

    if not configured_provider and not configured_model:
        # Heredar todo del parent
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
            pass  # caer al return final

    return {
        "model": configured_model or getattr(parent_agent, "model", None),
        "provider": getattr(parent_agent, "provider", None),
        "base_url": getattr(parent_agent, "base_url", None),
        "api_key": None,
        "api_mode": getattr(parent_agent, "api_mode", None),
    }


# ──────────────────────────────────────────────────────────────────────
# 2. _delegate_role — el spawn del reflection agent
# ──────────────────────────────────────────────────────────────────────

def _delegate_role(
    role: str,
    system_prompt: str,
    user_msg: str,
    parent_agent,
    expected_json_key: Optional[str] = None,
    max_iterations: int = 3,
) -> Any:
    """Spawnea un AIAgent con provider:model configurado para el rol.

    No pasa por delegate_task para evitar el override global. Construye un
    AIAgent directo con ephemeral_system_prompt y toolsets vacíos — el
    reflection agent solo razona, no usa tools.

    Args:
        role: uno de "hypothesis", "test", "patch", "execute", "decide".
        system_prompt: instrucciones del sistema (uno de PROMPT_HYPOTHESIS_GEN, etc.).
        user_msg: contenido del mensaje del usuario con skill + output + evidence.
        parent_agent: el AIAgent padre (para herencia de credentials si no hay override).
        expected_json_key: si se setea, parsea la respuesta como JSON y extrae esa key.
        max_iterations: máx iteraciones del tool loop del child (default 3).

    Returns:
        - Si expected_json_key es None: el texto de respuesta del child.
        - Si expected_json_key se setea: el valor de esa key (list/dict), o [] si falla.
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
        enabled_toolsets=[],               # sin tools → solo razona
        disabled_toolsets=None,
        ephemeral_system_prompt=system_prompt,
        skip_memory=True,                    # no contaminar con memoria del parent
        skip_background_review=True,         # no disparar curator fork
        skip_context_files=True,             # no cargar AGENTS.md/CLAUDE.md
        load_soul_identity=False,
        quiet_mode=True,                     # silenciar logs del child
        log_prefix=f"[see-{role}] ",
    )

    messages = [{"role": "user", "content": user_msg}]

    response_text = ""
    try:
        # AIAgent.run() retorna el texto final del assistant
        response_text = child.run(messages)
        if isinstance(response_text, dict):
            # Algunas instalaciones retornan un dict con metadata
            response_text = str(response_text.get("content") or
                                response_text.get("response") or "")
        response_text = str(response_text or "").strip()
    except Exception as e:
        logger.warning("_delegate_role[%s] failed: %s", role, e)
        return [] if expected_json_key else ""
    finally:
        # Liberar recursos del child agent
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
    """Extrae el valor de una key de un JSON embebido en texto."""
    # Estrategia 1: buscar el primer {...} y parsear
    j_start = text.find("{")
    j_end = text.rfind("}") + 1
    if j_start != -1 and j_end > j_start:
        try:
            return json.loads(text[j_start:j_end]).get(key, [])
        except json.JSONDecodeError:
            pass
    # Estrategia 2: buscar un array [...] si la key es plural
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
# 3. Wrappers por rol (establece system_prompt y expected_json_key)
# ──────────────────────────────────────────────────────────────────────

# Prompts operacionales — se mantienen en skill_evolution.py en el diseño completo.
# Aquí hay esqueletos para que el prototipo sea ejecutable.

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
) -> list[dict]:
    """Lanza el reflection agent en modo hypothesis."""
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
    )


def delegate_test(
    hypothesis_description: str,
    observable_behavior: str,
    output_stdout: str,
    file_list: list[str],
    parent_agent,
) -> Optional[str]:
    """Lanza el reflection agent en modo test (modelo caro).
    Retorna el código Python del test, o None si falla.
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
) -> list[dict]:
    """Lanza el reflection agent en modo patch."""
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
    )


def delegate_execute(
    skill_content: str,
    task_context: str,
    parent_agent,
) -> str:
    """Lanza el reflection agent en modo execute (modelo barato).
    Ejecuta el task con el skill inyectado. Retorna el output (stdout).
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
    )


def _extract_code_block(text: str) -> Optional[str]:
    """Extrae el primer bloque de código Python de la respuesta del LLM."""
    # ```python ... ``` o ``` ... ```
    pattern = r"```(?:python)?\s*\n(.*?)\n```"
    m = re.search(pattern, text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Sin code block — intentar toda la respuesta si parece Python
    if "def check(" in text:
        return text.strip()
    return None
