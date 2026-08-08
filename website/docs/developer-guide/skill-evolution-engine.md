---
sidebar_position: 20
title: "Skill Evolution Engine (SEE)"
description: "Self-evolving skills — PUCT search over skill patches, hypothesis-driven verification, and an evidence matrix"
---

# Skill Evolution Engine (SEE)

The **Skill Evolution Engine (SEE)** is Hermes Agent's self-improvement subsystem: when a skill fails, SEE *diagnoses why* (hypotheses), *proves it* (executable tests), and *fixes it* (candidate patches) — using real LLM delegates for each role, then picks the patch with the strongest evidence.

```text
failure → hypotheses → tests → patches → simulation → evidence matrix → best patch
```

## Module Layout

| File | Responsibility |
|---|---|
| `agent/skill_evolution.py` | Orchestrator: PUCT loop, evidence matrix, `evolve_skill()` entry point |
| `agent/skill_evolution_routing.py` | `_delegate_role()` — spawns per-role AIAgents with model routing |
| `agent/skill_evolution_sandbox.py` | Restricted sandbox that validates + executes generated tests |
| `agent/skill_evolution_test_cache.py` | Per-skill cache of expensive test code (thread-safe, best-effort) |
| `agent/skill_evolution_budget.py` | `BudgetTracker` — cost prediction, `max_cost_usd` enforcement |

This subsystem is fully extensible: every delegate (hypothesis, test, patch, execute) is injectable for testing, and all four helpers are independent modules.

## The PUCT Algorithm

PUCT (**P**redictor + **U**pper **C**onfidence bounds applied to **T**rees) balances exploration vs. exploitation when choosing which skill variant to test next:

```text
UCB1(node) = avg_value(node) + C · √( ln(N_parent) / n_node )
```

1. **Selection** — traverse from root, always picking the child with max UCB1 (`PUCTNode.ucb1()`)
2. **Expansion** — ask the LLM to generate K candidate patches (`delegate_patch`); each becomes a child node
3. **Simulation** — execute the patched skill, run every hypothesis's test, fill `M[node, test_id]`
4. **Backpropagation** — propagate the evidence score up the tree, updating visits/value

Exploration constant: `UCB_C = √2` (default). Unvisited nodes always get `∞` (explored first).

## Evidence Matrix

`M[node_id, test_id]` stores one of three values:

| Value | Meaning |
|---|---|
| `1` | Test **passed** — patched skill satisfies the hypothesis |
| `0` | Test **failed** — patched skill does not satisfy the hypothesis |
| `None` | Test not executed for this node |

A hypothesis is **confirmed** if all its tests pass in at least one node. The winning patch is the node with the highest `evidence_score` (passed/total tests), **preferring shallower depth** (simpler patches) on ties.

## Roles and Delegates

`_delegate_role()` spawns a dedicated `AIAgent` per role with:
- `ephemeral_system_prompt` — role-specific instructions (hypothesis/test/patch/execute prompts)
- `enabled_toolsets=[]` — the reflection agent only *reasons*, never calls tools
- `skip_memory=True`, `skip_background_review=True` — isolated from parent's context
- `quiet_mode=True` — silences child logs; `log_prefix=[see-<role>]`

Default model budgets per role (for predictive cost estimation):

| Role | Input tokens | Output tokens |
|---|---|---|
| `hypothesis` | 3,000 | 500 |
| `test` | 2,000 | 800 |
| `patch` | 4,000 | 1,000 |
| `execute` | 6,000 | 2,000 |
| `decide` | 2,000 | 200 |

## Test Sandbox

Generated tests must be `def check(output, files) -> dict` — deterministic, fast (<1s), pure functions.

- **Static validation** (`validate_test_strict`): syntax check, forbidden-token scan, return-dict heuristic
- **Trivial-test detection**: dry-run against 3 scenarios (empty, garbage, plausible); if all pass → rejected as "trivially permissive"
- **Restricted namespace**: only `json, re, os, os.path, yaml, io`; `subprocess/socket/urllib/requests/open()` and friends are blocked
- **Timeout**: SIGALRM-based 5s default
- Allowed imports: `json, re, os, os.path, yaml, io`

## Test Cache

Test code is expensive (created by the most capable model). `TestCache` stores validated tests per skill under `~/.hermes/skills/<category>/<skill>/.evolution_cache/`:

```
.evolution_cache/
├── .test_cache.lock      # advisory flock
├── tests/
│   ├── manifest.json     # metadata index (version 2)
│   ├── tests_archive.json # refuted-test audit trail
│   ├── H1_<hash>.py      # test source code
│   └── ...
```

- **Key** = hash(description + skill_version_hash + observable_behavior)
- **Stale invalidation**: when the skill changes, matching tests are marked stale
- **TTL pruning**: 30 days default
- **Best-effort**: any I/O or lock failure degrades to "miss" — never blocks evolution

## Budget Tracking

`BudgetTracker` (from `evolution.models.<role>`) predicts and enforces cost:

1. `check_budget(role, model)` → raises `BudgetExceededError` if `spent + predict(role) > max_cost_usd`
2. `track(role, model, in_tokens, out_tokens)` records actual usage
3. Pricing from `DEFAULT_PRICING` dict (2026 USD/1M-token estimates), defaulting to `$0.00` on unknown models (free-tier)

## Configuration

```yaml
# config.yaml
evolution:
  max_cost_usd: 0.50           # hard budget cap (None = unlimited)
  cost_warning_threshold: 0.8  # warn at 80% of budget
  budget: 5                    # PUCT iterations
  max_children: 3              # candidate patches per expansion (K)
  cache:
    test_ttl_days: 30
  models:
    hypothesis:                # per-role model override
      provider: nous
      model: hermes-4-405b
    test:
      provider: anthropic
      model: claude-opus-4-7
    patch:
      provider: openrouter
      model: meta-llama/llama-3.3-70b-instruct
```

Precedence: `evolution.models.<role>` → `delegation.*` (global) → parent `AIAgent` inheritance.

## Public API

```python
from agent.skill_evolution import evolve_skill

result = evolve_skill(
    skill_name="github-pr-create",
    skill_content=open("SKILL.md").read(),
    task_context="Create a PR with title X and body Y",
    output_stdout=stdout_from_failed_run,
    parent_agent=parent_aiagent,              # inherited credentials
    failure_signal="gh: command not found",   # optional diagnostic hint
    config={"evolution": {"max_cost_usd": 0.5}},
    category="github",                        # cache scope
)
```

Returned dict:
```python
{
  "skill_name": str,
  "iterations": int,           # PUCT iterations
  "nodes_explored": int,
  "hypotheses": [{"id", "description", "status"}],
  "evidence_matrix": {node_id: {test_id: {"passed", "reason", "category"}}},
  "budget_exceeded": bool,
  "elapsed_s": float,
  "best_patch": {              # None if no improvement found
    "node_id": str, "evidence_score": float, "depth": int,
    "patches": [{"rank", "old_string", "new_string"}],
    "patched_skill_preview": str,   # first 500 chars
  },
  "cost": {"cost_total_usd", "max_cost_usd", "cost_breakdown"},
}
```

## Validated End-to-End

Validated on 2026-08-08 with two real LLM runs (free-tier model, $0.00 cost):

| Phase | Skill | Hypotheses | Best patch result |
|---|---|---|---|
| Synthetic | `github-pr-create` | 4 | Added `command -v gh` pre-flight check + install remediation |
| Real | `infinityfree-deploy` | 5 | Added `command -v lftp` check + multi-OS install fallback |

Both runs produced a working patch that a human reviewer would have written by hand. See `tests/agent/see_e2e_*.py` for the harness.