#!/usr/bin/env python3
"""SEE E2E real-LLM harness — manual smoke test (not part of pytest suite)."""
import json
import sys
import os
import logging

# Enable logging from the skill_evolution modules
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

# Ensure we import from the repo, not installed package
sys.path.insert(0, "/usr/local/lib/hermes-agent")

from run_agent import AIAgent
from agent.skill_evolution import evolve_skill

SYNTHETIC_SKILL = """---
name: github-pr-create
title: Create a GitHub Pull Request via gh CLI
description: |
  Creates a GitHub PR using the gh command-line tool.
  Assumes gh is installed and authenticated.
---

# Create PR

1. Run: `gh pr create --title "X" --body "Y"`
2. Parse the PR URL from stdout.
3. Return the PR number.
"""

# Simulated failure output: the user ran the skill, but gh is NOT installed.
FAILURE_OUTPUT = """$ gh pr create --title "feat: add X" --body "detailed description"
bash: gh: command not found
"""

FAILURE_SIGNAL = "Skill failed because gh CLI was not installed. The skill did not pre-validate the environment."

TASK_CONTEXT = "User asked to create a GitHub pull request for branch 'feature-x'."


def build_parent_agent():
    """Build a minimal AIAgent parent for credential inheritance."""
    return AIAgent(
        model="",  # fall back to config default
        max_iterations=3,
        enabled_toolsets=[],  # parent doesn't need tools
        quiet_mode=False,  # we want to see child logs
    )


def main():
    print("=" * 70)
    print("SEE E2E Real-LLM Validation — Phase 1: Synthetic Skill")
    print("=" * 70)

    parent = build_parent_agent()
    print(f"[harness] parent_agent model={getattr(parent, 'model', '?')}")
    print(f"[harness] synthetic skill: github-pr-create ({len(SYNTHETIC_SKILL)} chars)")

    result = evolve_skill(
        skill_name="github-pr-create",
        skill_content=SYNTHETIC_SKILL,
        task_context=TASK_CONTEXT,
        output_stdout=FAILURE_OUTPUT,
        failure_signal=FAILURE_SIGNAL,
        parent_agent=parent,
        category="github",  # for cache namespace
        config={
            "evolution": {
                "budget": 5,             # PUCT iterations
                "max_children": 3,       # K candidates per expansion
                "max_cost_usd": 0.50,    # generous ceiling for free-tier models
            },
        },
    )

    print()
    print("=" * 70)
    print("SEE E2E RESULT")
    print("=" * 70)
    # Save result to file for later inspection
    with open("/tmp/see_e2e_phase1_result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[result saved to /tmp/see_e2e_phase1_result.json]")
    print()

    # Print a summary using the ACTUAL result schema from _result()
    best_patch = result.get("best_patch") or {}
    cost = result.get("cost") or {}
    print(f"skill_name:              {result.get('skill_name', '?')}")
    print(f"iterations:              {result.get('iterations', 0)}")
    print(f"nodes_explored:          {result.get('nodes_explored', 0)}")
    print(f"hypotheses:              {len(result.get('hypotheses', []))}")
    print(f"budget_exceeded:         {result.get('budget_exceeded', False)}")
    print(f"elapsed_s:              {result.get('elapsed_s', 0):.1f}")
    if cost:
        print(f"cost.total_usd:         ${cost.get('total_usd', 0):.4f}")
        print(f"cost.calls:             {cost.get('calls', 0)}")
    if best_patch:
        print(f"best_patch.score:       {best_patch.get('evidence_score', 0):.3f}")
        print(f"best_patch.depth:       {best_patch.get('depth', 0)}")
        print(f"best_patch.node_id:     {best_patch.get('node_id', '?')}")
        print(f"best_patch.patches:     {len(best_patch.get('patches', []))}")
    else:
        print("best_patch:             (none — no patch improved on original)")
    if result.get("error"):
        print(f"error:                  {result.get('error')}")
    print()

    # Heuristic pass criteria (soft — real LLM may be noisy)
    # The patched skill preview is in best_patch.patched_skill_preview
    patch_preview = (best_patch.get("patched_skill_preview", "") or "").lower()
    patches_list = best_patch.get("patches", []) or []
    # Combine all patch new_strings
    all_patch_text = patch_preview + " " + " ".join(
        p.get("new_string", "") for p in patches_list
    ).lower()

    if "gh" in all_patch_text and ("which" in all_patch_text or "command -v" in all_patch_text or "installed" in all_patch_text or "not found" in all_patch_text):
        print("✓ PASS: patched skill now references gh installation check")
    else:
        print("⚠ REVIEW: patched skill does NOT clearly add a gh installation check")
        print("  (may still be valid — inspect /tmp/see_e2e_phase1_result.json)")

    # Critical-link verification: with the corrected PROMPT_TEST_GEN, a
    # FUNCTIONAL patch (pre-flight gh guard added) SHOULD score > 0.0 because
    # tests now pass == defect RESOLVED. Before the fix, all nodes scored 0.0
    # even when the patch was correct (tests inverted: pass==defect present).
    best_score = float(best_patch.get("evidence_score", 0))
    if best_score > 0.0:
        print(f"✓ CRITICAL-LINK PASS: best node scored {best_score:.3f} > 0.0 — the "
              "test synthesis rewarded a functional patch (inversion fixed).")
    else:
        print(f"⚠ CRITICAL-LINK REVIEW: best node scored {best_score:.3f} — the "
              "engine still failed to reward the patch; inspect evidence_matrix "
              "in /tmp/see_e2e_phase1_result.json for test-by-test reasons.")

    # Close the parent
    close = getattr(parent, "close", None)
    if callable(close):
        close()


if __name__ == "__main__":
    main()
