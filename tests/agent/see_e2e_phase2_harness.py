#!/usr/bin/env python3
"""SEE E2E Phase 2: Real skill validation — manual smoke test."""
import json
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
sys.path.insert(0, "/usr/local/lib/hermes-agent")

from run_agent import AIAgent
from agent.skill_evolution import evolve_skill

# Real skill content from ~/.hermes/skills/software-development/infinityfree-deploy/SKILL.md
REAL_SKILL = """---
name: infinityfree-deploy
description: Deploy PHP/MySQL apps to InfinityFree free hosting.
---

# Deploy PHP/MySQL apps to InfinityFree

**Trigger:** user wants to deploy a PHP/MySQL web app to free hosting.

## Account anatomy (InfinityFree)
- cPanel login = MySQL username = if0_XXXXXXX
- MySQL password = your cPanel/vPanel password
- DB name is AUTO-PREFIXED: you type `todolist`, it becomes `if0_XXXXXXX_todolist`
- MySQL host = sql105.infinityfree.com
- FTP host = ftpupload.net
- Web root on FTP = htdocs/

## WORKFLOW
1. Write the PHP app using PDO + CREATE TABLE IF NOT EXISTS.
2. Create the MySQL DB in cPanel → MySQL Databases.
3. config.php: host=sql105.infinityfree.com, db=if0_XXXXXXX_name, user=if0_XXXXXXX, pass=cPanel password.
4. Upload via lftp.
5. Verify with the Hermes browser tool, not curl.

## GOTCHAS
- Anti-bot JS challenge: curl gets HTTP 000. Use browser tool.
- Remote MySQL is BLOCKED from outside. MySQL only works from PHP on their servers.
- FTP needs plain FTP: lftp must use `set ftp:ssl-allow no` + `set ssl:verify-certificate no`.
- Browser tool can flake — just retry.

## references/
- infinityfree-gotchas.md — anti-bot challenge, remote-MySQL block, lftp command.
- free-hosting-alternatives.md — Heroku/Render/Vercel/Neon status.
- templates/todolist/ — working CRUD app deployed on halldrix.freedev.app/todolist.
"""

# Simulated failure: lftp is not installed
FAILURE_OUTPUT = """$ lftp -c "open ftpupload.net; set ftp:ssl-allow no; set ssl:verify-certificate no; mirror -R /local/app htdocs/app"
bash: lftp: command not found
"""

FAILURE_SIGNAL = "Skill failed at step 4 (Upload via lftp) because lftp CLI is not installed. The skill does not pre-validate required tools."

TASK_CONTEXT = "User asked to deploy a PHP/MySQL todo app to InfinityFree hosting at halldrix.freedev.app."


def build_parent_agent():
    """Build a minimal AIAgent parent for credential inheritance."""
    return AIAgent(
        model="",
        max_iterations=3,
        enabled_toolsets=[],
        quiet_mode=False,
    )


def main():
    print("=" * 70)
    print("SEE E2E Real-LLM Validation — Phase 2: Real Skill (infinityfree-deploy)")
    print("=" * 70)

    parent = build_parent_agent()
    print(f"[harness] parent_agent model={getattr(parent, 'model', '?')}")
    print(f"[harness] real skill: infinityfree-deploy ({len(REAL_SKILL)} chars)")

    result = evolve_skill(
        skill_name="infinityfree-deploy",
        skill_content=REAL_SKILL,
        task_context=TASK_CONTEXT,
        output_stdout=FAILURE_OUTPUT,
        failure_signal=FAILURE_SIGNAL,
        parent_agent=parent,
        category="devops",  # different from Phase 1 to avoid cache collision
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
    print("SEE E2E PHASE 2 RESULT")
    print("=" * 70)
    with open("/tmp/see_e2e_phase2_result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[result saved to /tmp/see_e2e_phase2_result.json]")
    print()

    best_patch = result.get("best_patch") or {}
    cost = result.get("cost") or {}
    print(f"skill_name:              {result.get('skill_name', '?')}")
    print(f"iterations:              {result.get('iterations', 0)}")
    print(f"nodes_explored:          {result.get('nodes_explored', 0)}")
    print(f"hypotheses:              {len(result.get('hypotheses', []))}")
    print(f"budget_exceeded:         {result.get('budget_exceeded', False)}")
    print(f"elapsed_s:               {result.get('elapsed_s', 0):.1f}")
    if cost:
        print(f"cost.total_usd:         ${cost.get('cost_total_usd', 0):.4f}")
    if best_patch:
        print(f"best_patch.score:       {best_patch.get('evidence_score', 0):.3f}")
        print(f"best_patch.depth:       {best_patch.get('depth', 0)}")
        print(f"best_patch.node_id:     {best_patch.get('node_id', '?')}")
        print(f"best_patch.patches:     {len(best_patch.get('patches', []))}")
    else:
        print("best_patch:             (none)")
    if result.get("error"):
        print(f"error:                  {result.get('error')}")
    print()

    # Heuristic: does the patched skill now check for lftp?
    patch_preview = (best_patch.get("patched_skill_preview", "") or "").lower()
    patches_list = best_patch.get("patches", []) or []
    all_patch_text = patch_preview + " " + " ".join(
        p.get("new_string", "") + " " + p.get("old_string", "")
        for p in patches_list
    ).lower()

    checks = ["lftp" in all_patch_text and
              ("which" in all_patch_text or "command -v" in all_patch_text or
               "installed" in all_patch_text or "not found" in all_patch_text or
               "prerequisite" in all_patch_text or "pre-flight" in all_patch_text or
               "check" in all_patch_text)]
    if checks[0]:
        print("✓ PASS: patched skill now references lftp installation/availability check")
    else:
        print("⚠ REVIEW: patched skill does NOT clearly add an lftp check")
        print("  (inspect /tmp/see_e2e_phase2_result.json)")

    # Critical-link verification (same as Phase 1): with the corrected
    # PROMPT_TEST_GEN, a FUNCTIONAL patch (lftp pre-flight guard) should score > 0.0.
    best_score = float(best_patch.get("evidence_score", 0))
    if best_score > 0.0:
        print(f"✓ CRITICAL-LINK PASS: best node scored {best_score:.3f} > 0.0 — the "
              "test synthesis rewarded a functional patch (inversion fixed).")
    else:
        print(f"⚠ CRITICAL-LINK REVIEW: best node scored {best_score:.3f} — the "
              "engine still failed to reward the patch; inspect evidence_matrix "
              "in /tmp/see_e2e_phase2_result.json for test-by-test reasons.")

    # Print the patched preview
    if best_patch and best_patch.get("patched_skill_preview"):
        print()
        print("--- patched_skill_preview ---")
        print(best_patch["patched_skill_preview"])

    close = getattr(parent, "close", None)
    if callable(close):
        close()


if __name__ == "__main__":
    main()
