#!/usr/bin/env python3
"""
SEE Prototype — End-to-end test of the critical link.

Valida que:
1. The sandbox runs valid tests and rejects invalid ones.
2. _validate_test_strict detecta tests triviales (always-true).
3. El sandbox detecta imports prohibidos.
4. The sandbox runs real tests that an expensive model would generate.
5. El bloqueo de tests siempre-true funciona con scenarios contrastantes.

No requiere LLM ni AIAgent — solo el sandbox puro.
"""
import sys
import os

# Import the sandbox from the prototype
from agent.skill_evolution_sandbox import (
    validate_test,
    validate_test_strict,
    run_test_sandboxed,
    TestResult,
)

# ── Test 1: Valid test that an expensive model would generate ─────────────────

# Simulates a real test for the hypothesis:
# "The skill does not verify that the gh command is installed before using it"
# Test semantics (aligned with the corrected PROMPT_TEST_GEN):
#   pass=True  <=> the DEFECT is RESOLVED (guard present in files['SKILL.md'],
#                  OR output shows clean abort, OR output shows PR created — gh works)
#   pass=False <=> the DEFECT is still present (output shows "command not found" with no guard)
TEST_VALID_GH_CHECK = '''import json
import re

def check(output: str, files: dict) -> dict:
    """
    Regression test for the FIX: assert that gh availability is pre-validated.
    pass=True when the defect (blind gh invocation) is RESOLVED.
    """
    # Tier 1 (preferred): static artifact evidence — guard present in patched SKILL.md
    skill_content = None
    if files:
        for name, content in files.items():
            if not isinstance(name, str) or not isinstance(content, str):
                continue
            if name.upper().endswith("SKILL.MD"):
                skill_content = content
                break
    if skill_content is not None:
        guard_re = r"command\\s+-v\\s+gh|which\\s+gh|type\\s+-p\\s+gh|hash\\s+gh"
        has_guard = re.search(guard_re, skill_content, re.IGNORECASE) is not None
        gh_call_idx = skill_content.find("gh pr create")
        guard_idx = min(
            (m.start() for m in re.finditer(guard_re, skill_content, re.IGNORECASE)),
            default=-1,
        )
        ordered_before = guard_idx != -1 and gh_call_idx != -1 and guard_idx < gh_call_idx
        if has_guard and ordered_before:
            return {"pass": True, "reason": "tier1: pre-flight gh guard precedes gh invocation in SKILL.md", "category": "hard"}
        if has_guard:
            return {"pass": True, "reason": "tier1: pre-flight gh guard present in SKILL.md", "category": "hard"}

    # Tier 2: dynamic absence — defect signature absent from the freshly executed output
    if output and re.search(r"gh.*not found|command not found.*gh", output, re.IGNORECASE):
        return {"pass": False, "reason": "tier2: defect signature still in output (gh not found)", "category": "hard"}

    # Tier 3: resolution signal — clean abort or successful PR creation
    if output:
        if re.search(r"pre-flight|pre-validate|not installed.*abort|aborting.*install", output, re.IGNORECASE):
            return {"pass": True, "reason": "tier3: skill pre-validated gh and aborted cleanly", "category": "hard"}
        if re.search(r"PR #\\d+|pull request.*created", output, re.IGNORECASE):
            return {"pass": True, "reason": "tier3: PR created — gh works", "category": "semantic"}

    # No evidence of resolution
    return {"pass": False, "reason": "no evidence of fix (no guard in SKILL.md, no resolution signal in output)", "category": "semantic"}
'''

# Simulates a trivial (always-true) test that a cheap model might generate
TEST_TRIVIAL_ALWAYS_TRUE = '''def check(output: str, files: dict) -> dict:
    return {"pass": True, "reason": "always passes", "category": "semantic"}
'''

# Simulates a test with a forbidden import
TEST_FORBIDDEN_IMPORT = '''import subprocess

def check(output: str, files: dict) -> dict:
    result = subprocess.run(["gh", "auth", "status"], capture_output=True)
    return {"pass": result.returncode == 0, "reason": "checked gh", "category": "hard"}
'''

# Well-made test that passes on plausible but fails on garbage
TEST_GOOD_SEMANTIC = '''import re

def check(output: str, files: dict) -> dict:
    """
    Verify that the output contains a valid PR number.
    """
    if re.search(r"PR #\\d+", output):
        return {"pass": True, "reason": "PR number found", "category": "semantic"}
    return {"pass": False, "reason": "no PR number in output", "category": "semantic"}
'''


def test_rejects_trivial_always_true():
    """El validador reforzado debe rechazar tests que pasan en todo."""
    ok, reason = validate_test_strict(TEST_TRIVIAL_ALWAYS_TRUE)
    assert not ok, f"should reject always-true test, got: {reason}"
    assert "trivially" in reason.lower(), f"reason should mention trivial: {reason}"
    print("✓ test_rejects_trivial_always_true passed")


def test_rejects_forbidden_import():
    """Basic validator must reject subprocess."""
    v = validate_test(TEST_FORBIDDEN_IMPORT)
    assert not v.valid, f"should reject subprocess: {v.reason}"
    assert "forbidden" in v.reason.lower() or "subprocess" in v.reason
    print("✓ test_rejects_forbidden_import passed")


def test_accepts_valid_test():
    """El validador reforzado debe aceptar el test de gh_check."""
    ok, reason = validate_test_strict(TEST_VALID_GH_CHECK)
    assert ok, f"should accept valid test: {reason}"
    print("✓ test_accepts_valid_test passed")


def test_sandbox_executes_valid_test_pass_case():
    """Sandbox executes the test and detects the PASS case (FIX present).

    FIX-detection semantics: when the patched skill adds a pre-flight guard,
    we pass the patched SKILL.md in `files`, and the test must return pass=True
    because the defect (blind gh invocation) is RESOLVED.
    """
    patched_skill = (
        "name: github-pr-create\n"
        "# Create PR\n"
        "1. Run `command -v gh` first; abort if not found.\n"
        "2. Run `gh pr create --title \"...\" --body \"...\"`.\n"
    )
    output_no_error = "Pre-flight check: gh CLI not installed. Aborting."
    files = {"SKILL.md": patched_skill}
    result = run_test_sandboxed(
        TEST_VALID_GH_CHECK, output_no_error, files, timeout_s=5.0
    )
    assert result.passed is True, f"should pass (guard present in SKILL.md): {result.reason}"
    assert result.category == "hard"
    print(f"✓ test_sandbox_executes_valid_test_pass_case passed ({result.runtime_s:.3f}s)")


def test_sandbox_executes_valid_test_fail_case():
    """Sandbox executes the test and detects the FAIL case (defect still present).

    FIX-detection semantics: when the unpatched skill still invokes gh blindly,
    the output shows "command not found" with no guard in SKILL.md → pass=False.
    """
    output_gh_missing = (
        "Running gh pr create...\n"
        "gh: command not found\n"
        "Error: gh CLI is required but not installed."
    )
    # No files dict → no static evidence; output shows the defect → pass=False.
    result = run_test_sandboxed(
        TEST_VALID_GH_CHECK, output_gh_missing, {}, timeout_s=5.0
    )
    assert result.passed is False, f"should fail (defect present, no guard): {result.reason}"
    assert result.category == "hard"
    print(f"✓ test_sandbox_executes_valid_test_fail_case passed ({result.runtime_s:.3f}s)")


def test_sandbox_valid_test_passes_on_pr_created():
    """A successful PR-creation execution also resolves the defect → pass=True."""
    output_success = "PR #42 created successfully\nhttps://github.com/org/repo/pull/42"
    result = run_test_sandboxed(
        TEST_VALID_GH_CHECK, output_success, {}, timeout_s=5.0
    )
    assert result.passed is True, f"should pass (PR created = gh works): {result.reason}"
    print(f"✓ test_sandbox_valid_test_passes_on_pr_created passed ({result.runtime_s:.3f}s)")


def test_good_semantic_passes_validation():
    """Well-made semantic test passes reinforced validation."""
    ok, reason = validate_test_strict(TEST_GOOD_SEMANTIC)
    assert ok, f"should accept good semantic test: {reason}"
    print("✓ test_good_semantic_passes_validation passed")


def test_good_semantic_runs_correctly_on_plausible():
    """Good semantic test passes when output has PR#."""
    result = run_test_sandboxed(
        TEST_GOOD_SEMANTIC,
        "PR #42 created at https://github.com/org/repo/pull/42",
        {},
        timeout_s=5.0,
    )
    assert result.passed is True
    print(f"✓ test_good_semantic_runs_correctly_on_plausible passed ({result.runtime_s:.3f}s)")


def test_good_semantic_fails_on_garbage():
    """Good semantic test fails when output is garbage."""
    result = run_test_sandboxed(
        TEST_GOOD_SEMANTIC,
        "GARBAGE_TEST_INPUT_12345",
        {},
        timeout_s=5.0,
    )
    assert result.passed is False
    print(f"✓ test_good_semantic_fails_on_garbage passed ({result.runtime_s:.3f}s)")


def test_sandbox_handles_test_exception():
    """A test that raises an exception returns a TestResult with error."""
    TEST_WITH_BUG = '''def check(output: str, files: dict) -> dict:
    # Divide by zero bug
    x = 1 / 0
    return {"pass": True, "reason": "unreached", "category": "hard"}
'''
    result = run_test_sandboxed(TEST_WITH_BUG, "any", {}, timeout_s=5.0)
    assert result.passed is False
    assert result.error == "runtime_error" or "division" in result.reason.lower() or "zero" in result.reason.lower()
    print(f"✓ test_sandbox_handles_test_exception passed ({result.reason[:60]})")


def test_sandbox_handles_test_timeout():
    """Un test que se cuelga se aborta por timeout."""
    TEST_HANG = '''import time

def check(output: str, files: dict) -> dict:
    time.sleep(10)  # cuelga
    return {"pass": True, "reason": "unreached", "category": "hard"}
'''
    # time.sleep is not in the allowed builtins, so this test
    # will actually fail with NameError, not timeout. We adjust:
    TEST_HANG_ALT = '''def check(output: str, files: dict) -> dict:
    sum = 0
    for i in range(100000000):
        sum += i
    return {"pass": sum > 0, "reason": "slow", "category": "hard"}
'''
    result = run_test_sandboxed(TEST_HANG_ALT, "any", {}, timeout_s=0.5)
    # Might finish fast (Python optimizes) or timeout
    assert result.passed is False or result.runtime_s < 1.0
    print(f"✓ test_sandbox_handles_test_timeout passed ({result.runtime_s:.3f}s, error={result.error})")


def test_rejects_empty_code():
    """Validator rejects empty code."""
    v = validate_test("")
    assert not v.valid
    print("✓ test_rejects_empty_code passed")


def test_rejects_no_signature():
    """Validador rechaza tests sin def check()."""
    v = validate_test("def foo(): pass")
    assert not v.valid
    assert "check" in v.reason.lower()
    print("✓ test_rejects_no_signature passed")


def test_rejects_no_dict_return():
    """Validador rechaza tests que no retornan dict con pass."""
    TEST_NO_DICT = '''def check(output: str, files: dict) -> dict:
    return True
'''
    v = validate_test(TEST_NO_DICT)
    # El regex busca 'return {"pass": ...}'
    # Este test retorna True, no dict
    assert not v.valid
    assert "pass" in v.reason.lower() or "dict" in v.reason.lower()
    print("✓ test_rejects_no_dict_return passed")


def test_files_passed_to_test():
    """The sandbox passes files dict to the test. The test must be designed
    para no fallar trivialmente en los 3 escenarios contrastantes."""
    TEST_USES_FILES = '''def check(output: str, files: dict) -> dict:
    # Si no hay files, no opinamos — pasamos neutral
    if not files:
        return {"pass": True, "reason": "no files to check (neutral)", "category": "hard"}
    if "output.json" in files:
        return {"pass": "PR" in files["output.json"], "reason": "checked file", "category": "hard"}
    return {"pass": False, "reason": f"unexpected files: {list(files.keys())}", "category": "hard"}
'''
    result = run_test_sandboxed(
        TEST_USES_FILES,
        "any",
        {"output.json": '{"PR": "42"}'},
        timeout_s=5.0,
    )
    assert result.passed is True, f"should pass with PR in file: {result.reason}"
    print(f"✓ test_files_passed_to_test passed ({result.runtime_s:.3f}s)")


if __name__ == "__main__":
    test_rejects_trivial_always_true()
    test_rejects_forbidden_import()
    test_accepts_valid_test()
    test_sandbox_executes_valid_test_pass_case()
    test_sandbox_executes_valid_test_fail_case()
    test_sandbox_valid_test_passes_on_pr_created()
    test_good_semantic_passes_validation()
    test_good_semantic_runs_correctly_on_plausible()
    test_good_semantic_fails_on_garbage()
    test_sandbox_handles_test_exception()
    test_sandbox_handles_test_timeout()
    test_rejects_empty_code()
    test_rejects_no_signature()
    test_rejects_no_dict_return()
    test_files_passed_to_test()
    print("\n✅ All sandbox E2E tests passed")
