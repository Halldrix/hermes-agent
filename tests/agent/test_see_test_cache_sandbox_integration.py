"""SEE — Integration test: cache + sandbox + validator.

End-to-end test of the critical link in the SEE pipeline:
1. Generate a test (simulates what the expensive model produces).
2. Store it in the cache.
3. Retrieve from cache (hit).
4. Run it in the sandbox against a real output.
5. Invalidate on skill_version_hash change.
6. Archive after refutation.

No LLM or AIAgent required — only the prototype modules.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from agent.skill_evolution_sandbox import run_test_sandboxed, validate_test_strict
from agent.skill_evolution_test_cache import TestCache, TestCase

# Test code that a premium model would generate for hypothesis:
# "Skill does not verify gh CLI is installed before using it."
# Test semantics (aligned with the corrected PROMPT_TEST_GEN):
#   pass=True  <=> defect RESOLVED (guard in files['SKILL.md'] OR clean abort signal OR PR created)
#   pass=False <=> defect still present (gh not found in output, no guard)
TEST_CODE_GH_CHECK = '''import re

def check(output: str, files: dict) -> dict:
    """
    Regression test for the FIX: assert gh availability is pre-validated.
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
            return {"pass": True, "reason": "tier1: guard precedes gh invocation", "category": "hard"}
        if has_guard:
            return {"pass": True, "reason": "tier1: guard present in SKILL.md", "category": "hard"}

    # Tier 2: dynamic absence — defect signature still in output
    if output and re.search(r"gh.*not found|command not found.*gh", output, re.IGNORECASE):
        return {"pass": False, "reason": "tier2: defect signature still in output", "category": "hard"}

    # Tier 3: resolution signal — clean abort or successful PR creation
    if output:
        if re.search(r"pre-flight|pre-validate|not installed.*abort|aborting.*install", output, re.IGNORECASE):
            return {"pass": True, "reason": "tier3: skill pre-validated and aborted cleanly", "category": "hard"}
        if re.search(r"PR #\\d+|pull request.*created", output, re.IGNORECASE):
            return {"pass": True, "reason": "tier3: PR created — gh works", "category": "semantic"}

    return {"pass": False, "reason": "no evidence of fix", "category": "semantic"}
'''

# A patched SKILL.md that adds a pre-flight gh guard (the corrective construct)
PATCHED_SKILL_WITH_GUARD = (
    "---\nname: github-pr-create\n---\n\n"
    "# Create PR\n\n"
    "0. Pre-flight: run `command -v gh` — if missing, abort with install guidance.\n"
    "1. Run `gh pr create --title \"...\" --body \"...\"` to create a PR.\n"
)


@pytest.fixture
def tmp_cache_dir():
    tmp = tempfile.mkdtemp(prefix="see_smoke_")
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


def test_strict_validator_accepts_good_test() -> None:
    """validate_test_strict accepts a well-formed test with required signature."""
    ok, reason = validate_test_strict(TEST_CODE_GH_CHECK)
    assert ok, f"good test should pass validation: {reason}"


def test_strict_validator_rejects_trivial_test() -> None:
    """validate_test_strict rejects a trivial test that always passes."""
    trivial = "def check(output, files):\n    return {'pass': True}\n"
    ok, reason = validate_test_strict(trivial)
    assert not ok, "trivial test should be rejected"
    assert reason


def test_cache_store_and_retrieve(tmp_cache_dir: str) -> None:
    """Cache stores a test and retrieves it on a hit."""
    cache = TestCache(
        "github", "pr-workflow", base_dir=Path(tmp_cache_dir)
    )
    skill_hash = "abc123def456"

    test = TestCase(
        test_id="H1_abc12345",
        hypothesis_id="H1",
        hypothesis_description="Skill does not verify gh CLI is installed",
        observable_behavior="output contains 'command not found'",
        code=TEST_CODE_GH_CHECK,
        category="hard",
        validated=True,
    )
    cache.store_test(test, skill_hash=skill_hash)

    # Retrieve
    cached = cache.get_cached_test(
        hypothesis_desc=test.hypothesis_description,
        skill_hash=skill_hash,
        observable_behavior=test.observable_behavior,
    )
    assert cached is not None, "cache should hit on same hypothesis + skill"
    assert cached.code == TEST_CODE_GH_CHECK
    assert cached.test_id == "H1_abc12345"


def test_cache_miss_on_different_skill_hash(tmp_cache_dir: str) -> None:
    """Cache miss when skill hash changes (invalidation by skill_version_hash)."""
    cache = TestCache(
        "github", "pr-workflow", base_dir=Path(tmp_cache_dir)
    )
    test = TestCase(
        test_id="H1_abc12345",
        hypothesis_id="H1",
        hypothesis_description="Skill does not verify gh CLI",
        observable_behavior="output contains 'command not found'",
        code=TEST_CODE_GH_CHECK,
        category="hard",
        validated=True,
    )
    cache.store_test(test, skill_hash="hash_v1")

    # Different skill hash -> miss
    cached = cache.get_cached_test(
        hypothesis_desc=test.hypothesis_description,
        skill_hash="hash_v2_DIFFERENT",
        observable_behavior=test.observable_behavior,
    )
    assert cached is None, "should miss on different skill hash"


def test_sandbox_runs_cached_test_correctly(tmp_cache_dir: str) -> None:
    """Sandbox runs a cached test against the FIX (guard in patched SKILL.md) → pass=True.

    FIX-detection semantics: with the corrected prompt, a test passes when the
    defect is RESOLVED. We feed the patched SKILL.md (with the guard) via `files`
    so Tier-1 static evidence triggers pass=True.
    """
    cache = TestCache(
        "github", "pr-workflow", base_dir=Path(tmp_cache_dir)
    )
    skill_hash = "abc123def456"

    test = TestCase(
        test_id="H1_abc12345",
        hypothesis_id="H1",
        hypothesis_description="Skill does not verify gh CLI",
        observable_behavior="output contains 'command not found'",
        code=TEST_CODE_GH_CHECK,
        category="hard",
        validated=True,
    )
    cache.store_test(test, skill_hash=skill_hash)

    # Retrieve and run against the FIX (patched SKILL.md provided in files)
    cached = cache.get_cached_test(
        hypothesis_desc=test.hypothesis_description,
        skill_hash=skill_hash,
        observable_behavior=test.observable_behavior,
    )
    assert cached is not None

    files = {"SKILL.md": PATCHED_SKILL_WITH_GUARD}
    result = run_test_sandboxed(cached.code, "", files, timeout_s=5.0)
    assert result.passed, f"test should pass when the fix (guard) is present: {result.reason}"
    assert result.category == "hard"


def test_sandbox_rejects_when_defect_present(tmp_cache_dir: str) -> None:
    """Sandbox test fails when the defect is still present (no guard, gh not found).

    FIX-detection semantics: with the corrected prompt, feeding the unpatched
    failure output (gh: command not found) with no guard in files → pass=False.
    """
    result = run_test_sandboxed(
        TEST_CODE_GH_CHECK, "gh: command not found", {}, timeout_s=5.0
    )
    assert not result.passed, "test should fail when defect still present (no fix)"


def test_archive_refuted_test(tmp_cache_dir: str) -> None:
    """Archive moves a refuted test to the archive for audit trail."""
    cache = TestCache(
        "github", "pr-workflow", base_dir=Path(tmp_cache_dir)
    )
    skill_hash = "abc123def456"

    test = TestCase(
        test_id="H1_abc12345",
        hypothesis_id="H1",
        hypothesis_description="Skill does not verify gh CLI",
        observable_behavior="output contains 'command not found'",
        code=TEST_CODE_GH_CHECK,
        category="hard",
        validated=True,
    )
    cache.store_test(test, skill_hash=skill_hash)
    cache.archive_refuted_tests("H1")

    # Archive should exist
    archive_path = Path(tmp_cache_dir) / "skills" / "github" / "pr-workflow" / ".evolution_cache" / "tests" / "tests_archive.json"
    assert archive_path.exists(), f"archive should exist at {archive_path}"


def test_cache_hit_rate_tracking(tmp_cache_dir: str) -> None:
    """Cache hit_rate reflects hits and misses."""
    cache = TestCache(
        "github", "pr-workflow", base_dir=Path(tmp_cache_dir)
    )
    skill_hash = "abc123def456"

    test = TestCase(
        test_id="H1_abc12345",
        hypothesis_id="H1",
        hypothesis_description="Skill does not verify gh CLI",
        observable_behavior="output contains 'command not found'",
        code=TEST_CODE_GH_CHECK,
        category="hard",
        validated=True,
    )
    cache.store_test(test, skill_hash=skill_hash)

    # Hit
    cache.get_cached_test(
        hypothesis_desc=test.hypothesis_description,
        skill_hash=skill_hash,
        observable_behavior=test.observable_behavior,
    )
    # Miss (different skill hash)
    cache.get_cached_test(
        hypothesis_desc=test.hypothesis_description,
        skill_hash="DIFFERENT_HASH",
        observable_behavior=test.observable_behavior,
    )

    rate = cache.hit_rate
    assert 0.0 < rate < 1.0, f"hit_rate should be between 0 and 1: {rate}"


def test_prune_expired_tests(tmp_cache_dir: str) -> None:
    """Prune removes tests older than TTL."""
    cache = TestCache(
        "github", "pr-workflow", base_dir=Path(tmp_cache_dir)
    )
    skill_hash = "abc123def456"

    test = TestCase(
        test_id="H1_abc12345",
        hypothesis_id="H1",
        hypothesis_description="Skill does not verify gh CLI",
        observable_behavior="output contains 'command not found'",
        code=TEST_CODE_GH_CHECK,
        category="hard",
        validated=True,
    )
    cache.store_test(test, skill_hash=skill_hash)
    # Prune with TTL of 0 days should remove everything
    pruned = cache.prune_expired_tests(ttl_days=0)
    assert pruned >= 1, f"should prune at least 1 test: {pruned}"


def test_invalidate_stale_tests(tmp_cache_dir: str) -> None:
    """Invalidate stale tests returns list of invalidated test IDs."""
    cache = TestCache(
        "github", "pr-workflow", base_dir=Path(tmp_cache_dir)
    )
    skill_hash_v1 = "abc123def456"

    test = TestCase(
        test_id="H1_abc12345",
        hypothesis_id="H1",
        hypothesis_description="Skill does not verify gh CLI",
        observable_behavior="output contains 'command not found'",
        code=TEST_CODE_GH_CHECK,
        category="hard",
        validated=True,
    )
    cache.store_test(test, skill_hash=skill_hash_v1)

    # Invalidate with a different current skill hash
    invalidated = cache.invalidate_stale_tests(current_skill_hash="DIFFERENT_HASH")
    assert isinstance(invalidated, list)


def test_end_to_end_critical_link(tmp_cache_dir: str) -> None:
    """Full end-to-end: generate → store → retrieve → run → verify (FIX-detection).

    With the corrected prompt semantics, the critical link verifies that the
    test passes when the FIX is present (patched SKILL.md with guard) and fails
    when the defect is still present (gh not found, no guard).
    """
    cache = TestCache(
        "github", "pr-workflow", base_dir=Path(tmp_cache_dir)
    )
    skill_hash = "abc123def456"

    # 1. Validate test code
    ok, reason = validate_test_strict(TEST_CODE_GH_CHECK)
    assert ok, f"validation failed: {reason}"

    # 2. Store test
    test = TestCase(
        test_id="H1_abc12345",
        hypothesis_id="H1",
        hypothesis_description="Skill does not verify gh CLI",
        observable_behavior="output contains 'command not found'",
        code=TEST_CODE_GH_CHECK,
        category="hard",
        validated=True,
    )
    cache.store_test(test, skill_hash=skill_hash)

    # 3. Retrieve (hit)
    cached = cache.get_cached_test(
        hypothesis_desc=test.hypothesis_description,
        skill_hash=skill_hash,
        observable_behavior=test.observable_behavior,
    )
    assert cached is not None, "cache hit failed"

    # 4. Run against the FIX (patched SKILL.md via files) → must pass
    result = run_test_sandboxed(
        cached.code, "", {"SKILL.md": PATCHED_SKILL_WITH_GUARD}, timeout_s=5.0
    )
    assert result.passed, f"sandbox test should pass when fix is present: {result.reason}"
    assert result.category == "hard"

    # 5. Run against the DEFECT (no fix, gh missing) → must fail
    result_defect = run_test_sandboxed(
        cached.code, "gh: command not found", {}, timeout_s=5.0
    )
    assert not result_defect.passed, "test should fail when defect still present"
