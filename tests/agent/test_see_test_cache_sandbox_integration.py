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
TEST_CODE_GH_CHECK = '''import re

def check(output: str, files: dict) -> dict:
    """
    Verifies that the output indicates gh is not installed.
    Hypothesis H1: the skill should have verified gh before using it.
    """
    if not output:
        return {"pass": False, "reason": "empty output", "category": "hard"}

    if re.search(r"gh.*not found|command not found.*gh", output, re.IGNORECASE):
        return {"pass": True, "reason": "output confirms gh missing", "category": "hard"}

    if re.search(r"PR #\\d+|pull request.*created", output, re.IGNORECASE):
        return {"pass": False, "reason": "output shows PR created — gh works", "category": "semantic"}

    return {"pass": False, "reason": "ambiguous output", "category": "semantic"}
'''


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
    """Sandbox runs a cached test against real output — gh missing -> pass."""
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

    # Retrieve and run
    cached = cache.get_cached_test(
        hypothesis_desc=test.hypothesis_description,
        skill_hash=skill_hash,
        observable_behavior=test.observable_behavior,
    )
    assert cached is not None

    output_gh_missing = "gh: command not found"
    result = run_test_sandboxed(cached.code, output_gh_missing, {}, timeout_s=5.0)
    assert result.passed, f"test should pass for gh missing output: {result.reason}"
    assert result.category == "hard"


def test_sandbox_rejects_when_gh_works(tmp_cache_dir: str) -> None:
    """Sandbox test fails when gh works (PR created) — hypothesis refuted."""
    result = run_test_sandboxed(
        TEST_CODE_GH_CHECK, "PR #42 created successfully", {}, timeout_s=5.0
    )
    assert not result.passed, "test should fail when gh works (PR created)"


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
    """Full end-to-end: generate → store → retrieve → run → verify."""
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

    # 4. Run in sandbox against real output
    result = run_test_sandboxed(cached.code, "gh: command not found", {}, timeout_s=5.0)
    assert result.passed, f"sandbox test should pass: {result.reason}"
    assert result.category == "hard"
